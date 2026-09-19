"""The LangGraph agent behind the in-app chatbot.

A ReAct-style loop with one addition that matters: the graph interrupts before
`send_purchase_order`. Everything else runs freely, but the one tool that emails
a real supplier stops the graph and hands control back, so the manager confirms
an order they have actually seen.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from agent.tools import ALL_TOOLS

load_dotenv()

MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
REQUIRE_CONFIRMATION = os.getenv("REQUIRE_ORDER_CONFIRMATION", "true").lower() in {
    "1", "true", "yes", "on"
}

SYSTEM_PROMPT = """You are the assistant inside the management app of a staff \
restaurant. You are talking to the site manager, who is busy and wants an answer, \
not a tour of the system.

Today is {today}.

What you can do:
- Answer questions about expected customer numbers from the forecast.
- Answer questions about stock levels.
- Tell the manager what they are about to run short of, and by when they must order.
- Prepare and send a purchase order to a registered supplier.

How to behave:
- Always call a tool for anything factual. Never state a stock level, a covers \
number or a supplier from memory, even if it was mentioned earlier in the \
conversation - stock moves between messages.
- Answer in one or two sentences plus a short list where a list helps. The manager \
is reading this on a phone between services.
- When you report a stock-out risk, give the order-by date, not just the date it \
runs out. The lead time is the part that makes the warning actionable.
- When the manager asks to order something, work out the quantity yourself from the \
alert or the par level, then create a DRAFT order and show them the supplier, the \
quantity, the total and the delivery day.
- Never call send_purchase_order until the manager has confirmed that specific \
order in their own words. "Order three cases of water" is a request to draft, not \
permission to send. If they have already said "yes, send it", that is permission.
- If a tool returns an error, say what went wrong in plain language and what you \
need from the manager. Do not retry the same call unchanged.
- If a request would need a supplier that is not registered, say so and stop. Do \
not invent contact details."""


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(require_confirmation: bool | None = None):
    """Compile the agent graph.

    `require_confirmation` interrupts before the tool node whenever the model has
    asked to send an order, which is how the human-in-the-loop step is enforced
    in the graph rather than in the prompt.
    """
    if require_confirmation is None:
        require_confirmation = REQUIRE_CONFIRMATION

    llm = ChatOpenAI(model=MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def chatbot(state: State) -> dict:
        system = {
            "role": "system",
            "content": SYSTEM_PROMPT.format(today=dt.date.today().strftime("%A %d %B %Y")),
        }
        return {"messages": [llm_with_tools.invoke([system, *state["messages"]])]}

    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(tools=ALL_TOOLS))

    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "chatbot")

    # A checkpointer is required to interrupt and resume, so it is always on.
    return builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["tools"] if require_confirmation else None,
    )


def _wants_to_send(state) -> str | None:
    """Return the PO number if the pending tool call is a send, else None."""
    messages = state.values.get("messages", [])
    if not messages:
        return None
    for call in getattr(messages[-1], "tool_calls", []) or []:
        if call["name"] == "send_purchase_order":
            return call["args"].get("po_number", "unknown")
    return None


def chat(message: str, thread_id: str = "default", graph=None, resume: bool = False) -> dict:
    """Run one turn.

    Returns the assistant reply, or a pending-confirmation marker when the agent
    is about to send a purchase order.
    """
    graph = graph or build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    state = (
        graph.invoke(None, config)
        if resume
        else graph.invoke({"messages": [HumanMessage(content=message)]}, config)
    )

    snapshot = graph.get_state(config)
    while snapshot.next:
        po_number = _wants_to_send(snapshot)
        if po_number:
            return {
                "status": "awaiting_confirmation",
                "pending_action": "send_purchase_order",
                "po_number": po_number,
                "response": (
                    f"Ready to email purchase order {po_number} to the supplier. "
                    f"Confirm and I will send it."
                ),
                "thread_id": thread_id,
            }
        # Any other interrupted tool is a read: let it through without asking.
        state = graph.invoke(None, config)
        snapshot = graph.get_state(config)

    return {
        "status": "ok",
        "response": state["messages"][-1].content,
        "thread_id": thread_id,
    }


def confirm_send(thread_id: str = "default", graph=None) -> dict:
    """Resume a thread that is waiting on an order confirmation."""
    graph = graph or build_graph()
    return chat("", thread_id=thread_id, graph=graph, resume=True)


agent_app = None  # Built lazily by the API so import does not require an API key.
