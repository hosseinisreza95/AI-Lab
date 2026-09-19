"""The logistics assistant.

All five tools are read-only, so there is no confirmation gate here and no
checkpointer: the agent cannot change anything, and a question about where a
truck is should not need a conversation to answer.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from rag.tools import ALL_TOOLS

load_dotenv()

MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are the logistics assistant used by supply chain teams \
across a retail network of warehouses and stores. Today is {today}.

You answer two kinds of question, and most real questions are both at once:

- Live operational state: where a shipment is, what is late into a store, how a \
carrier or lane is performing, which stores are about to run out. These come from \
the shipment and planning tools.
- What the procedures say: escalation tiers, claim eligibility, cover targets, \
what a status actually means. These come from search_logistics_procedures.

Rules:
- Always use a tool. Never state a location, an ETA, an on-time rate or a policy \
threshold from memory.
- When a shipment is late or projected to be late, look up the escalation \
procedure and say which tier applies and who has to act. A delay figure without \
the resulting action is half an answer.
- Three different numbers get called the ETA. Be explicit about which one you are \
quoting: the promised date is what the store was told and what performance is \
measured against; the predicted arrival is the model's estimate and is a forecast, \
not a commitment. Never present a predicted arrival to a store as a promise.
- Report the prediction source when you give a predicted arrival.
- If a lane has too few shipments to measure, say so rather than quoting its \
on-time rate.
- If the procedures do not cover something, say so and say who to escalate to. Do \
not fill the gap from general logistics knowledge.
- Be brief. These users are working an exception queue, not reading a report."""


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph():
    llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools(ALL_TOOLS)

    def assistant(state: State) -> dict:
        system = {
            "role": "system",
            "content": SYSTEM_PROMPT.format(today=dt.date.today().strftime("%A %d %B %Y")),
        }
        return {"messages": [llm.invoke([system, *state["messages"]])]}

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools=ALL_TOOLS))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "assistant")
    return builder.compile()


def ask(question: str, graph=None) -> str:
    graph = graph or build_graph()
    state = graph.invoke({"messages": [HumanMessage(content=question)]})
    return state["messages"][-1].content


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "what is late into the Spanish stores right now?"
    print(ask(query))
