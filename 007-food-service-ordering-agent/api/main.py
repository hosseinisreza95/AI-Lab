"""FastAPI backend for the restaurant management app."""
import logging
import time

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from agent.graph import build_graph, chat, confirm_send
from database.models import init_db
from forecasting.inventory import build_alerts
from forecasting.model import backtest, forecast_covers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Food Service Demand and Ordering Agent",
    description="Covers forecasting, stock-out alerts, and an agent that places supplier orders.",
    version="1.0.0",
)

# One compiled graph for the process, so the in-memory checkpointer keeps threads
# alive between the draft and the confirmation.
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, examples=["order three cases of still water"])
    thread_id: str = Field("default", description="Conversation thread. Reuse it to confirm an order.")


class ChatResponse(BaseModel):
    status: str
    response: str
    thread_id: str
    pending_action: str | None = None
    po_number: str | None = None
    latency_ms: int


class ConfirmRequest(BaseModel):
    thread_id: str = Field("default")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    logger.info(
        "Path: %s | Method: %s | Status: %s | Time: %.4fs",
        request.url.path, request.method, response.status_code, time.time() - start,
    )
    return response


@app.on_event("startup")
async def startup() -> None:
    init_db()


@app.get("/")
async def health() -> dict:
    return {"status": "ok", "message": "Food service agent API is running."}


@app.get("/forecast")
async def forecast(days: int = 14) -> dict:
    """Covers forecast for the next N days, closures included."""
    try:
        points = forecast_covers(horizon_days=max(1, min(days, 60)))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "horizon_days": days,
        "forecast": [
            {
                "date": point.service_date.isoformat(),
                "weekday": point.service_date.strftime("%A"),
                "predicted_covers": point.predicted_covers,
                "lower": point.lower,
                "upper": point.upper,
                "closed": point.is_closed,
                "note": point.note,
            }
            for point in points
        ],
    }


@app.get("/forecast/accuracy")
async def forecast_accuracy() -> dict:
    """Rolling-origin backtest against a seasonal-naive baseline."""
    return backtest().to_dict()


@app.get("/alerts")
async def alerts(horizon_days: int = 14, include_all: bool = False) -> dict:
    """Products forecast to run out inside the horizon, with order-by dates."""
    try:
        found = build_alerts(horizon_days=max(1, min(horizon_days, 60)), include_all=include_all)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "horizon_days": horizon_days,
        "alert_count": len(found),
        "alerts": [alert.to_dict() for alert in found],
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Talk to the agent. Reuse thread_id to confirm an order it has drafted."""
    start = time.time()
    logger.info("[%s] %s", request.thread_id, request.message)
    try:
        result = chat(request.message, thread_id=request.thread_id, graph=get_graph())
    except Exception as exc:
        logger.error("Agent failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(**result, latency_ms=int((time.time() - start) * 1000))


@app.post("/chat/confirm", response_model=ChatResponse)
async def confirm_endpoint(request: ConfirmRequest) -> ChatResponse:
    """Confirm a pending purchase order send on a thread.

    Separate from /chat on purpose: sending a supplier order should be an explicit
    call from the UI, not an inference about what the manager meant by "ok".
    """
    start = time.time()
    logger.info("[%s] confirm send", request.thread_id)
    try:
        result = confirm_send(thread_id=request.thread_id, graph=get_graph())
    except Exception as exc:
        logger.error("Confirmation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(**result, latency_ms=int((time.time() - start) * 1000))
