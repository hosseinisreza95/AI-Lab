"""FastAPI service: planning views plus the logistics assistant."""
import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from rag.agent import ask, build_graph
from rag.tools import (
    CURATED_DIR,
    MODEL_DIR,
    find_shipments,
    get_lane_performance,
    get_store_cover,
    track_shipment,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Retail Supply Chain - Forecasting and Logistics RAG",
    description="Delivery-time forecasting, store cover planning views, and a "
                "natural-language logistics assistant.",
    version="1.0.0",
)

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["where is order ORD-481923?"])


class AskResponse(BaseModel):
    question: str
    answer: str
    latency_ms: int


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    logger.info(
        "Path: %s | Method: %s | Status: %s | Time: %.4fs",
        request.url.path, request.method, response.status_code, time.time() - start,
    )
    return response


@app.get("/health")
async def health() -> dict:
    """Readiness: reports which pipeline artifacts exist.

    A pod that is up but has no curated data cannot answer anything, so the probe
    reports artifact state rather than just process liveness.
    """
    artifacts = {
        "shipment_features": (CURATED_DIR / "shipment_features.parquet").exists(),
        "store_cover": (CURATED_DIR / "store_cover.parquet").exists(),
        "lane_performance": (CURATED_DIR / "lane_performance.parquet").exists(),
        "delivery_time_model": (MODEL_DIR / "delivery_time_model.joblib").exists(),
    }
    return {
        "status": "ok" if all(artifacts.values()) else "degraded",
        "artifacts": artifacts,
    }


@app.get("/model/metrics")
async def model_metrics() -> dict:
    """The delivery-time model's evaluation, as written by the training job."""
    path = MODEL_DIR / "delivery_time_metrics.json"
    if not path.exists():
        raise HTTPException(status_code=409, detail="Model has not been trained yet.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/shipments/{identifier}")
async def shipment(identifier: str) -> dict:
    return json.loads(track_shipment.invoke({"identifier": identifier}))


@app.get("/shipments")
async def shipments(store_or_region: str = "", status: str = "", late_only: bool = False) -> dict:
    return json.loads(find_shipments.invoke({
        "store_or_region": store_or_region, "status": status, "late_only": late_only
    }))


@app.get("/lanes")
async def lanes(lane_or_carrier: str = "", worst: bool = True) -> dict:
    return json.loads(get_lane_performance.invoke({
        "lane_or_carrier": lane_or_carrier, "worst": worst
    }))


@app.get("/cover")
async def cover(store_or_region: str = "", at_risk_only: bool = True) -> dict:
    return json.loads(get_store_cover.invoke({
        "store_or_region": store_or_region, "at_risk_only": at_risk_only
    }))


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest) -> AskResponse:
    """Natural-language logistics question."""
    start = time.time()
    logger.info("Question: %s", request.question)
    try:
        answer = ask(request.question, graph=get_graph())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Assistant failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        question=request.question,
        answer=answer,
        latency_ms=int((time.time() - start) * 1000),
    )
