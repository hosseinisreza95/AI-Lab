"""FastAPI backend consumed by the internal technician mobile app."""
import logging
import time

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ingestion.ingest import ingest
from rag.config import settings
from rag.orchestrator import RagOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Field Service Technician RAG Assistant",
    description="Retrieval-augmented maintenance assistant for field technicians.",
    version="1.0.0",
)

# Built lazily: the container has to start even when the index has not been built
# yet, so that /health and /ingest stay reachable.
_orchestrator: RagOrchestrator | None = None


def get_orchestrator() -> RagOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RagOrchestrator()
    return _orchestrator


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["The GC-200 keeps surging, E-121 then E-134"])
    equipment: str | None = Field(
        None,
        description="Equipment tag the technician is standing at, e.g. GC-200. "
                    "Scopes retrieval to that machine plus site-wide documents.",
        examples=["GC-200"],
    )


class Citation(BaseModel):
    marker: str
    title: str
    section: str
    source_type: str
    equipment: str
    revision: str
    fusion_score: float
    rerank_score: float | None = None
    matched_by: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    model: str
    reranked: bool
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    backend: str
    chat_model: str
    embedding_model: str
    index_ready: bool


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info(
        "Path: %s | Method: %s | Status: %s | Time: %.4fs",
        request.url.path, request.method, response.status_code, elapsed,
    )
    return response


@app.get("/", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check. Also reports whether the vector index has been built."""
    try:
        get_orchestrator()
        index_ready = True
    except Exception:
        index_ready = False
    return HealthResponse(
        status="ok",
        backend="azure-openai" if settings.use_azure else "openai",
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        index_ready=index_ready,
    )


@app.post("/ingest")
async def rebuild_index() -> dict:
    """Re-chunk and re-embed the knowledge base. Called after documentation changes."""
    global _orchestrator
    try:
        stats = ingest(reset=True)
        _orchestrator = None  # Force a rebuild against the new index.
        logger.info("Index rebuilt: %s", stats)
        return {"status": "ok", **stats}
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """Answer a technician question from the indexed documentation."""
    logger.info("Question: %s | equipment=%s", request.question, request.equipment)
    start = time.time()
    try:
        result = get_orchestrator().ask(request.question, equipment=request.equipment)
    except RuntimeError as exc:
        # Index missing is a caller-fixable state, not a server fault.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Answer generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        question=result.question,
        answer=result.answer,
        citations=[Citation(**citation) for citation in result.citations],
        model=result.model,
        reranked=result.reranked,
        latency_ms=int((time.time() - start) * 1000),
    )
