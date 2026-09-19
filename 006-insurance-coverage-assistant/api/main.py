"""FastAPI backend behind the advisor web app."""
import logging
import time

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ingestion.ingest import ingest
from rag.answer import CoverageAssistant, CoverageAnswer
from rag.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Insurance Coverage Assistant",
    description="Advisors describe a customer case; RAG returns the applicable clauses.",
    version="1.0.0",
)

_assistant: CoverageAssistant | None = None


def get_assistant() -> CoverageAssistant:
    global _assistant
    if _assistant is None:
        _assistant = CoverageAssistant()
    return _assistant


class CaseRequest(BaseModel):
    case: str = Field(
        ...,
        min_length=10,
        examples=[
            "Customer is on sick leave since 12 February with a slipped disc "
            "confirmed by MRI, on the 2025 income protection. When do payments start?"
        ],
    )


class ClauseOut(BaseModel):
    citation: str
    contract_id: str
    article: str
    heading: str
    product: str
    version: str
    status: str
    text: str
    score: float
    matched_queries: list[str]


class CaseResponse(BaseModel):
    case: str
    verdict: str
    headline: str
    explanation: str
    governing_clauses: list[str]
    conditions: list[str]
    what_would_change_the_answer: list[str]
    advisor_note: str
    search_queries: list[str]
    facts: dict[str, str]
    missing_facts: list[str]
    clauses: list[ClauseOut]
    model: str
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


@app.get("/")
async def health() -> dict:
    try:
        assistant = get_assistant()
        contracts = list(assistant.retriever.contract_index.keys())
        index_ready = True
    except Exception:
        contracts, index_ready = [], False
    return {
        "status": "ok",
        "index_ready": index_ready,
        "contracts": contracts,
        "chat_model": settings.chat_model,
    }


@app.post("/ingest")
async def rebuild_index() -> dict:
    """Re-parse and re-index the contract library, after a wording is updated."""
    global _assistant
    try:
        stats = ingest(reset=True)
        _assistant = None
        return {"status": "ok", **stats}
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/coverage", response_model=CaseResponse)
async def coverage(request: CaseRequest) -> CaseResponse:
    start = time.time()
    logger.info("Case: %s", request.case[:160])
    try:
        result = get_assistant().answer(request.case)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Coverage lookup failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer: CoverageAnswer = result.answer
    return CaseResponse(
        case=result.case,
        verdict=answer.verdict.value,
        headline=answer.headline,
        explanation=answer.explanation,
        governing_clauses=answer.governing_clauses,
        conditions=answer.conditions,
        what_would_change_the_answer=answer.what_would_change_the_answer,
        advisor_note=answer.advisor_note,
        search_queries=result.structured_query.search_queries,
        facts=result.structured_query.facts,
        missing_facts=result.structured_query.missing_facts,
        clauses=[
            ClauseOut(
                citation=clause.citation,
                contract_id=clause.contract_id,
                article=clause.article,
                heading=clause.heading,
                product=clause.product,
                version=clause.version,
                status=clause.status,
                text=clause.text,
                score=round(clause.score, 5),
                matched_queries=clause.matched_queries,
            )
            for clause in result.clauses
        ],
        model=result.model,
        latency_ms=int((time.time() - start) * 1000),
    )
