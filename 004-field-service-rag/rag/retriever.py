"""Hybrid retrieval: dense vectors plus BM25, fused with Reciprocal Rank Fusion.

Why both halves are needed here: a technician types an alarm code ("E-121") or a
part number, and a dense embedding is poor at exact token matches like that. They
also type symptoms in their own words ("it is making a gravel noise"), where BM25
finds nothing and the embedding does all the work. Neither retriever alone covers
both kinds of question.
"""
from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from ingestion.ingest import get_collection, load_chunk_store
from rag.clients import embed
from rag.config import settings
from rag.text import tokenize


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    title: str
    section: str
    source_type: str
    equipment: str
    revision: str
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        return f"{self.title} > {self.section}" if self.section else self.title


class HybridRetriever:
    def __init__(self) -> None:
        self.collection = get_collection(reset=False)
        self.chunks = load_chunk_store()
        if not self.chunks:
            raise RuntimeError(
                "Chunk store is empty. Run `python -m ingestion.ingest` first."
            )
        self.by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self.bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in self.chunks])
        self.ids = [chunk["chunk_id"] for chunk in self.chunks]

    # -- individual retrievers -------------------------------------------------

    def _vector_search(self, question: str, equipment: str | None) -> list[str]:
        where = None
        if equipment:
            # Documentation tagged ALL (safety procedures, generic guides) has to
            # stay reachable even when the technician filtered to one machine.
            where = {"equipment": {"$in": [equipment, "ALL"]}}
        result = self.collection.query(
            query_embeddings=embed([question]),
            n_results=min(settings.vector_top_k, len(self.chunks)),
            where=where,
        )
        return result["ids"][0] if result["ids"] else []

    def _keyword_search(self, question: str, equipment: str | None) -> list[str]:
        scores = self.bm25.get_scores(tokenize(question))
        ranked = sorted(zip(self.ids, scores), key=lambda pair: pair[1], reverse=True)
        out: list[str] = []
        for chunk_id, score in ranked:
            if score <= 0:
                break
            if equipment and self.by_id[chunk_id]["equipment"] not in (equipment, "ALL"):
                continue
            out.append(chunk_id)
            if len(out) >= settings.keyword_top_k:
                break
        return out

    # -- fusion ----------------------------------------------------------------

    def retrieve(
        self, question: str, equipment: str | None = None, limit: int | None = None
    ) -> list[RetrievedChunk]:
        vector_ids = self._vector_search(question, equipment)
        keyword_ids = self._keyword_search(question, equipment)

        fused: dict[str, float] = {}
        vector_rank: dict[str, int] = {}
        keyword_rank: dict[str, int] = {}

        for rank, chunk_id in enumerate(vector_ids, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (settings.rrf_k + rank)
            vector_rank[chunk_id] = rank

        for rank, chunk_id in enumerate(keyword_ids, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (settings.rrf_k + rank)
            keyword_rank[chunk_id] = rank

        ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
        cutoff = limit or settings.final_top_k

        results: list[RetrievedChunk] = []
        for chunk_id, score in ordered[: max(cutoff * 2, cutoff)]:
            chunk = self.by_id[chunk_id]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=chunk["text"],
                    title=chunk["title"],
                    section=chunk["section"],
                    source_type=chunk["source_type"],
                    equipment=chunk["equipment"],
                    revision=chunk["revision"],
                    score=score,
                    vector_rank=vector_rank.get(chunk_id),
                    keyword_rank=keyword_rank.get(chunk_id),
                )
            )
        return results
