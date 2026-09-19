"""Stage 2 - multi-query clause retrieval with metadata filters.

Each search query from the query builder runs independently, and the ranked lists
are fused. This is the reason the assistant finds the waiting period clause and
the exclusion clause and the definition, rather than three near-duplicates of
whichever clause matched the whole sentence best.
"""
from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from ingestion.ingest import embed, get_collection, load_clause_store
from rag.config import settings
from rag.text import tokenize


@dataclass
class RetrievedClause:
    clause_id: str
    article: str
    heading: str
    text: str
    contract_id: str
    product: str
    version: str
    status: str
    score: float
    matched_queries: list[str]

    @property
    def citation(self) -> str:
        return f"{self.contract_id} {self.article} - {self.heading}"


class ClauseRetriever:
    def __init__(self) -> None:
        store = load_clause_store()
        self.clauses = store["clauses"]
        self.contract_index = store["contract_index"]
        if not self.clauses:
            raise RuntimeError(
                "Clause store is empty. Run `python -m ingestion.ingest` first."
            )
        self.collection = get_collection(reset=False)
        self.by_id = {clause["clause_id"]: clause for clause in self.clauses}
        self.ids = [clause["clause_id"] for clause in self.clauses]
        self.bm25 = BM25Okapi(
            [tokenize(clause["text"] + " " + clause["heading"]) for clause in self.clauses]
        )

    def _where(self, product_line: str | None, contract_ids: list[str] | None) -> dict | None:
        """Build the Chroma metadata filter.

        Cross-product definitions (applies_to_all) must survive every filter. A
        benefit expressed as "200% of the statutory tariff" is meaningless without
        ART-D.1, and ART-D.1 does not live in the contract being filtered to.
        """
        clauses: list[dict] = []
        if contract_ids:
            clauses.append({"contract_id": {"$in": list(contract_ids)}})
        elif product_line and product_line != "unknown":
            clauses.append({"product_line": product_line})

        if not clauses:
            return None
        return {"$or": [*clauses, {"applies_to_all": True}]}

    def _vector_search(self, query: str, where: dict | None) -> list[str]:
        result = self.collection.query(
            query_embeddings=embed([query]),
            n_results=min(settings.vector_top_k, len(self.clauses)),
            where=where,
        )
        return result["ids"][0] if result["ids"] else []

    def _keyword_search(
        self, query: str, product_line: str | None, contract_ids: list[str] | None
    ) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.ids, scores), key=lambda pair: pair[1], reverse=True)

        out: list[str] = []
        for clause_id, score in ranked:
            if score <= 0:
                break
            clause = self.by_id[clause_id]
            if clause["applies_to_all"]:
                out.append(clause_id)
            elif contract_ids and clause["contract_id"] not in contract_ids:
                continue
            elif (
                not contract_ids
                and product_line
                and product_line != "unknown"
                and clause["product_line"] != product_line
            ):
                continue
            else:
                out.append(clause_id)
            if len(out) >= settings.keyword_top_k:
                break
        return out

    def retrieve(
        self,
        queries: list[str],
        product_line: str | None = None,
        contract_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[RetrievedClause]:
        where = self._where(product_line, contract_ids)

        fused: dict[str, float] = {}
        matched: dict[str, set[str]] = {}

        for query in queries:
            for ranked_ids in (
                self._vector_search(query, where),
                self._keyword_search(query, product_line, contract_ids),
            ):
                for rank, clause_id in enumerate(ranked_ids, start=1):
                    fused[clause_id] = fused.get(clause_id, 0.0) + 1.0 / (settings.rrf_k + rank)
                    matched.setdefault(clause_id, set()).add(query)

        ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
        cutoff = limit or settings.final_top_k

        results = []
        for clause_id, score in ordered[:cutoff]:
            clause = self.by_id[clause_id]
            results.append(
                RetrievedClause(
                    clause_id=clause_id,
                    article=clause["article"],
                    heading=clause["heading"],
                    text=clause["text"],
                    contract_id=clause["contract_id"],
                    product=clause["product"],
                    version=clause["version"],
                    status=clause["status"],
                    score=score,
                    matched_queries=sorted(matched.get(clause_id, set())),
                )
            )
        return results
