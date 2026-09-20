from pathlib import Path

from rank_bm25 import BM25Okapi

from embedder import query_chunks, load_chunks
from bm25_retriever import build_bm25_index, bm25_search


def reciprocal_rank_fusion(dense_ids: list[str], bm25_ids: list[str], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}

    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    return scores


def hybrid_search(query: str, persist_dir: Path, bm25: BM25Okapi, chunks: list[dict], top_k: int = 7, fetch_k: int = 20) -> list[dict]:
    dense_results = query_chunks(query, persist_dir=persist_dir, top_k=fetch_k)
    dense_ids = dense_results["ids"][0]

    bm25_results = bm25_search(query, bm25, chunks, top_k=fetch_k)
    bm25_ids = [r["chunk_id"] for r in bm25_results]

    fused_scores = reciprocal_rank_fusion(dense_ids, bm25_ids)
    top_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]

    chunks_by_id = {str(i): c for i, c in enumerate(chunks)}
    return [
        {"heading": chunks_by_id[cid]["heading"], "text": chunks_by_id[cid]["text"], "rrf_score": fused_scores[cid]}
        for cid in top_ids
    ]


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    chunks = load_chunks(project_root / "chunks" / "structure_aware.json")
    bm25 = build_bm25_index(chunks)

    query = "Who is NVIDIA's Chief Financial Officer?"
    results = hybrid_search(query, persist_dir=project_root / "vector_store", bm25=bm25, chunks=chunks, top_k=7)

    for r in results:
        print(f"[{r['rrf_score']:.4f}] {r['heading']}")
        print(r["text"][:150])
        print("---")