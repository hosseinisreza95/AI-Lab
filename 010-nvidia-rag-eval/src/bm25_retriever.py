import re
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def bm25_search(query: str, bm25: BM25Okapi, chunks: list[dict], top_k: int = 5) -> list[dict]:
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {"chunk_id": str(i), "heading": chunks[i]["heading"], "text": chunks[i]["text"], "score": scores[i]}
        for i in ranked_indices
    ]


if __name__ == "__main__":
    from embedder import load_chunks

    project_root = Path(__file__).parent.parent
    chunks = load_chunks(project_root / "chunks" / "structure_aware.json")

    bm25 = build_bm25_index(chunks)

    query = "Who is NVIDIA's Chief Financial Officer?"
    results = bm25_search(query, bm25, chunks)

    for r in results:
        print(f"[{r['score']:.2f}] {r['heading']}")
        print(r["text"][:150])
        print("---")