from pathlib import Path

from embedder import load_chunks, query_chunks
from bm25_retriever import build_bm25_index, tokenize


def find_chunk_index(chunks: list[dict], heading: str, must_contain: str) -> int:
    matches = [
        i for i, c in enumerate(chunks)
        if c["heading"] == heading and must_contain in c["text"]
    ]
    if len(matches) != 1:
        print(f"Warning: found {len(matches)} matches, expected exactly 1")
    return matches[0] if matches else -1


def check_rank_in_bm25(target_index: int, query: str, bm25, chunks: list[dict]) -> int:
    scores = bm25.get_scores(tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked_indices.index(target_index) + 1


def check_rank_in_dense(target_index: int, query: str, persist_dir: Path, total_chunks: int) -> int:
    results = query_chunks(query, persist_dir=persist_dir, top_k=total_chunks)
    all_ids = results["ids"][0]
    return all_ids.index(str(target_index)) + 1


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    chunks = load_chunks(project_root / "chunks" / "structure_aware.json")
    bm25 = build_bm25_index(chunks)

    target_index = find_chunk_index(
        chunks,
        heading="Note 3 - Stock-Based Compensation",
        must_contain="4,676",
    )
    print(f"Target chunk index: {target_index}")
    print(f"Heading: {chunks[target_index]['heading']}")
    print(f"Text preview: {chunks[target_index]['text'][:200]}")

    query = "According to Note 3, how much stock-based compensation expense was recorded under Research and Development for fiscal year 2026?"

    bm25_rank = check_rank_in_bm25(target_index, query, bm25, chunks)
    dense_rank = check_rank_in_dense(target_index, query, project_root / "vector_store", len(chunks))

    print(f"BM25 rank: {bm25_rank} out of {len(chunks)}")
    print(f"Dense rank: {dense_rank} out of {len(chunks)}")