from pathlib import Path

from embedder import load_chunks
from bm25_retriever import build_bm25_index
from hybrid_retriever import hybrid_search
from generator import generate_answer


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    chunks = load_chunks(project_root / "chunks" / "structure_aware.json")
    bm25 = build_bm25_index(chunks)

    query = "Which two NVIDIA executives both previously worked at Texas Instruments?"

    chunks_retrieved = hybrid_search(
        query,
        persist_dir=project_root / "vector_store",
        bm25=bm25,
        chunks=chunks,
    )
    gen = generate_answer(query, chunks_retrieved)

    print(f"Retrieved headings: {[c['heading'] for c in chunks_retrieved]}")
    print(f"Answer: {gen['answer']}")