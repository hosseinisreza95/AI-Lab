from pathlib import Path

from embedder import load_chunks
from bm25_retriever import build_bm25_index
from hybrid_retriever import hybrid_search
from generator import generate_answer, client as default_client


def hybrid_rag_answer(query: str, persist_dir: Path, bm25, chunks: list[dict], model: str = "gpt-5.4-mini", top_k: int = 7, client=default_client) -> dict:
    retrieved_chunks = hybrid_search(query, persist_dir=persist_dir, bm25=bm25, chunks=chunks, top_k=top_k)
    generation = generate_answer(query, retrieved_chunks, model=model, client=client)
    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        **generation,
    }


test_questions = [
    {
        "query": "Who is NVIDIA's Chief Financial Officer?",
        "expected": "Colette M. Kress",
    },
    {
        "query": "According to Note 3, how much stock-based compensation expense was recorded under Research and Development for fiscal year 2026?",
        "expected": "$4,676 million",
    },
    {
        "query": "How much stock-based compensation expense was recorded under Cost of Revenue for fiscal year 2026?",
        "expected": "$261 million",
    },
]


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    chunks = load_chunks(project_root / "chunks" / "structure_aware.json")
    bm25 = build_bm25_index(chunks)

    for item in test_questions:
        result = hybrid_rag_answer(
            item["query"],
            persist_dir=project_root / "vector_store",
            bm25=bm25,
            chunks=chunks,
        )
        print(f"Q: {item['query']}")
        print(f"Expected: {item['expected']}")
        print(f"Got: {result['answer']}")
        print(f"Retrieved headings: {[c['heading'] for c in result['retrieved_chunks']]}")
        print("---")