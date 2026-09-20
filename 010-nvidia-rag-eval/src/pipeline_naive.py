from pathlib import Path

from embedder import query_chunks
from generator import generate_answer, client as default_client
from local_client import local_client, LOCAL_MODEL_NAME


def naive_rag_answer(query: str, persist_dir: Path, model: str = "gpt-5.4-mini", top_k: int = 5, client=default_client) -> dict:
    results = query_chunks(query, persist_dir=persist_dir, top_k=top_k)
    retrieved_chunks = [
        {"heading": meta["heading"], "text": doc}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
    generation = generate_answer(query, retrieved_chunks, model=model, client=client)
    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        **generation,
    }


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    result = naive_rag_answer(
        "Who is NVIDIA's Chief Financial Officer?",
        persist_dir=project_root / "vector_store",
        model=LOCAL_MODEL_NAME,
        client=local_client,
    )
    print(result["answer"])