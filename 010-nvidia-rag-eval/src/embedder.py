from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "BAAI/bge-m3"


def load_chunks(chunks_path: Path) -> list[dict]:
    return json.loads(chunks_path.read_text(encoding="utf-8"))


def build_vector_store(chunks: list[dict], persist_dir: Path, collection_name: str = "nvda_10k"):
    # use_safetensors=True avoids the torch.load vulnerability restriction (CVE-2025-32434)
    model = SentenceTransformer(EMBED_MODEL_NAME, model_kwargs={"use_safetensors": True})
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(collection_name)

    texts = [c["text"] for c in chunks]
    ids = [str(i) for i in range(len(chunks))]
    metadatas = [{"heading": c["heading"], "type": c["type"]} for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    return collection


def query_chunks(query: str, persist_dir: Path, collection_name: str = "nvda_10k", top_k: int = 5):
    model = SentenceTransformer(EMBED_MODEL_NAME, model_kwargs={"use_safetensors": True})
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(collection_name)

    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )
    return results


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent

    # Uncomment this block only the first time, to build the index.
    # chunks = load_chunks(project_root / "chunks" / "structure_aware.json")
    # collection = build_vector_store(chunks, persist_dir=project_root / "vector_store")
    # print(f"Indexed {collection.count()} chunks into Chroma.")

    results = query_chunks(
    "Who is NVIDIA's Chief Financial Officer?",
    persist_dir=project_root / "vector_store",
    top_k=10,
    )
    
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"[{dist:.3f}] ({meta['heading']}, {meta['type']})")
        print(doc[:150])
        print("---")