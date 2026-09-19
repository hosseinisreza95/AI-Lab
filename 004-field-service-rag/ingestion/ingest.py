"""Build the vector index from the knowledge base.

Run as a module:  python -m ingestion.ingest
"""
from __future__ import annotations

import json
import shutil

import chromadb

from ingestion.chunker import Chunk, chunk_knowledge_base
from rag.clients import embed
from rag.config import CHUNK_STORE, COLLECTION_NAME, INDEX_DIR, KNOWLEDGE_BASE_DIR, settings

EMBED_BATCH = 64


def get_collection(reset: bool = False):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            # Nothing to delete on a first run.
            pass
    # Cosine matches how the OpenAI embedding models are trained; Chroma defaults
    # to squared L2, which would quietly rank longer chunks differently.
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def save_chunk_store(chunks: list[Chunk]) -> None:
    """The keyword (BM25) half of retrieval needs the raw text, and Chroma is not
    a good store to stream a full corpus out of. Keep a JSON sidecar."""
    CHUNK_STORE.parent.mkdir(parents=True, exist_ok=True)
    CHUNK_STORE.write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], indent=2), encoding="utf-8"
    )


def load_chunk_store() -> list[dict]:
    if not CHUNK_STORE.exists():
        return []
    return json.loads(CHUNK_STORE.read_text(encoding="utf-8"))


def ingest(reset: bool = True) -> dict:
    if reset and INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)

    chunks = chunk_knowledge_base(
        KNOWLEDGE_BASE_DIR, settings.chunk_size, settings.chunk_overlap
    )
    if not chunks:
        raise RuntimeError(f"No markdown documents found under {KNOWLEDGE_BASE_DIR}")

    collection = get_collection(reset=reset)

    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start:start + EMBED_BATCH]
        vectors = embed([chunk.text for chunk in batch])
        collection.add(
            ids=[chunk.chunk_id for chunk in batch],
            embeddings=vectors,
            documents=[chunk.text for chunk in batch],
            metadatas=[
                {
                    "document": chunk.document,
                    "title": chunk.title,
                    "source_type": chunk.source_type,
                    "equipment": chunk.equipment,
                    "equipment_family": chunk.equipment_family,
                    "section": chunk.section,
                    "revision": chunk.revision,
                }
                for chunk in batch
            ],
        )
        print(f"  indexed {start + len(batch)}/{len(chunks)} chunks")

    save_chunk_store(chunks)

    by_type: dict[str, int] = {}
    for chunk in chunks:
        by_type[chunk.source_type] = by_type.get(chunk.source_type, 0) + 1

    return {
        "documents": len({chunk.document for chunk in chunks}),
        "chunks": len(chunks),
        "chunks_by_source_type": by_type,
    }


if __name__ == "__main__":
    print(f"Ingesting knowledge base from {KNOWLEDGE_BASE_DIR} ...")
    stats = ingest(reset=True)
    print(json.dumps(stats, indent=2))
