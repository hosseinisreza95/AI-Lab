"""Build the clause index.

Run as a module:  python -m ingestion.ingest
"""
from __future__ import annotations

import json
import os
import shutil

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from ingestion.clause_parser import Clause, contract_index, parse_contract_library
from rag.config import CLAUSE_STORE, COLLECTION_NAME, CONTRACTS_DIR, INDEX_DIR, settings

load_dotenv()

EMBED_BATCH = 64


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]


def get_collection(reset: bool = False):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def save_clause_store(clauses: list[Clause]) -> None:
    CLAUSE_STORE.parent.mkdir(parents=True, exist_ok=True)
    CLAUSE_STORE.write_text(
        json.dumps(
            {
                "clauses": [clause.to_dict() for clause in clauses],
                "contract_index": contract_index(clauses),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_clause_store() -> dict:
    if not CLAUSE_STORE.exists():
        return {"clauses": [], "contract_index": {}}
    return json.loads(CLAUSE_STORE.read_text(encoding="utf-8"))


def ingest(reset: bool = True) -> dict:
    if reset and INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)

    clauses = parse_contract_library(CONTRACTS_DIR)
    if not clauses:
        raise RuntimeError(f"No contract wordings found under {CONTRACTS_DIR}")

    collection = get_collection(reset=reset)

    for start in range(0, len(clauses), EMBED_BATCH):
        batch = clauses[start:start + EMBED_BATCH]
        collection.add(
            ids=[clause.clause_id for clause in batch],
            embeddings=embed([clause.embedding_text for clause in batch]),
            documents=[clause.embedding_text for clause in batch],
            metadatas=[
                {
                    "article": clause.article,
                    "heading": clause.heading,
                    "contract_id": clause.contract_id,
                    "product": clause.product,
                    "product_line": clause.product_line,
                    "version": clause.version,
                    "status": clause.status,
                    "applies_to_all": clause.applies_to_all,
                }
                for clause in batch
            ],
        )
        print(f"  indexed {start + len(batch)}/{len(clauses)} clauses")

    save_clause_store(clauses)

    return {
        "contracts": len({clause.contract_id for clause in clauses}),
        "clauses": len(clauses),
        "by_contract": {
            contract_id: entry["clauses"]
            for contract_id, entry in contract_index(clauses).items()
        },
    }


if __name__ == "__main__":
    print(f"Indexing contract library from {CONTRACTS_DIR} ...")
    print(json.dumps(ingest(reset=True), indent=2))
