"""Configuration for the coverage assistant."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = BASE_DIR / "contracts"
INDEX_DIR = BASE_DIR / ".index"
CLAUSE_STORE = INDEX_DIR / "clauses.json"

COLLECTION_NAME = "insurance_clauses"


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    vector_top_k: int = field(default_factory=lambda: _int("VECTOR_TOP_K", 15))
    keyword_top_k: int = field(default_factory=lambda: _int("KEYWORD_TOP_K", 15))
    final_top_k: int = field(default_factory=lambda: _int("FINAL_TOP_K", 8))
    rrf_k: int = field(default_factory=lambda: _int("RRF_K", 60))

    temperature: float = field(default_factory=lambda: _float("TEMPERATURE", 0.0))
    max_answer_tokens: int = field(default_factory=lambda: _int("MAX_ANSWER_TOKENS", 1200))

    api_url: str = field(
        default_factory=lambda: os.getenv("API_URL", "http://localhost:8000")
    )


settings = Settings()
