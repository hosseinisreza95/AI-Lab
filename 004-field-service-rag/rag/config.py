"""Central configuration for the technician RAG assistant.

The production deployment runs on Azure OpenAI. The same code path works against
the public OpenAI API so the experiment can be reproduced without an Azure
subscription: set AZURE_OPENAI_ENDPOINT to switch, leave it unset to use OpenAI.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
INDEX_DIR = BASE_DIR / ".index"
CHUNK_STORE = INDEX_DIR / "chunks.json"

COLLECTION_NAME = "field_service_kb"


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


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- Models -----------------------------------------------------------
    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # --- Azure OpenAI (optional) ------------------------------------------
    azure_endpoint: str | None = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT") or None
    )
    azure_api_version: str = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    )

    # --- Chunking ---------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 150))

    # --- Retrieval --------------------------------------------------------
    vector_top_k: int = field(default_factory=lambda: _int("VECTOR_TOP_K", 12))
    keyword_top_k: int = field(default_factory=lambda: _int("KEYWORD_TOP_K", 12))
    final_top_k: int = field(default_factory=lambda: _int("FINAL_TOP_K", 5))
    rrf_k: int = field(default_factory=lambda: _int("RRF_K", 60))
    rerank_enabled: bool = field(default_factory=lambda: _bool("RERANK_ENABLED", True))
    min_rerank_score: float = field(default_factory=lambda: _float("MIN_RERANK_SCORE", 3.0))

    # --- Generation -------------------------------------------------------
    temperature: float = field(default_factory=lambda: _float("TEMPERATURE", 0.0))
    max_answer_tokens: int = field(default_factory=lambda: _int("MAX_ANSWER_TOKENS", 900))

    @property
    def use_azure(self) -> bool:
        return self.azure_endpoint is not None


settings = Settings()
