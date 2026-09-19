"""Text normalisation shared by indexing and retrieval.

Kept in its own module with no heavy imports so the tokenizer can be tested
without pulling in the vector store or an API client.
"""
import re

# Article and contract references have to survive as single tokens: ART-3.1,
# IP-2025, HP-2024. A default word-boundary tokenizer splits ART-3.1 into
# "art", "3" and "1", and an advisor who types a clause number then matches
# every clause in the library.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())
