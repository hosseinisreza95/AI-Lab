"""Text normalisation shared by indexing and retrieval.

Kept in its own module with no heavy imports so the tokenizer can be tested
without pulling in the vector store or an API client.
"""
import re

# Hyphenated identifiers have to survive as single tokens: E-121, BFP-40, GC-200.
# A default word-boundary tokenizer splits those into "e" and "121", which is
# exactly the query the keyword half of retrieval exists to serve.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())
