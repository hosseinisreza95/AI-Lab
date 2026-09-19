"""Tests for clause parsing and the contract index.

These cover the part of the pipeline that decides what a "unit of retrieval" is.
If clause splitting is wrong, every downstream stage is answering from fragments.
No API key needed.

Run with:  pytest -q
"""
from __future__ import annotations

from pathlib import Path

from ingestion.clause_parser import (
    contract_index,
    parse_contract,
    parse_contract_library,
    parse_frontmatter,
)
from rag.config import CONTRACTS_DIR
from rag.text import tokenize

SAMPLE = """---
contract_id: TEST-2025
product: Test Product
product_line: income_protection
version: "2025.1"
effective_from: 2025-01-01
status: active
---

# Test Product 2025 - Policy Wording

### ART-1.1 - Purpose of cover

This contract pays a daily indemnity.

### ART-3.1 — Waiting period

Indemnity is payable from the 91st day of continuous incapacity for illness,
and from the 31st day for incapacity resulting from an accident.

### ART-6.2 - Psychological conditions

Covered subject to a specific waiting period of 180 days.
"""


def test_frontmatter_is_parsed():
    meta, body = parse_frontmatter(SAMPLE)
    assert meta["contract_id"] == "TEST-2025"
    assert meta["product_line"] == "income_protection"
    assert body.lstrip().startswith("# Test Product 2025")


def test_clauses_split_on_article_boundaries(tmp_path: Path):
    path = tmp_path / "test.md"
    path.write_text(SAMPLE, encoding="utf-8")

    clauses = parse_contract(path)
    assert [clause.article for clause in clauses] == ["ART-1.1", "ART-3.1", "ART-6.2"]

    # A clause must arrive whole. Half a waiting-period clause is quotable-looking
    # and wrong, which is the failure mode clause-level chunking exists to avoid.
    waiting = next(c for c in clauses if c.article == "ART-3.1")
    assert "91st day" in waiting.text
    assert "31st day" in waiting.text
    assert "daily indemnity" not in waiting.text


def test_clause_headings_accept_both_dash_characters(tmp_path: Path):
    path = tmp_path / "test.md"
    path.write_text(SAMPLE, encoding="utf-8")

    clauses = parse_contract(path)
    headings = {clause.article: clause.heading for clause in clauses}

    # ART-3.1 uses an em dash in the source, the others use a hyphen. Both are
    # what a wording exported from a document editor actually contains.
    assert headings["ART-3.1"] == "Waiting period"
    assert headings["ART-1.1"] == "Purpose of cover"


def test_clause_ids_are_namespaced_by_contract(tmp_path: Path):
    path = tmp_path / "test.md"
    path.write_text(SAMPLE, encoding="utf-8")

    clauses = parse_contract(path)
    assert clauses[0].clause_id == "TEST-2025::ART-1.1"

    # Two generations both have an ART-3.1 that says different things. Namespacing
    # is what stops one overwriting the other in the vector store.
    assert all(clause.clause_id.startswith("TEST-2025::") for clause in clauses)


def test_embedding_text_carries_contract_identity(tmp_path: Path):
    path = tmp_path / "test.md"
    path.write_text(SAMPLE, encoding="utf-8")

    clause = parse_contract(path)[2]
    assert "Test Product" in clause.embedding_text
    assert "TEST-2025" in clause.embedding_text
    assert "ART-6.2" in clause.embedding_text
    assert "180 days" in clause.embedding_text


def test_real_library_parses_and_generations_differ():
    clauses = parse_contract_library(CONTRACTS_DIR)
    assert len(clauses) > 60

    index = contract_index(clauses)
    assert {"IP-2025", "IP-2022", "HP-2024", "HE-2021", "GEN-DEF"} <= set(index)

    # The definitions document must be flagged as cross-product, otherwise a
    # contract filter hides it and percentages lose their meaning.
    assert index["GEN-DEF"]["applies_to_all"] is True
    assert index["IP-2025"]["applies_to_all"] is False

    # Superseded generations stay in the index on purpose: members are still on
    # them, and a claim is judged under the wording in force at the event date.
    assert index["IP-2022"]["status"] == "closed_to_new_business"


def test_the_same_article_differs_between_generations():
    clauses = {c.clause_id: c for c in parse_contract_library(CONTRACTS_DIR)}

    new = clauses["IP-2025::ART-6.2"].text
    old = clauses["IP-2022::ART-6.2"].text

    # This pair is the reason the assistant must refuse to answer when the case
    # does not say which generation the member is on.
    assert "180 days" in new
    assert "excluded" in old.lower()
    assert new != old


def test_tokenizer_keeps_article_references_intact():
    tokens = tokenize("does ART-3.1 apply to the IP-2025 contract")
    assert "art-3.1" in tokens
    assert "ip-2025" in tokens
