"""Tests for the parts of the pipeline that do not need an API key.

Chunking and BM25 tokenization are where retrieval quality is silently won or
lost, so they are worth testing without paying for embeddings.

Run with:  pytest -q
"""
from pathlib import Path

from ingestion.chunker import chunk_document, chunk_knowledge_base, parse_frontmatter
from rag.config import KNOWLEDGE_BASE_DIR
from rag.text import tokenize

SAMPLE = """---
title: Test Machine TM-1 - Manual
source_type: manual
equipment: TM-1
equipment_family: test
revision: "1.0"
---

# Test Machine TM-1 - Manual

## 1. Overview

A short overview paragraph.

## 2. Alarm Codes

### E-999 - Test alarm

Check the thing before replacing the other thing.
"""


def test_parse_frontmatter_extracts_metadata():
    meta, body = parse_frontmatter(SAMPLE)
    assert meta["equipment"] == "TM-1"
    assert meta["source_type"] == "manual"
    assert body.lstrip().startswith("# Test Machine TM-1")


def test_chunks_carry_heading_trail(tmp_path: Path):
    path = tmp_path / "tm-1.md"
    path.write_text(SAMPLE, encoding="utf-8")

    chunks = chunk_document(path, chunk_size=900, overlap=150)
    sections = [chunk.section for chunk in chunks]

    # The nested alarm-code heading keeps its parent in the trail, which is what
    # makes the citation readable.
    assert any("2. Alarm Codes > E-999 - Test alarm" in section for section in sections)
    assert all(chunk.equipment == "TM-1" for chunk in chunks)
    assert all(chunk.title.startswith("Test Machine TM-1") for chunk in chunks)


def test_chunk_text_is_prefixed_with_context(tmp_path: Path):
    path = tmp_path / "tm-1.md"
    path.write_text(SAMPLE, encoding="utf-8")

    chunks = chunk_document(path, chunk_size=900, overlap=150)
    alarm_chunk = next(c for c in chunks if "E-999" in c.section)

    # Without the prefix, this chunk embeds as a sentence about "the thing" with
    # no indication of which machine it belongs to.
    assert alarm_chunk.text.startswith("Test Machine TM-1 - Manual")
    assert "E-999" in alarm_chunk.text


def test_long_section_is_split_with_overlap(tmp_path: Path):
    body = "---\ntitle: Long\nsource_type: manual\n---\n\n# Long\n\n## S\n\n"
    body += "\n\n".join(f"Paragraph number {i} with some filler text." for i in range(120))
    path = tmp_path / "long.md"
    path.write_text(body, encoding="utf-8")

    chunks = chunk_document(path, chunk_size=400, overlap=80)
    assert len(chunks) > 1
    assert all(len(chunk.text) < 900 for chunk in chunks)


def test_tokenizer_keeps_alarm_codes_intact():
    # This is the whole reason BM25 is in the pipeline: a technician types the
    # alarm code, and it has to survive tokenization as one token.
    assert "e-121" in tokenize("I have alarm E-121 on the panel")
    assert "bfp-40" in tokenize("the BFP-40 is rattling")
    assert "gc-200" in tokenize("GC-200 surge")


def test_real_knowledge_base_chunks_cleanly():
    chunks = chunk_knowledge_base(KNOWLEDGE_BASE_DIR, chunk_size=900, overlap=150)
    assert len(chunks) > 20

    source_types = {chunk.source_type for chunk in chunks}
    assert {"manual", "technical_doc", "expert_note"} <= source_types

    assert all(chunk.text.strip() for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
