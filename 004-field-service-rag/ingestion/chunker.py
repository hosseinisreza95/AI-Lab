"""Markdown-aware chunking.

Splitting technician documentation on a fixed character count destroys the thing
that makes it useful: a step list, an alarm-code block or a limits table has to
stay together. So we split on markdown headings first and only fall back to
character splitting inside a section that is too long on its own.

Every chunk carries the heading trail it came from, which is what lets the answer
cite "GC-200 manual, section 4. Surge".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    document: str
    title: str
    source_type: str
    equipment: str
    equipment_family: str
    section: str
    revision: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def citation(self) -> str:
        return f"{self.title} > {self.section}" if self.section else self.title


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta = yaml.safe_load(match.group(1)) or {}
    return meta, raw[match.end():]


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Return (heading_trail, text) pairs, one per markdown section."""
    sections: list[tuple[str, str]] = []
    trail: dict[int, str] = {}
    current_heading = ""
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append((current_heading, text))

    for line in body.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            flush()
            buffer = []
            level = len(heading.group(1))
            title = heading.group(2).strip()
            trail[level] = title
            # Drop any deeper headings we have moved out of.
            for deeper in [lvl for lvl in trail if lvl > level]:
                del trail[deeper]
            current_heading = " > ".join(trail[lvl] for lvl in sorted(trail))
        else:
            buffer.append(line)

    flush()
    return sections


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Character split with overlap, preferring paragraph boundaries."""
    if len(text) <= chunk_size:
        return [text]

    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Prefer to break on a blank line, then a newline, then a space.
            for separator in ("\n\n", "\n", " "):
                pivot = text.rfind(separator, start + chunk_size // 2, end)
                if pivot != -1:
                    end = pivot
                    break
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [part for part in parts if part]


def chunk_document(path: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw)

    title = meta.get("title", path.stem)
    source_type = meta.get("source_type", path.parent.name)
    equipment = str(meta.get("equipment", "ALL"))
    equipment_family = str(meta.get("equipment_family", "unknown"))
    revision = str(meta.get("revision", "n/a"))
    document = path.name

    chunks: list[Chunk] = []
    for section, text in _split_sections(body):
        for piece in _split_long_text(text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{path.stem}::{len(chunks):03d}",
                    # Prefixing the heading trail gives the embedding the context
                    # a bare paragraph would be missing.
                    text=f"{title}\n{section}\n\n{piece}" if section else f"{title}\n\n{piece}",
                    document=document,
                    title=title,
                    source_type=source_type,
                    equipment=equipment,
                    equipment_family=equipment_family,
                    section=section,
                    revision=revision,
                )
            )
    return chunks


def chunk_knowledge_base(root: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*.md")):
        chunks.extend(chunk_document(path, chunk_size, overlap))
    return chunks
