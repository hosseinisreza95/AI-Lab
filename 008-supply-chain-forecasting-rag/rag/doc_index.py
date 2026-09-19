"""Document retrieval over the logistics procedures.

Deliberately lighter than EXP-004 and EXP-006. The corpus is three controlled
documents of a few dozen sections, and it changes about twice a year. A BM25
index built in memory at import answers in under a millisecond, needs no vector
store, no embedding spend and no ingestion step in the Airflow DAG.

Reaching for a vector database on a corpus this size is a habit, not a decision.
The section-level splitting and the heading trail - the parts that actually
determine retrieval quality - are the same either way.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from rank_bm25 import BM25Okapi

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Section:
    document: str
    title: str
    doc_type: str
    revision: str
    heading: str
    text: str

    def to_dict(self) -> dict:
        return {
            "document": self.document,
            "title": self.title,
            "doc_type": self.doc_type,
            "revision": self.revision,
            "section": self.heading,
            "text": self.text,
        }


def _parse(path: Path) -> list[Section]:
    raw = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(raw)
    meta = yaml.safe_load(match.group(1)) if match else {}
    body = raw[match.end():] if match else raw

    sections: list[Section] = []
    trail: dict[int, str] = {}
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(
                Section(
                    document=path.name,
                    title=str(meta.get("title", path.stem)),
                    doc_type=str(meta.get("doc_type", "document")),
                    revision=str(meta.get("revision", "n/a")),
                    heading=heading,
                    text=text,
                )
            )

    for line in body.splitlines():
        found = HEADING_RE.match(line)
        if found:
            flush()
            buffer = []
            level = len(found.group(1))
            trail[level] = found.group(2).strip()
            for deeper in [key for key in trail if key > level]:
                del trail[deeper]
            heading = " > ".join(trail[key] for key in sorted(trail))
        else:
            buffer.append(line)

    flush()
    return sections


@lru_cache(maxsize=1)
def _index() -> tuple[list[Section], BM25Okapi]:
    sections: list[Section] = []
    for path in sorted(KB_DIR.glob("*.md")):
        sections.extend(_parse(path))
    if not sections:
        raise RuntimeError(f"No procedure documents found in {KB_DIR}")

    corpus = [
        # Heading and title join the searchable text: a section titled "Carrier
        # Claims" whose body never repeats the word "claim" is otherwise
        # unreachable by the obvious query.
        tokenize(f"{section.title} {section.heading} {section.text}")
        for section in sections
    ]
    return sections, BM25Okapi(corpus)


def search(query: str, top_k: int = 4) -> list[dict]:
    sections, bm25 = _index()
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(zip(sections, scores), key=lambda pair: pair[1], reverse=True)
    return [
        {**section.to_dict(), "score": round(float(score), 3)}
        for section, score in ranked[:top_k]
        if score > 0
    ]


if __name__ == "__main__":
    import json
    import sys

    question = " ".join(sys.argv[1:]) or "a delivery is 30 hours late, what do I do"
    print(json.dumps(search(question), indent=2))
