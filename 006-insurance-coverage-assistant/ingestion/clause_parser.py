"""Clause-level parsing of contract wordings.

A contract is not prose, it is a numbered list of atomic obligations. The unit of
retrieval here is therefore the clause, never a fixed-size window: an advisor
answering "is this covered" has to be able to point at ART-4.2 and read it in
full. A chunk containing the second half of ART-4.1 and the first half of ART-4.2
is worse than useless, because it looks quotable and is not.

The trade-off is that some clauses are two lines long and embed poorly. That is
handled by prefixing every clause with its contract identity and heading, not by
merging clauses together.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# Matches "### ART-4.2 — Optical - lenses" with either dash character.
CLAUSE_RE = re.compile(r"^###\s+(ART-[A-Z0-9.]+)\s*[-—]\s*(.+)$")


@dataclass
class Clause:
    clause_id: str          # e.g. "IP-2025::ART-4.2"
    article: str            # e.g. "ART-4.2"
    heading: str
    text: str
    contract_id: str
    product: str
    product_line: str
    version: str
    effective_from: str
    status: str
    applies_to_all: bool

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def citation(self) -> str:
        return f"{self.contract_id} {self.article} - {self.heading}"

    @property
    def embedding_text(self) -> str:
        """What actually gets embedded.

        A bare clause body like "Not covered under this contract." carries no
        signal at all. Prefixing product, article and heading is what makes short
        clauses retrievable.
        """
        return (
            f"{self.product} ({self.contract_id}, version {self.version})\n"
            f"{self.article} - {self.heading}\n\n{self.text}"
        )


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    return yaml.safe_load(match.group(1)) or {}, raw[match.end():]


def parse_contract(path: Path) -> list[Clause]:
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))

    contract_id = str(meta.get("contract_id", path.stem))
    common = {
        "contract_id": contract_id,
        "product": str(meta.get("product", path.stem)),
        "product_line": str(meta.get("product_line", "unknown")),
        "version": str(meta.get("version", "n/a")),
        "effective_from": str(meta.get("effective_from", "")),
        "status": str(meta.get("status", "active")),
        "applies_to_all": str(meta.get("applies_to", "")).upper() == "ALL",
    }

    clauses: list[Clause] = []
    article = heading = None
    buffer: list[str] = []

    def flush() -> None:
        if article is None:
            return
        text = "\n".join(buffer).strip()
        if not text:
            return
        clauses.append(
            Clause(
                clause_id=f"{contract_id}::{article}",
                article=article,
                heading=heading,
                text=text,
                **common,
            )
        )

    for line in body.splitlines():
        match = CLAUSE_RE.match(line)
        if match:
            flush()
            article, heading = match.group(1), match.group(2).strip()
            buffer = []
        elif article is not None:
            buffer.append(line)

    flush()
    return clauses


def parse_contract_library(root: Path) -> list[Clause]:
    clauses: list[Clause] = []
    for path in sorted(root.rglob("*.md")):
        clauses.extend(parse_contract(path))
    return clauses


def contract_index(clauses: list[Clause]) -> dict[str, dict]:
    """Summary of the library, used to tell the query builder what exists.

    The query builder cannot filter by a contract it has never heard of, so this
    index is injected into its prompt rather than hard-coded.
    """
    index: dict[str, dict] = {}
    for clause in clauses:
        entry = index.setdefault(
            clause.contract_id,
            {
                "contract_id": clause.contract_id,
                "product": clause.product,
                "product_line": clause.product_line,
                "version": clause.version,
                "status": clause.status,
                "applies_to_all": clause.applies_to_all,
                "clauses": 0,
            },
        )
        entry["clauses"] += 1
    return index
