"""Retrieve, rerank, prompt, answer.

The orchestrator is deliberately the only place that builds a prompt, so the
grounding rules live in exactly one file and can be changed without touching
retrieval or the API layer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from rag.clients import get_client
from rag.config import settings
from rag.retriever import HybridRetriever, RetrievedChunk

SYSTEM_PROMPT = """You are the maintenance assistant inside the mobile app used by \
field technicians at an energy-supply company. The technician is standing in front \
of a machine, often under time pressure, sometimes with gloves on.

Rules you must follow:
1. Answer only from the CONTEXT passages. If the context does not contain the \
answer, say so plainly and name what documentation would be needed. Never invent a \
set point, a torque figure, an alarm code or a part number.
2. Every factual claim must carry a citation marker in the form [n] pointing at the \
numbered context passage it came from.
3. Lead with the single most likely cause and the first thing to check. The first \
action belongs in the first sentence, not after a summary of the manual.
4. Then give the ordered steps to follow. Keep them short enough to read on a phone \
screen.
5. When an expert note and the official manual disagree about how likely a cause is, \
say so explicitly and give both, marking which is which. Field notes reflect the \
experience of one site; the manual is the controlled document.
6. If any passage describes a safety requirement that applies (isolation, gas \
testing, permit to work, do-not-do warnings), surface it as a SAFETY line. Never \
suggest bypassing a trip or an interlock.

Respond in this structure:

**Most likely:** one sentence with the leading cause and the first check.

**Steps:**
1. ...
2. ...

**SAFETY:** only if the context contains a relevant requirement or warning.

**If that is not it:** the next one or two causes to work through."""

RERANK_PROMPT = """You score how useful a documentation passage is for answering a \
question from a field technician. Output only a single integer from 0 to 10.

10 = directly answers the question, or contains the exact alarm code, limit or \
procedure asked about.
5  = same equipment and same subject area, useful supporting context.
0  = unrelated, or about different equipment.

QUESTION: {question}

PASSAGE:
{passage}

Score:"""

SCORE_RE = re.compile(r"\d+")

SOURCE_LABELS = {
    "manual": "CONTROLLED MANUAL",
    "technical_doc": "TECHNICAL DOCUMENT",
    "expert_note": "EXPERT FIELD NOTE (experience from one site)",
}


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    used_chunk_ids: list[str] = field(default_factory=list)
    model: str = ""
    reranked: bool = False


class RagOrchestrator:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.client = get_client()

    # -- stage 2: rerank -------------------------------------------------------

    def _rerank(self, question: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Score each candidate with the chat model and keep the best ones.

        RRF gets the right passages into the candidate pool, but its ordering is
        driven by rank position rather than by meaning. A cheap pointwise LLM score
        fixes the ordering, which matters because only the top few reach the prompt.
        """
        for candidate in candidates:
            try:
                response = self.client.chat.completions.create(
                    model=settings.chat_model,
                    temperature=0,
                    max_tokens=4,
                    messages=[
                        {
                            "role": "user",
                            "content": RERANK_PROMPT.format(
                                question=question, passage=candidate.text[:1500]
                            ),
                        }
                    ],
                )
                match = SCORE_RE.search(response.choices[0].message.content or "")
                candidate.rerank_score = float(match.group()) if match else 0.0
            except Exception:
                # A failed rerank must not drop the passage: leaving rerank_score
                # unset falls the candidate back to its fusion ordering.
                candidate.rerank_score = None

        scored = [c for c in candidates if c.rerank_score is not None]
        if not scored:
            return candidates[: settings.final_top_k]

        kept = [c for c in scored if c.rerank_score >= settings.min_rerank_score]
        if not kept:
            # Everything scored low. Keep the best one anyway, so the model can say
            # the documentation does not cover the question.
            kept = sorted(scored, key=lambda c: c.rerank_score, reverse=True)[:1]

        kept.sort(key=lambda c: (c.rerank_score, c.score), reverse=True)
        return kept[: settings.final_top_k]

    # -- stage 3: prompt and answer -------------------------------------------

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for index, chunk in enumerate(chunks, start=1):
            label = SOURCE_LABELS.get(chunk.source_type, chunk.source_type.upper())
            blocks.append(
                f"[{index}] {label} - {chunk.citation} (rev {chunk.revision})\n{chunk.text}"
            )
        return "\n\n---\n\n".join(blocks)

    def ask(self, question: str, equipment: str | None = None) -> Answer:
        candidates = self.retriever.retrieve(question, equipment=equipment)

        if settings.rerank_enabled:
            chunks = self._rerank(question, candidates)
        else:
            chunks = candidates[: settings.final_top_k]

        if not chunks:
            return Answer(
                question=question,
                answer=(
                    "I could not find anything in the knowledge base for that "
                    "question. Raise a work order and escalate to the equipment "
                    "engineer."
                ),
                model=settings.chat_model,
                reranked=settings.rerank_enabled,
            )

        context = self._build_context(chunks)
        equipment_line = f"EQUIPMENT THE TECHNICIAN IS AT: {equipment}\n\n" if equipment else ""

        response = self.client.chat.completions.create(
            model=settings.chat_model,
            temperature=settings.temperature,
            max_tokens=settings.max_answer_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"CONTEXT:\n{context}\n\n"
                        f"{equipment_line}"
                        f"TECHNICIAN QUESTION: {question}"
                    ),
                },
            ],
        )

        return Answer(
            question=question,
            answer=response.choices[0].message.content or "",
            citations=[
                {
                    "marker": f"[{index}]",
                    "title": chunk.title,
                    "section": chunk.section,
                    "source_type": chunk.source_type,
                    "equipment": chunk.equipment,
                    "revision": chunk.revision,
                    "fusion_score": round(chunk.score, 5),
                    "rerank_score": chunk.rerank_score,
                    "matched_by": _matched_by(chunk),
                }
                for index, chunk in enumerate(chunks, start=1)
            ],
            used_chunk_ids=[chunk.chunk_id for chunk in chunks],
            model=settings.chat_model,
            reranked=settings.rerank_enabled,
        )


def _matched_by(chunk: RetrievedChunk) -> str:
    if chunk.vector_rank and chunk.keyword_rank:
        return "vector+keyword"
    return "vector" if chunk.vector_rank else "keyword"


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "The compressor is surging and I have E-121 and E-134"
    orchestrator = RagOrchestrator()
    result = orchestrator.ask(query)
    print(result.answer)
    print("\n--- sources ---")
    print(json.dumps(result.citations, indent=2))
