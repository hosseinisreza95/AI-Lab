"""Stage 3 - the grounded coverage answer.

The output is structured rather than free prose, because the advisor is going to
repeat it to a customer in the next ten seconds. A verdict they can read at a
glance, the clauses it rests on, and an explicit list of what would change the
answer is more useful than three paragraphs that are technically complete.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from ingestion.clause_parser import Clause  # noqa: F401  (documents the shape)
from rag.config import settings
from rag.query_builder import StructuredQuery, build_query
from rag.retriever import ClauseRetriever, RetrievedClause

load_dotenv()


class Verdict(str, Enum):
    COVERED = "covered"
    NOT_COVERED = "not_covered"
    COVERED_WITH_CONDITIONS = "covered_with_conditions"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    NO_APPLICABLE_CLAUSE = "no_applicable_clause"


class CoverageAnswer(BaseModel):
    verdict: Verdict
    headline: str = Field(
        ..., description="One sentence the advisor can say to the customer verbatim."
    )
    explanation: str = Field(
        ..., description="Two to four sentences. Every factual statement cites a clause."
    )
    governing_clauses: list[str] = Field(
        ..., description="Clause citations relied on, e.g. 'IP-2025 ART-3.1'."
    )
    conditions: list[str] = Field(
        ..., description="Conditions the customer must meet. Empty list if none."
    )
    what_would_change_the_answer: list[str] = Field(
        ...,
        description="Facts not stated in the case that would change the verdict.",
    )
    advisor_note: str = Field(
        ...,
        description="Anything the advisor should be careful about: a superseded "
                    "contract generation, a cap the customer will hit, a common "
                    "misreading. Empty string if there is nothing.",
    )


SYSTEM_PROMPT = """You are the coverage assistant used by insurance advisors while \
they are on the phone with a customer. The advisor will repeat your answer to the \
customer within seconds, so it must be correct and it must be short.

Absolute rules:
1. Answer only from the CLAUSES provided. If they do not settle the question, the \
verdict is insufficient_information or no_applicable_clause. Never reason from \
what insurance contracts usually say.
2. Cite the clause behind every factual statement, in the form (IP-2025 ART-3.1). \
An uncited statement is a defect, not a style choice.
3. If the case does not identify which contract generation the customer is on, and \
the clauses shown come from more than one generation with different answers, the \
verdict is insufficient_information. Say which generations differ and how. Do not \
pick the more generous one.
4. Where a benefit is expressed as a percentage of the statutory tariff, apply the \
definition in the cross-product definitions if it was retrieved, and say what the \
percentage actually means.
5. Do not compute an amount in euros unless every figure needed appears in the \
clauses. An advisor quoting a wrong number to a customer is the worst outcome this \
system can produce.
6. If a retrieved clause comes from a contract marked closed_to_new_business or \
superseded, and the case does not establish that the customer is on it, flag that \
in advisor_note.
7. what_would_change_the_answer is not optional padding. Coverage questions hinge \
on dates, on whether the cause was an accident or an illness, on whether imaging \
exists, on age. List the ones that apply here."""


@dataclass
class AnswerResult:
    case: str
    structured_query: StructuredQuery
    answer: CoverageAnswer
    clauses: list[RetrievedClause] = field(default_factory=list)
    model: str = ""


class CoverageAssistant:
    def __init__(self) -> None:
        self.retriever = ClauseRetriever()
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @staticmethod
    def _format_clauses(clauses: list[RetrievedClause]) -> str:
        blocks = []
        for clause in clauses:
            status = "" if clause.status == "active" else f" [{clause.status.upper()}]"
            blocks.append(
                f"({clause.contract_id} {clause.article}){status} "
                f"{clause.product} v{clause.version} - {clause.heading}\n{clause.text}"
            )
        return "\n\n---\n\n".join(blocks)

    def answer(self, case_text: str) -> AnswerResult:
        query = build_query(case_text, self.retriever.contract_index, client=self.client)

        clauses = self.retriever.retrieve(
            queries=query.search_queries,
            product_line=query.product_line,
            contract_ids=query.contract_ids or None,
        )

        if not clauses:
            return AnswerResult(
                case=case_text,
                structured_query=query,
                answer=CoverageAnswer(
                    verdict=Verdict.NO_APPLICABLE_CLAUSE,
                    headline="No clause in the contract library addresses this situation.",
                    explanation="The search returned no applicable clause. Escalate to "
                                "the underwriting team rather than answering from "
                                "general knowledge.",
                    governing_clauses=[],
                    conditions=[],
                    what_would_change_the_answer=query.missing_facts,
                    advisor_note="",
                ),
                model=settings.chat_model,
            )

        schema = CoverageAnswer.model_json_schema()
        schema.pop("title", None)

        response = self.client.chat.completions.create(
            model=settings.chat_model,
            temperature=settings.temperature,
            max_tokens=settings.max_answer_tokens,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                    + "\n\nRespond with JSON matching this schema:\n"
                    + json.dumps(schema, indent=2),
                },
                {
                    "role": "user",
                    "content": (
                        f"CLAUSES:\n{self._format_clauses(clauses)}\n\n"
                        f"ADVISOR CASE:\n{case_text}\n\n"
                        f"FACTS EXTRACTED FROM THE CASE:\n"
                        f"{json.dumps(query.facts, indent=2)}\n\n"
                        f"FACTS THE CASE DOES NOT STATE:\n"
                        f"{json.dumps(query.missing_facts, indent=2)}"
                    ),
                },
            ],
        )

        try:
            parsed = CoverageAnswer.model_validate_json(
                response.choices[0].message.content or "{}"
            )
        except ValidationError:
            # Never fall back to an unstructured answer: an advisor reading a
            # verdict-shaped response that is not grounded is the failure this
            # system exists to prevent.
            parsed = CoverageAnswer(
                verdict=Verdict.INSUFFICIENT_INFORMATION,
                headline="The assistant could not produce a validated answer.",
                explanation="The model response did not match the required answer "
                            "schema. Read the retrieved clauses directly.",
                governing_clauses=[clause.citation for clause in clauses],
                conditions=[],
                what_would_change_the_answer=query.missing_facts,
                advisor_note="Automatic answer suppressed. Clauses are still shown.",
            )

        return AnswerResult(
            case=case_text,
            structured_query=query,
            answer=parsed,
            clauses=clauses,
            model=settings.chat_model,
        )


if __name__ == "__main__":
    import sys

    case = " ".join(sys.argv[1:]) or (
        "Customer has been off work since 12 February with a slipped disc, "
        "confirmed on an MRI. She wants to know when the payments start."
    )
    result = CoverageAssistant().answer(case)
    print(json.dumps(result.answer.model_dump(), indent=2, default=str))
    print("\n--- clauses retrieved ---")
    for clause in result.clauses:
        print(f"  {clause.citation}  (score {clause.score:.5f})")
