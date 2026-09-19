"""Stage 1 - turn an advisor's free-text case into a structured retrieval query.

An advisor types what the customer told them:

    "customer is on sick leave since February with a slipped disc, she has the
     2022 prevoyance, when does she start getting paid"

Embedding that sentence directly retrieves clauses about sick leave in general.
What is actually needed is three separate retrievals - waiting period, back and
spinal conditions, and the definition of incapacity - all filtered to IP-2022.
This stage does that decomposition, and it is the difference between an assistant
that finds the topic and one that finds the clause.

It also pulls out the facts that determine the answer (dates, contract, whether
the cause was an accident), which is what lets the answer stage say "this depends
on X and your case does not state X" instead of quietly assuming.
"""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL = os.getenv("QUERY_BUILDER_MODEL", os.getenv("CHAT_MODEL", "gpt-4o-mini"))


class StructuredQuery(BaseModel):
    coverage_question: str = Field(
        ..., description="The coverage question in one neutral sentence."
    )
    search_queries: list[str] = Field(
        ...,
        description="Two to four distinct retrieval queries in contract language, "
                    "each targeting a different clause the answer depends on.",
    )
    product_line: str = Field(
        ...,
        description="One of: health, income_protection, unknown.",
    )
    contract_ids: list[str] = Field(
        ...,
        description="Contract identifiers the case explicitly points to, from the "
                    "available library. Empty list if the case does not identify one.",
    )
    facts: dict[str, str] = Field(
        ...,
        description="Facts stated in the case that affect the answer: dates, cause "
                    "(illness or accident), age, dependant status, treatment type.",
    )
    missing_facts: list[str] = Field(
        ...,
        description="Facts the answer will depend on that the case does not state.",
    )


SYSTEM_PROMPT = """You prepare a contract search for an insurance advisor.

The advisor describes a customer situation in their own words. Your job is to turn \
it into retrieval queries phrased the way the contract wording is phrased, not the \
way the customer described it.

Rules:
- Produce two to four search queries, each aimed at a DIFFERENT clause. A coverage \
answer almost always depends on several clauses: what the benefit is, what the \
waiting period is, whether an exclusion or a limitation applies, and how the \
relevant term is defined. One query cannot find all of those.
- Write the queries in contract register. The customer says "off work with a bad \
back"; the wording says "incapacity resulting from a back or spinal condition" and \
"waiting period".
- Set contract_ids only when the case identifies a contract or generation. Guessing \
here is worse than leaving it empty, because it filters out the right clause.
- facts records what the case states and that matters to the answer.
- missing_facts records what the answer will hinge on but the case does not say. Be \
specific: "date the incapacity began", "whether imaging confirmed a lesion", "which \
contract generation the member is on".

Available contracts:
{contract_library}"""


def build_query(case_text: str, contract_library: dict, client: OpenAI | None = None) -> StructuredQuery:
    client = client or OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    schema = StructuredQuery.model_json_schema()
    schema["additionalProperties"] = False
    schema["required"] = list(schema["properties"].keys())
    # `facts` is an open map, which strict mode cannot express, so this call uses
    # plain json_object mode and relies on Pydantic to validate the result.
    schema.pop("title", None)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    contract_library=json.dumps(list(contract_library.values()), indent=2)
                )
                + "\n\nRespond with JSON matching this schema:\n"
                + json.dumps(schema, indent=2),
            },
            {"role": "user", "content": f"ADVISOR CASE:\n{case_text}"},
        ],
    )

    payload = json.loads(response.choices[0].message.content or "{}")
    # Defend against a model that returns a list where a dict was asked for; the
    # rest of the pipeline treats facts as a mapping.
    if not isinstance(payload.get("facts"), dict):
        payload["facts"] = {}
    return StructuredQuery.model_validate(payload)
