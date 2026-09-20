import json
from pathlib import Path

from generator import client, generate_answer
from bm25_retriever import build_bm25_index
from hybrid_retriever import hybrid_search
from embedder import load_chunks


JUDGE_PROMPT = """You are evaluating whether the context below is enough to fully and confidently answer the question.

Context:
{context}

Question: {query}

Respond with ONLY a JSON object:
{{
  "sufficient": true or false,
  "reason": "one short sentence explaining why",
  "reformulated_query": "a more specific search query to find the missing information, or empty string if sufficient"
}}
"""


def assess_sufficiency(query: str, retrieved_chunks: list[dict], model: str = "gpt-5.4-mini") -> dict:
    context = "\n\n".join(f"[{c['heading']}]\n{c['text']}" for c in retrieved_chunks)
    prompt = JUDGE_PROMPT.format(context=context, query=query)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def agentic_rag_answer(
    query: str,
    persist_dir: Path,
    bm25,
    chunks: list[dict],
    model: str = "gpt-5.4-mini",
    max_iterations: int = 3,
    client=client,
) -> dict:
    current_query = query
    all_chunks = []
    seen_texts = set()
    iteration_log = []

    for i in range(max_iterations):
        new_chunks = hybrid_search(current_query, persist_dir=persist_dir, bm25=bm25, chunks=chunks)

        for c in new_chunks:
            if c["text"] not in seen_texts:
                all_chunks.append(c)
                seen_texts.add(c["text"])

        judgement = assess_sufficiency(query, all_chunks, model="gpt-5.4-mini")  # judge always stays on mini
        iteration_log.append({
            "iteration": i + 1,
            "query_used": current_query,
            "sufficient": judgement["sufficient"],
            "reason": judgement["reason"],
        })

        if judgement["sufficient"]:
            break
        if not judgement.get("reformulated_query"):
            break
        current_query = judgement["reformulated_query"]

    generation = generate_answer(query, all_chunks, model=model, client=client)
    return {
        "query": query,
        "retrieved_chunks": all_chunks,
        "iterations": iteration_log,
        **generation,
    }


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    chunks = load_chunks(project_root / "chunks" / "structure_aware.json")
    bm25 = build_bm25_index(chunks)

    query = "According to Note 3, how much stock-based compensation expense was recorded under Research and Development for fiscal year 2026?"
    result = agentic_rag_answer(query, persist_dir=project_root / "vector_store", bm25=bm25, chunks=chunks)

    for log in result["iterations"]:
        print(f"Iteration {log['iteration']}: query='{log['query_used']}' -> sufficient={log['sufficient']} ({log['reason']})")
    print(f"\nFinal answer: {result['answer']}")