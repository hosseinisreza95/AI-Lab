from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

client = OpenAI()


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(f"[Source {i} - {chunk['heading']}]\n{chunk['text']}")
    context = "\n\n".join(context_blocks)

    return f"""Answer the question using ONLY the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:"""


def generate_answer(query: str, retrieved_chunks: list[dict], model: str = "gpt-5.4-mini", client=client) -> dict:
    prompt = build_prompt(query, retrieved_chunks)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return {
        "answer": response.choices[0].message.content,
        "model": model,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }