from openai import OpenAI

from generator import build_prompt

OLLAMA_BASE_URL = "http://172.28.57.218:11434/v1"
LOCAL_MODEL_NAME = "qwen2.5:7b"

local_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def generate_answer_local(query: str, retrieved_chunks: list[dict], model: str = LOCAL_MODEL_NAME) -> dict:
    prompt = build_prompt(query, retrieved_chunks)

    response = local_client.chat.completions.create(
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


if __name__ == "__main__":
    fake_chunks = [
        {"heading": "Test", "text": "Colette M. Kress is NVIDIA's Chief Financial Officer."}
    ]
    result = generate_answer_local("Who is NVIDIA's CFO?", fake_chunks)
    print(result["answer"])
    print(f"Tokens: {result['total_tokens']}")