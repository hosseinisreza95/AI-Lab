import asyncio

from ragas.metrics import FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.dataset_schema import SingleTurnSample
from langchain_openai.chat_models import ChatOpenAI


async def main():
    llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-5.4-mini", temperature=0, seed=42)
    )
    metric = FactualCorrectness(llm=llm)

    sample = SingleTurnSample(
        response="Ajay K. Puri and Debora Shoquist.",
        reference="The two NVIDIA executives who both previously worked at Hewlett-Packard are Ajay K. Puri and Debora Shoquist.",
    )

    for i in range(3):
        score = await metric.single_turn_ascore(sample)
        print(f"Run {i+1}: {score}")


if __name__ == "__main__":
    asyncio.run(main())