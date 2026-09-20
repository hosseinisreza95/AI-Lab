import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
from pathlib import Path
from functools import partial

import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain_openai.chat_models import ChatOpenAI

from pipeline_naive import naive_rag_answer
from pipeline_hybrid import hybrid_rag_answer
from graph_builder import load_graph, graph_rag_answer
from agentic_rag import agentic_rag_answer
from embedder import load_chunks
from bm25_retriever import build_bm25_index
from local_client import local_client, LOCAL_MODEL_NAME


EVALUATOR_LLM = LangchainLLMWrapper(ChatOpenAI(model="gpt-5.4-mini", temperature=0))
METRICS = [LLMContextRecall(), Faithfulness(), FactualCorrectness()]


def setup_pipelines(project_root: Path):
    chunks = load_chunks(project_root / "chunks" / "structure_aware.json")
    bm25 = build_bm25_index(chunks)
    graph = load_graph(project_root / "graph_store" / "knowledge_graph.pkl")
    persist_dir = project_root / "vector_store"

    return {
        "naive_gpt": partial(naive_rag_answer, persist_dir=persist_dir),
        "naive_local": partial(naive_rag_answer, persist_dir=persist_dir, model=LOCAL_MODEL_NAME, client=local_client),
        "hybrid_gpt": partial(hybrid_rag_answer, persist_dir=persist_dir, bm25=bm25, chunks=chunks),
        "hybrid_local": partial(hybrid_rag_answer, persist_dir=persist_dir, bm25=bm25, chunks=chunks, model=LOCAL_MODEL_NAME, client=local_client),
        "graph_gpt": partial(graph_rag_answer, graph=graph),
        "graph_local": partial(graph_rag_answer, graph=graph, model=LOCAL_MODEL_NAME, client=local_client),
        "agentic_gpt": partial(agentic_rag_answer, persist_dir=persist_dir, bm25=bm25, chunks=chunks),
        "agentic_local": partial(agentic_rag_answer, persist_dir=persist_dir, bm25=bm25, chunks=chunks, model=LOCAL_MODEL_NAME, client=local_client),
    }


def build_ragas_dataset(pipeline_fn, test_questions: list[dict]) -> list[dict]:
    ragas_data = []
    for item in test_questions:
        result = pipeline_fn(item["query"])
        ragas_data.append({
            "user_input": item["query"],
            "retrieved_contexts": [c["text"] for c in result["retrieved_chunks"]],
            "response": result["answer"],
            "reference": item["ground_truth"],
        })
    return ragas_data


def evaluate_pipeline(method_name: str, pipeline_fn, test_questions: list[dict], n_eval_runs: int = 3):
    ragas_rows = build_ragas_dataset(pipeline_fn, test_questions)
    dataset = EvaluationDataset.from_list(ragas_rows)

    score_runs = []
    base_df = None
    for _ in range(n_eval_runs):
        result = evaluate(dataset=dataset, metrics=METRICS, llm=EVALUATOR_LLM)
        df = result.to_pandas().rename(columns={"factual_correctness(mode=f1)": "factual_correctness"})
        if base_df is None:
            base_df = df
        score_runs.append(df[["context_recall", "faithfulness", "factual_correctness"]])

    avg_scores = sum(score_runs) / n_eval_runs

    final_df = base_df.drop(columns=["context_recall", "faithfulness", "factual_correctness"])
    final_df = pd.concat([final_df, avg_scores], axis=1)

    final_df["method"] = method_name
    final_df["category"] = [q["category"] for q in test_questions]

    return final_df


def run_full_evaluation(project_root: Path) -> pd.DataFrame:
    pipelines = setup_pipelines(project_root)
    test_questions = json.loads((project_root / "evals" / "testset.json").read_text(encoding="utf-8"))

    all_results = []
    for method_name, pipeline_fn in pipelines.items():
        print(f"\n=== Evaluating: {method_name} ===")
        df = evaluate_pipeline(method_name, pipeline_fn, test_questions)
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    return combined


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent

    combined_df = run_full_evaluation(project_root)

    output_path = project_root / "results" / "eval_results_full.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    print(f"\nSaved full results to {output_path}")

    summary = combined_df.groupby("method")[["context_recall", "faithfulness", "factual_correctness"]].mean()
    print("\n=== Summary by method ===")
    print(summary)

    summary_by_category = combined_df.groupby(["method", "category"])[["context_recall", "faithfulness", "factual_correctness"]].mean()
    print("\n=== Summary by method + category ===")
    print(summary_by_category)