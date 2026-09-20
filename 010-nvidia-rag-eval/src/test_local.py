from pathlib import Path
from generator import client, generate_answer

from local_client import local_client, LOCAL_MODEL_NAME
from embedder import load_chunks
from bm25_retriever import build_bm25_index
from graph_builder import load_graph, graph_rag_answer
from pipeline_hybrid import hybrid_rag_answer
from agentic_rag import agentic_rag_answer


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    chunks = load_chunks(project_root / "chunks" / "structure_aware.json")
    bm25 = build_bm25_index(chunks)
    graph = load_graph(project_root / "graph_store" / "knowledge_graph.pkl")

    query = "Who is NVIDIA's Chief Financial Officer?"

    hybrid_result = hybrid_rag_answer(query, persist_dir=project_root / "vector_store", bm25=bm25, chunks=chunks, model=LOCAL_MODEL_NAME, client=local_client)
    print("Hybrid + local:", hybrid_result["answer"])

    graph_result = graph_rag_answer(query, graph=graph, model=LOCAL_MODEL_NAME, client=local_client)
    print("Graph + local:", graph_result["answer"])

    agentic_result = agentic_rag_answer(query, persist_dir=project_root / "vector_store", bm25=bm25, chunks=chunks, model=LOCAL_MODEL_NAME, client=local_client)
    print("Agentic + local:", agentic_result["answer"])