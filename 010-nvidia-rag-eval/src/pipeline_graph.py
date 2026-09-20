import json
import pickle
from pathlib import Path

import networkx as nx
from tqdm import tqdm

from generator import client, generate_answer
from embedder import load_chunks


EXTRACTION_PROMPT = """Extract entities and relationships from the text below.
Return ONLY a JSON object with this exact structure, nothing else:
{{
  "entities": ["Entity1", "Entity2", ...],
  "relationships": [
    {{"source": "Entity1", "relation": "short verb phrase", "target": "Entity2"}}
  ]
}}

Only extract entities that are people, companies, or organizations. Ignore numbers and dates as entities.

Text:
{text}
"""


def extract_entities_relationships(text: str, model: str = "gpt-5.4-mini") -> dict:
    prompt = EXTRACTION_PROMPT.format(text=text)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def build_graph(chunks: list[dict], model: str = "gpt-5.4-mini") -> nx.DiGraph:
    graph = nx.DiGraph()
    text_chunks = [c for c in chunks if c["type"] == "text"]

    for chunk in tqdm(text_chunks, desc="Extracting entities/relationships"):
        try:
            result = extract_entities_relationships(chunk["text"], model=model)
        except Exception as e:
            print(f"Skipping a chunk due to error: {e}")
            continue

        for entity in result.get("entities", []):
            graph.add_node(entity)

        for rel in result.get("relationships", []):
            source, target = rel.get("source"), rel.get("target")
            if source and target:
                graph.add_edge(
                    source, target,
                    relation=rel.get("relation", ""),
                    heading=chunk["heading"],
                    text=chunk["text"],
                )

    return graph


def merge_duplicate_entities(graph: nx.DiGraph) -> nx.DiGraph:
    nodes = sorted(graph.nodes, key=len, reverse=True)
    merged_into = {}

    for short_name in sorted(graph.nodes, key=len):
        for long_name in nodes:
            if short_name != long_name and short_name in long_name and len(short_name) > 3:
                merged_into[short_name] = long_name
                break

    mapping = {k: v for k, v in merged_into.items() if k in graph.nodes}
    return nx.relabel_nodes(graph, mapping, copy=True)


COMMON_ENTITIES = {"nvidia"}


def find_matching_nodes(entity: str, graph: nx.DiGraph) -> list[str]:
    entity_lower = entity.lower()
    return [
        n for n in graph.nodes
        if entity_lower in n.lower() or n.lower() in entity_lower
    ]


def graph_search(query: str, graph: nx.DiGraph, model: str = "gpt-5.4-mini") -> list[dict]:
    query_entities_result = extract_entities_relationships(query, model=model)
    query_entities = [
        e for e in query_entities_result.get("entities", [])
        if e.lower() not in COMMON_ENTITIES
    ]

    matched_nodes = set()
    for entity in query_entities:
        matched_nodes.update(find_matching_nodes(entity, graph))

    relevant_edges = set()
    for node in matched_nodes:
        for _, target, data in graph.out_edges(node, data=True):
            relevant_edges.add((node, target, data["relation"], data["heading"], data["text"]))
        for source, _, data in graph.in_edges(node, data=True):
            relevant_edges.add((source, node, data["relation"], data["heading"], data["text"]))

    seen_texts = set()
    context_chunks = []
    for source, target, relation, heading, text in relevant_edges:
        if text not in seen_texts:
            context_chunks.append({"heading": heading, "text": text})
            seen_texts.add(text)

    return context_chunks


def graph_rag_answer(query, graph, model="gpt-5.4-mini", client=client):
    retrieved_chunks = graph_search(query, graph, model="gpt-5.4-mini")
    generation = generate_answer(query, retrieved_chunks, model=model, client=client)
    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        **generation,
    }


def save_graph(graph: nx.DiGraph, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(graph, f)


def load_graph(path: Path) -> nx.DiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    graph = load_graph(project_root / "graph_store" / "knowledge_graph.pkl")

    query = "Which two NVIDIA executives both previously worked at Texas Instruments?"
    result = graph_rag_answer(query, graph)

    print(f"Retrieved {len(result['retrieved_chunks'])} chunks")
    print(f"Answer: {result['answer']}")