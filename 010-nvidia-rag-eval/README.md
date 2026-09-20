# RAG Methods Comparison on a Real 10-K Filing

A hands-on evaluation of four Retrieval-Augmented Generation (RAG) architectures — **Naive**, **Hybrid (dense + BM25)**, **GraphRAG**, and **Agentic RAG** — built from scratch and benchmarked on NVIDIA's FY2026 10-K filing, using [Ragas](https://github.com/explodinggradients/ragas) for evaluation and tested against both a cloud model (GPT-5.4-mini) and a local 7B model (Qwen2.5, via Ollama).

This is a learning/portfolio project. The goal was not just to build four RAG pipelines, but to **document exactly where and why each one fails**, using real numbers instead of anecdotes.

---

## TL;DR

- No single RAG method wins overall. **Hybrid and Agentic RAG performed roughly equally** on the combined metric, both clearly ahead of Naive RAG; **GraphRAG only outperforms when the question is genuinely entity-relationship-based** — on every other question type it scored close to zero.
- A large share of the failures traced back to **chunking decisions**, not fundamental limitations of the retrieval method (e.g., a table separated from the sentence that introduces it, a small table whose embedding blends five unrelated names together).
- Some failures are **not fixable by any of these four methods**: lexical ambiguity, where the same phrase ("cost of revenue") legitimately appears in several unrelated contexts, defeated Naive, Hybrid, and Agentic RAG identically.
- The evaluation framework itself introduced noise: the `FactualCorrectness` Ragas metric scored a bare name ("Ajay K. Puri") as `0.0` against an identical reference, purely because it could not extract an atomic "claim" from a name with no verb — this alone changed the outcome of several category-level comparisons before it was caught and fixed.
- Comparing a local 7B model to GPT-5.4-mini is easy to get wrong: their answers were near-identical in *content*, but Qwen2.5 tends to answer in fuller sentences, which coincidentally scores higher against sentence-form ground truth. Retrieval quality (`context_recall`), which is style-independent, was **identical** between the cloud and local variant of every pipeline — as it should be, since only the generator changed.

---

## 1. The Document and the Task

**Source document:** NVIDIA Corporation's Form 10-K for fiscal year 2026 (period ended January 25, 2026), pulled directly from [SEC EDGAR](https://www.sec.gov/), ~150–300 pages, heavy with financial tables, footnotes, executive biographies, and risk factors.

**Why this document:** it's genuinely hard. It mixes long narrative sections with dense financial tables, repeats similar phrases ("cost of revenue," "stock-based compensation") in unrelated contexts dozens of times, and contains information that appears redundantly in both structured tables and prose — all of which turned out to matter a great deal.

## 2. Pipeline

### 2.1 Parsing
- **[Docling](https://github.com/docling-project/docling)** — chosen because it's free, fully local, and its TableFormer model is specifically built for table structure recognition, which mattered enormously for a financial filing. It also runs OCR (RapidOCR) on embedded images.
- Output: a single Markdown file with headings preserved and tables reconstructed as proper Markdown tables.

### 2.2 Chunking
A custom **structure-aware chunker**:
- Splits on the document's own Markdown headings (`#`, `##`, `###`).
- Extracts every table as its **own, never-split chunk** — deliberately, even if that makes the chunk larger than the target size, because breaking a financial table mid-row is worse than a slightly oversized chunk.
- Falls back to fixed-size token splitting (`tiktoken`, `cl100k_base`) with overlap for any section that's still too long after splitting by heading.

This one design choice — never splitting a table — turned out to be a double-edged sword (see Finding #3 below).

### 2.3 Embedding & Vector Store
- **Embedding model:** [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — chosen specifically because it supports up to 8,192 tokens (needed for the large, never-split table chunks) and is free to run locally via `sentence-transformers`.
- **Vector store:** [Chroma](https://www.trychroma.com/), persistent, local, no server required.
- The same embedding model is used for *every* pipeline and for *both* the cloud and local generation tracks, so the only variable that changes across experiments is the generator model, not retrieval.

### 2.4 The Four RAG Methods

| Method | Retrieval mechanism |
|---|---|
| **Naive** | Dense (cosine similarity) top-5 |
| **Hybrid** | Dense (top-10) + BM25 (top-10, regex `\w+` tokenizer) fused with Reciprocal Rank Fusion (k=60), final top-7 |
| **GraphRAG** | LLM-extracted (entity, relation, entity) triples per chunk → NetworkX directed graph → query-time entity extraction + graph traversal (in/out edges) |
| **Agentic RAG** | Iterative Hybrid search + an LLM "judge" that decides if context is sufficient and, if not, reformulates the query (up to 3 iterations) |

All four share the same final generation step: a strict "answer only from context, say you don't know otherwise" prompt, `temperature=0`.

### 2.5 Generator Models
- **Cloud:** GPT-5.4 and GPT-5.4-mini via the OpenAI-compatible API. Given a limited token budget (200K for GPT-5.4, 2.5M for mini), the strategy was: **development, debugging, entity/relationship extraction, and the Ragas judge always use mini**; GPT-5.4 itself is reserved for a small number of "official" comparison runs.
- **Local:** Qwen2.5:7B (Q4_K_M quantization), served via Ollama inside WSL2, running on an RTX 4060 Laptop GPU (8GB VRAM). Reached from the Windows Python environment through the WSL instance's assigned IP address (WSL2 mirrored networking was not active on this machine).

## 3. Evaluation Methodology

### 3.1 Test Set
15 questions across 5 categories, each grounded in facts manually verified against the actual document (see [`evals/testset.json`](evals/testset.json)):

| Category | # | Designed to test |
|---|---|---|
| `control` | 5 | Simple, single-location facts — a sanity baseline every method should pass |
| `small_table_averaging` | 3 | Small tables mixing several named entities (e.g., a 5-row executive officer table) — does the embedding blend them into an unhelpfully generic vector? |
| `lexical_distractor` | 2 | Phrases (e.g., "cost of revenue") that legitimately recur in multiple unrelated contexts |
| `intro_table_split` | 3 | Tables preceded by an introductory sentence ("...as follows:") that ends up in a different chunk than the table itself |
| `pseudo_multihop_lexical` | 2 | Originally designed as multi-hop reasoning tests (two executives sharing a former employer); later found to be solvable by lexical match alone — kept and honestly re-labeled rather than discarded |

### 3.2 Metrics
Three Ragas metrics: `LLMContextRecall`, `Faithfulness`, `FactualCorrectness` — judged by GPT-5.4-mini at `temperature=0`. Each (question, method) pair is evaluated **three times and averaged**, because even at `temperature=0` (and with an explicit `seed`), the Ragas judge itself proved non-deterministic (see Finding #9).

### 3.3 Results — GPT-5.4-mini as generator

| Method | Context Recall | Faithfulness | Factual Correctness |
|---|---|---|---|
| Hybrid | 0.756 | 0.844 | **0.596** |
| Agentic | 0.778 | 0.889 | **0.625** |
| Naive | 0.733 | 0.933 | 0.514 |
| GraphRAG | 0.133 | 0.222 | 0.108 |

By category (GPT-5.4-mini):

| Method | control | intro_table_split | lexical_distractor | pseudo_multihop | small_table_avg |
|---|---|---|---|---|---|
| Naive | 0.568 | 0.333 | 0.500 | 0.523 | 0.610 |
| Hybrid | 0.768 | 0.333 | 0.500 | 0.427 | 0.750 |
| GraphRAG | 0.178 | 0.000 | 0.000 | 0.365 | 0.000 |
| Agentic | 0.768 | 0.333 | 0.500 | 0.433 | 0.890 |

### 3.4 Cloud vs. Local Generator

| Method | Context Recall (both identical) | Faithfulness (GPT / Local) | Factual Correctness (GPT / Local) |
|---|---|---|---|
| Naive | 0.733 | 0.933 / 0.868 | 0.514 / 0.613 |
| Hybrid | 0.756 | 0.844 / 0.903 | 0.596 / 0.692 |
| GraphRAG | 0.133 | 0.222 / 0.833 | 0.108 / 0.134 |
| Agentic | 0.778 | 0.889 / 0.913 | 0.625 / 0.704 |

**Important caveat before reading this table as "local wins":** see Finding #13. A real, direction-consistent local weakness only showed up on `pseudo_multihop_lexical` (Naive: 0.523 → 0.200), where synthesizing two separate names into one fluent sentence is genuinely harder for a 7B model.

## 4. Findings Worth Remembering

These are listed roughly in the order they were discovered, because the order matters — several later findings only make sense in light of earlier ones.

1. **Small-table entity averaging.** A 5-row executive-officer table (CFO, CEO, EVPs) embeds into one vector that represents "several unrelated names," not any one of them — the CFO's row never even entered the top-10 dense results for "Who is NVIDIA's CFO?" Hybrid retrieval fixed this specific case because BM25 matches the literal phrase "Chief Financial Officer" regardless of embedding quality.
2. **Naive whitespace tokenization breaks BM25.** `"officer?".split()` never matches `"officer"` in the corpus. Fixed with a `re.findall(r"\w+", text.lower())` tokenizer.
3. **Intro-sentence/table split.** The chunker's "never split a table" rule has a blind spot: the sentence introducing a table ("...as follows:") lands in a *different* chunk than the table it refers to. The model sees the promise of data, not the data — and, credibly, says "I don't know" rather than guessing.
4. **Lexical ambiguity is not fixable by better retrieval alone.** "Cost of revenue" appears in the correct answer ($261M, stock-based comp) and in two unrelated footnotes ($3.2B expense, $4.0B inventory provision). Naive, Hybrid, and Agentic RAG all scored identically here (0.500) and Agentic occasionally hallucinated one of the wrong figures — this needs a different technique entirely (see Future Work).
5. **The retrieval fusion window matters as much as the final top-k.** Widening Hybrid's pre-fusion candidate pool from `fetch_k=10` to `fetch_k=20` pulled in a target chunk ranked 11th in dense retrieval that was otherwise permanently excluded before Reciprocal Rank Fusion even ran.
6. **Entity resolution is a real cost of GraphRAG.** The same person appeared as two separate graph nodes ("Puri" and "Ajay K. Puri") purely because a biography got split across a chunk boundary. Fixed with a naive substring-merge heuristic — acceptable for this project, but a known source of false merges at scale.
7. **Fuzzy entity matching needs a stopword list.** Allowing "NVIDIA" to fuzzy-match any node containing that substring pulled in 19 unrelated product/litigation nodes and exploded retrieved context to 46 chunks for one query. Excluding the document's own subject entity from fuzzy matching brought this down to 2 relevant chunks.
8. **Agentic loops can get stuck without making progress.** The query-reformulation judge sometimes proposed the *exact same* rewritten query twice in a row, burning an iteration (and tokens) for zero new information.
9. **Neither generation nor judging is fully deterministic at `temperature=0`.** The same question, same context, same code produced both a correct answer and a hallucinated one across separate runs. An explicit `seed` did not resolve this for the Ragas judge either — the only practical mitigation was averaging multiple evaluation runs.
10. **The evaluation library itself is a moving target.** Within one evaluation session: a hard `ImportError` from an unrelated `langchain-community` version bump, several metric-import deprecation warnings, and a confirmed upstream bug where the "recommended" new `ragas.metrics.collections` API is incompatible with `evaluate()` ([ragas#2624](https://github.com/vibrantlabsai/ragas/issues/2624)). The project deliberately stayed on the "legacy" `ragas.metrics` + `evaluate()` + `LangchainLLMWrapper` path.
11. **`FactualCorrectness` penalizes short, correct answers.** A bare name response scored `0.0` against an identical bare-name reference, because the metric decomposes both into "claims" and a name alone yields none. Rewriting every `ground_truth` entry as a full sentence fixed this — but it means the category-level scores reported earlier in the project (before this fix) understated `small_table_averaging` performance across the board.
12. **Style, not just accuracy, drives metric scores across models.** Qwen2.5:7B tends to answer in complete sentences; GPT-5.4-mini is terser by default. Against sentence-form ground truth, this stylistic difference alone shifted `FactualCorrectness` scores by up to 0.5 points on questions where both models' actual content was equally correct.
13. **Not every apparent weakness of a method is fundamental.** A follow-up review distinguished implementation choices from genuine method limits — summarized below.

## 5. Implementation Limitations vs. Fundamental Method Limitations

| Failure | Fundamental to the method? | Implementation choice that could fix it |
|---|---|---|
| CFO table averaging | Partially — generic sentence embeddings are known to be weak on raw tabular text | Row-level chunking, or **Contextual Retrieval** (prepend an LLM-written summary to each chunk before embedding) |
| Cost-of-revenue ambiguity | Mostly yes, for both BM25 and dense embeddings | Contextual Retrieval again — a short caption disambiguating each footnote's actual subject |
| Intro/table split | **No** — this is a chunker bug, not a RAG limitation | Attach the last sentence before a table to the table's own chunk |
| GraphRAG's near-zero score outside relationship questions | Yes, by design — a knowledge graph has no answer for "what was total revenue" | N/A — use GraphRAG selectively, not as a general-purpose retriever |

## 6. Repository Structure

```
010-nvidia-rag-eval/
├── dataset/                # Source 10-K PDF
├── parsed/                 # Docling output (Markdown)
├── chunks/                 # structure_aware.json
├── vector_store/           # Chroma persistent store
├── graph_store/            # Pickled NetworkX graph
├── evals/testset.json      # 15-question eval set
├── results/                # Ragas output CSVs
└── src/
    ├── parser.py            # Docling parsing
    ├── chunkers.py           # Structure-aware chunking
    ├── embedder.py           # BGE-M3 + Chroma
    ├── bm25_retriever.py     # BM25 index + search
    ├── hybrid_retriever.py   # Reciprocal Rank Fusion
    ├── graph_builder.py      # Entity/relationship extraction + graph search
    ├── agentic_rag.py        # Iterative retrieval + judge
    ├── generator.py          # Prompting + OpenAI-compatible generation
    ├── local_client.py       # Ollama (Qwen2.5:7B) client
    ├── pipeline_naive.py / pipeline_hybrid.py / pipeline_graph.py / pipeline_agentic.py
    └── eval_runner.py        # Full Ragas evaluation across all 8 (method × generator) variants
```

## 7. Running It

```bash
pip install docling tiktoken chromadb sentence-transformers rank_bm25 \
            networkx tqdm openai python-dotenv "langchain-community<0.4.2" \
            ragas langchain-openai pandas

# 1. Parse and chunk
python src/parser.py
python src/chunkers.py

# 2. Build the vector store and knowledge graph (uncomment the build step inside each file)
python src/embedder.py
python src/graph_builder.py

# 3. Try any single pipeline
python src/pipeline_hybrid.py

# 4. Full evaluation (cloud + local, all 4 methods)
python src/eval_runner.py
```

For the local model track: install [Ollama](https://ollama.com), pull `qwen2.5:7b`, and run it with `OLLAMA_HOST=0.0.0.0 ollama serve` (adjust `local_client.py`'s `OLLAMA_BASE_URL` to match your machine's address if running inside WSL2).

## 8. Future Work

- **Contextual Retrieval** — prepend an LLM-generated summary of each chunk's context before embedding; the most promising untried fix for the lexical-ambiguity failure mode.
- **Late chunking** with a long-context embedding model, to preserve whole-document context without the intro/table split problem.
- A dedicated NER model instead of LLM-based entity/relationship extraction for GraphRAG, to cut cost and improve entity resolution.
- A larger, more diverse test set — the current 15 questions were enough to find and characterize failure modes, but too small for tight confidence intervals.
- Testing additional local models at the same size class (Llama 3.3 8B) to see whether the local-vs-cloud gap on multi-part questions is model-specific or general to this parameter range.

---

*Built as a learning project to understand RAG evaluation in depth — every failure mode above was found through actual debugging, not simulated for the writeup.*
