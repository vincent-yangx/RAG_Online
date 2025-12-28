# RAG_Online — Retrieval Pipeline (Dense/Sparse/Hybrid + Multi-Query + Rerank)

A modular retrieval pipeline for RAG and QA. It supports **Dense (FAISS)**, **Sparse (BM25)**, **Hybrid**, **Online Search (Tavily)**, and **Hybrid+Online** retrieval, with optional **Multi-Query rewriting** and **Cross-Encoder reranking**.

This repo is designed for experimenting with retrieval strategies (local + web) and improving precision via reranking.  
**Work in progress:** extending to **multi-hop retrieval** (e.g., HotpotQA-style hop-1/hop-2 evidence).

---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Data Format](#data-format)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Quickstart](#quickstart)
- [Arguments](#arguments)
- [Output Format](#output-format)
- [Notes & Reproducibility](#notes--reproducibility)
- [Work in Progress: Multi-hop Retrieval](#work-in-progress-multi-hop-retrieval)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

### Retrieval Modes
- **Dense Retrieval**: BGE-M3 embeddings + FAISS (FlatIP / HNSW)
- **Sparse Retrieval**: BM25 (rank_bm25)
- **Hybrid Retrieval**: Dense + Sparse fusion (min-max normalization + dynamic alpha)
- **Online Retrieval**: Tavily results converted into temporary chunks  
  - Optional **domain-dup penalty** to reduce repeated domains in top results
- **Hybrid + Online**: Local + Online fusion (score-level + rank-level)

### Query Expansion & Fusion
- **Multi-Query Rewriting** (HuggingFace Inference): generate diverse rewrites that preserve the original intent
- **Multi-Query Fusion**:
  - RRF (rank-based fusion)
  - normalized score fusion (score-based fusion)

### Precision Improvement
- **Cross-Encoder Reranking**: BGE reranker to reorder candidate documents

---

## Project Structure

Recommended layout (matches default paths in the script):

```
.
├── data/
│   ├── chunks/
│   │   └── chunks_<dataset>.jsonl
│   └── test/
│       └── question_<dataset>.txt
├── index/
│   ├── embeddings_<dataset>_<model>.npy
│   ├── ids_<dataset>_<model>.npy
│   ├── faiss_index_<dataset>_<model>.faiss
│   └── bm25_<dataset>.pkl
├── rag_pipeline.py
└── requirements.txt
```

- `chunks_<dataset>.jsonl`: pre-built chunk collection
- `question_<dataset>.txt`: one query per line
- `index/`: saved embeddings, ids, and indexes for fast retrieval

---

## Data Format

### 1) Chunk file (`data/chunks/chunks_<dataset>.jsonl`)
Each line is a JSON object with at least:

```json
{"chunk_id":"123", "source":"...", "text":"..."}
```

- `chunk_id`: unique identifier (string or int)
- `source`: URL / file / title, used for logging and debugging
- `text`: chunk text content (string)

### 2) Questions file (`data/test/question_<dataset>.txt`)
One query per line:

```
What is ...?
Who invented ...?
...
```

---

## Installation

Recommended Python version: **3.10+**

### 1) Create environment (optional)
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### FAISS notes
- CPU: `faiss-cpu`
- GPU: `faiss-gpu` (requires CUDA and matching versions)

If FAISS installation fails, try installing it separately.

---

## Environment Variables

### Tavily (Online Search)
To enable `online` and `hybrid_online` retrievers:

```bash
export TAVILY_API_KEY="xxxx"
```

### HuggingFace Token (Multi-Query Rewriting)
To enable `--rewrite mq` with HuggingFace Inference:

```bash
export HUGGINGFACE_HUB_TOKEN="hf_xxx"
# or
export HUGGINGFACEHUB_API_TOKEN="hf_xxx"
```

If no token is set, the script will fall back to using only the original query.

---

## Quickstart

> Replace `<dataset>` with your dataset name (matching `chunks_<dataset>.jsonl` and `question_<dataset>.txt`).

### 1) Hybrid + Multi-Query + Rerank (recommended default)
```bash
python rag_pipeline.py \
  --chunk <dataset> \
  --retriever hybrid \
  --rewrite mq --rewrite_n 4 \
  --rerank bge --top_k 5 \
  --out_dir results/
```

### 2) Dense only
```bash
python rag_pipeline.py \
  --chunk <dataset> \
  --retriever dense \
  --rewrite none \
  --rerank bge \
  --top_k 5 \
  --out_dir results/
```

### 3) Sparse (BM25) only
```bash
python rag_pipeline.py \
  --chunk <dataset> \
  --retriever sparse \
  --rewrite none \
  --rerank none \
  --top_k 5 \
  --out_dir results/
```

### 4) Hybrid + Online (Tavily)
```bash
python rag_pipeline.py \
  --chunk <dataset> \
  --retriever hybrid_online \
  --rewrite mq --rewrite_n 4 \
  --rerank bge \
  --tavily_key $TAVILY_API_KEY \
  --top_k 5 \
  --out_dir results/
```

---

## Arguments

Common arguments:

- `--chunk`: dataset name (required)
- `--model`: embedding model (default: `BAAI/bge-m3`)
- `--questions`: optional path to question file (default inferred from chunk)
- `--top_k`: final number of retrieved documents (default: 5)

Retrieval selection:

- `--retriever`: `dense | sparse | hybrid | online | hybrid_online`
- `--index_type`: FAISS index type: `flat | hnsw`
- `--alpha`: hybrid fusion base weight for dense (dynamic adjustment may apply)
- `--truncate`: truncate retrieved text to N characters (0 = no truncation)

Multi-query rewriting:

- `--rewrite`: `none | mq`
- `--rewrite_n`: number of rewrites to generate (excluding original)
- `--rrf_k`: RRF hyperparameter (larger → smoother)

Reranking:

- `--rerank`: `none | bge`
- `--rerank_model`: default `BAAI/bge-reranker-large`
- `--candidate_k`: number of candidates before rerank
- `--rerank_batch_size`: default 64
- `--rerank_max_chars`: truncate doc text for rerank scoring

Online search:

- `--tavily_key`: Tavily API key (default reads `TAVILY_API_KEY` env)

> Some extended versions of the script may include:
> - score/rank fusion weight (e.g., `--mq_fuse_beta`)
> - online domain-dup penalty controls (e.g., `--online_dedup_gamma`)
> - hybrid_online online-weight (e.g., `--hybrid_online_w`)

---

## Output Format

The script writes two files into `--out_dir`:

### 1) Human-readable log
`retrieval_info_<retriever>_<rewrite>_<chunk>.txt`

Includes:
- question
- rewrites (if enabled)
- top-k retrieved chunks with scores, sources, and text snippets

### 2) JSONL output
`retrieval_<chunk>.jsonl`

One line per query:
```json
{
  "qid": 1,
  "question": " ... ",
  "retrieved": [
    {
      "rank": 1,
      "chunk_id": "123",
      "score": 0.91,
      "pre_score": 0.74,
      "source": "...",
      "text": "..."
    }
  ]
}
```

- `score`: final score (if rerank enabled, typically cross-encoder score)
- `pre_score`: score before reranking (useful for debugging; may not appear when rerank is disabled)

---

## Notes & Reproducibility

- **Online retrieval is non-deterministic**: Tavily results may change over time.
- For fair comparisons, fix:
  - `--rewrite`, `--rewrite_n`
  - `--candidate_k`, `--rrf_k`
  - `--top_k`, `--alpha`, `--index_type`
- If using rerank, runtime will be dominated by the reranker. Use smaller `--candidate_k` to speed up.

---

## Work in Progress: Multi-hop Retrieval

I am currently extending this pipeline for **multi-hop retrieval** (e.g., HotpotQA-style evidence):

Planned modules:
- **Hop-1 retrieval**: retrieve initial evidence
- **Query decomposition / bridge extraction**: generate follow-up query from hop-1 evidence
- **Hop-2 retrieval**: retrieve complementary evidence
- **Path scoring**: score (doc1, doc2) evidence chains
- **Supporting-facts selection**: sentence-level selection for HotpotQA supporting facts
- **Multi-hop evaluation**: Recall@K, chain recall, supporting-facts F1

Updates and experiments will be added in future commits.

---

## License
MIT

---

## Acknowledgements
This project uses:
- **FAISS** for vector search
- **rank_bm25** for BM25 retrieval
- **FlagEmbedding (BGE)** for embeddings and reranking
- **Tavily** for online search (optional)
- **HuggingFace Inference API** for query rewriting (optional)
