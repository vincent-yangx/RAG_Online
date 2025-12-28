# RAG_Online — Retrieval Pipeline (Dense/Sparse/Hybrid + Mullti Query + Rerank)

Structure：Support **Dense (FAISS)**、**Sparse (BM25)**、**Hybrid**、**Online (Tavily)**、**Hybrid+Online**
Components:
- **Multi-Query Query Rewriting**（HuggingFace Inference）
- **Multi-Query Fusion**（RRF + Score Fusion）
- **Cross-Encoder Reranker**（BGE Reranker）

---

## Features

- **Dense Retrieval**: BGE-M3 embedding + FAISS (FlatIP / HNSW)
- **Sparse Retrieval**: BM25 (rank_bm25)
- **Hybrid Retrieval**: Dense + Sparse Fusion（min-max + dynammic alpha）
- **Online Search**: convert Tavily result into chunks (apply dup domain penalty)
- **Hybrid + Online**: Local and Online Retrieval Fusion（score-level + rank-level）
- **Multi-Query Expansion**: Generate queries that have the exact meaning as the originall one
- **Multi-Query Fusion**: RRF + normalization score fusion
- **Rerank**: BGE Cross-Encoder


