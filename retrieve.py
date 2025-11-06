import os
import re
import json
import argparse
import pickle
import numpy as np
import faiss
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi

# ------------------ Argparse ------------------
parser = argparse.ArgumentParser(description="Dense/Sparse/Hybrid retrieval with FAISS (BGE) + BM25")
parser.add_argument("--chunk", type=str, required=True, help="dataset name, e.g., littleItaly")
parser.add_argument("--model", type=str, default="BAAI/bge-m3", help="embedding model, e.g., BAAI/bge-m3")
parser.add_argument("--questions", type=str, default=None, help="questions file (one per line). If None, auto infer from chunk")
parser.add_argument("--top_k", type=int, default=5, help="top-k to retrieve")
parser.add_argument("--retriever", type=str, default="dense", choices=["dense", "sparse", "hybrid"], help="retrieval mode")
parser.add_argument("--alpha", type=float, default=0.5, help="hybrid weight: final = alpha*dense + (1-alpha)*sparse (after per-query min-max)")
parser.add_argument("--index_type", type=str, default="flat", choices=["flat", "hnsw"], help="FAISS index type for dense")
parser.add_argument("--truncate", type=int, default=0, help="truncate retrieved text to N chars (0 = no truncation)")

args = parser.parse_args()

MODEL = args.model
model_name_simple = MODEL.split("/")[-1].replace(":", "_").replace("/", "_")


CHUNKS_PATH = f"data/chunks/chunks_{args.chunk}.jsonl"
EMB_PATH    = f"index/embeddings_{args.chunk}_{model_name_simple}.npy"
IDS_PATH    = f"index/ids_{args.chunk}_{model_name_simple}.npy"
FAISS_PATH  = f"index/faiss_index_{args.chunk}_{model_name_simple}.faiss"
BM25_PATH   = f"index/bm25_{args.chunk}.pkl" 
QUESTIONS_PATH = args.questions or f"data/test/question_{args.chunk}.txt"
TOP_K = args.top_k

# ------------------ IO helpers ------------------
def load_chunks(path):
    """Load chunk data, return a dict: str(chunk_id) -> chunk_dict"""
    chunk_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunk_map[str(d["chunk_id"])] = d
    return chunk_map

def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# ------------------ Text utils ------------------
_word_re = re.compile(r"[A-Za-z0-9]+")
def tokenize(text: str):
    return [w.lower() for w in _word_re.findall(text)]

# ------------------ FAISS (dense) ------------------
def load_or_build_faiss(emb_path, faiss_path, index_type="flat"):
    if os.path.exists(faiss_path):
        index = faiss.read_index(faiss_path)
        print(f"Loaded FAISS index: {faiss_path} (ntotal={index.ntotal})")
        return index
    print("Loading embeddings...")
    embs = np.load(emb_path).astype("float32")
    print(f"Embeddings shape: {embs.shape}")
    faiss.normalize_L2(embs) 
    dim = embs.shape[1]
    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
    index.add(embs)
    faiss.write_index(index, faiss_path)
    print(f"Built & saved FAISS index: {faiss_path} (ntotal={index.ntotal})")
    return index

def embed_queries_bge(model_name, questions):
    try:
        model = BGEM3FlagModel(model_name, use_fp16=True)
    except Exception:
        model = BGEM3FlagModel(model_name, use_fp16=False)
    q = model.encode(questions, batch_size=64)["dense_vecs"].astype("float32")
    faiss.normalize_L2(q)
    return q

def dense_scores(index, query_embs, ids):
    """
    返回每个 query 的 {chunk_id: score} 字典，便于后续融合。
    """
    D, I = index.search(query_embs, min(TOP_K*50, max(50, len(ids))))  # 多取一些供融合
    id_str = np.array([str(x) for x in ids])
    all_scores = []
    for scores, idxs in zip(D, I):
        d = {}
        for s, i in zip(scores, idxs):
            if 0 <= i < len(id_str):
                d[id_str[i]] = float(s)
        all_scores.append(d)
    return all_scores

# ------------------ BM25 (sparse) ------------------
def load_or_build_bm25(ids, chunk_map, bm25_path):
    """
    语料严格按 ids 顺序对齐，确保与向量库一一对应。
    """
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        print(f"Loaded BM25 from {bm25_path}")
        return bm25

    corpus_tokens = []
    for cid in ids:
        c = chunk_map.get(str(cid))
        text = c.get("text", "") if c else ""
        corpus_tokens.append(tokenize(text))
    bm25 = BM25Okapi(corpus_tokens)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"Built & saved BM25 to {bm25_path}")
    return bm25

def sparse_scores_bm25(bm25, queries, ids):
    """
    返回每个 query 的 {chunk_id: score} 字典。
    BM25Okapi.get_scores 返回与语料顺序一致的分数数组。
    """
    id_str = np.array([str(x) for x in ids])
    all_scores = []
    for q in queries:
        q_tok = tokenize(q)
        scores = bm25.get_scores(q_tok)  # np.array, shape=(N_docs,)
        d = {id_str[i]: float(scores[i]) for i in range(len(id_str))}
        all_scores.append(d)
    return all_scores

# ------------------ Hybrid fuse ------------------
def min_max_norm(score_dict):
    """把单个 query 的 {id: score} 做 min-max 归一化到 [0,1]。全相等则全部置 0。"""
    if not score_dict:
        return score_dict
    vals = list(score_dict.values())
    vmin, vmax = min(vals), max(vals)
    if vmax <= vmin:
        return {k: 0.0 for k in score_dict}
    return {k: (v - vmin) / (vmax - vmin) for k, v in score_dict.items()}

def hybrid_fuse(dense_dict, sparse_dict, alpha=0.5):
    """
    先对 dense / sparse 分数各自做 per-query min-max，再加权和。
    只要某边缺失，就用另一边的归一化值。
    """
    dn = min_max_norm(dense_dict)
    sn = min_max_norm(sparse_dict)
    keys = set(dn.keys()) | set(sn.keys())
    fused = {}
    for k in keys:
        dv = dn.get(k, 0.0)
        sv = sn.get(k, 0.0)
        fused[k] = alpha * dv + (1 - alpha) * sv
    return fused

# ------------------ Assemble results ------------------
def materialize_topk(score_dict, chunk_map, top_k=5, truncate=0):
    items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for rank, (cid, s) in enumerate(items, 1):
        ch = chunk_map.get(str(cid))
        if not ch:
            continue
        text = ch.get("text", "")
        if truncate and len(text) > truncate:
            text = text[:truncate] + "..."
        results.append({
            "rank": rank,
            "chunk_id": str(cid),
            "score": float(s),
            "source": ch.get("source", ""),
            "text": text
        })
    return results

# ------------------ Main ------------------
def main():
    # 1) load chunks / ids
    chunk_map = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunk_map)} chunks from {CHUNKS_PATH}")

    ids = np.load(IDS_PATH, allow_pickle=True)
    ids = np.array([str(x) for x in ids])  # 保持为 str，后续统一

    # 2) load questions
    questions = load_questions(QUESTIONS_PATH)
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH}")

    # 3) prepare retrievers
    dense_dicts = sparse_dicts = None

    if args.retriever in ("dense", "hybrid"):
        index = load_or_build_faiss(EMB_PATH, FAISS_PATH, index_type=args.index_type)
        q_embs = embed_queries_bge(MODEL, questions)
        dense_dicts = dense_scores(index, q_embs, ids)

    if args.retriever in ("sparse", "hybrid"):
        bm25 = load_or_build_bm25(ids, chunk_map, BM25_PATH)
        sparse_dicts = sparse_scores_bm25(bm25, questions, ids)

    # 4) score & output
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    out_txt = os.path.join(
    output_dir, 
    f"retrieval_info_{args.retriever}_{args.chunk}_{model_name_simple}.txt"
    )

    with open(out_txt, "w", encoding="utf-8") as f:
        for qi, q in enumerate(questions, 1):
            f.write("=" * 80 + "\n")
            f.write(f"[Q{qi}] {q}\n\n")

            if args.retriever == "dense":
                sdict = dense_dicts[qi-1]
            elif args.retriever == "sparse":
                sdict = sparse_dicts[qi-1]
            else:
                sdict = hybrid_fuse(dense_dicts[qi-1], sparse_dicts[qi-1], alpha=args.alpha)

            results = materialize_topk(sdict, chunk_map, top_k=TOP_K, truncate=args.truncate)
            for r in results:
                f.write(f"  Rank {r['rank']}: score={r['score']:.4f}\n")
                f.write(f"  chunk_id={r['chunk_id']} | source={r['source']}\n")
                f.write(f"  text: {r['text']}\n\n")

    print(f"✅ Written results to {out_txt}")

if __name__ == "__main__":
    main()
