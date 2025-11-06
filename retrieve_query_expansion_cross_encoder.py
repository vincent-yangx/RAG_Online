import os
import re
import json
import argparse
import pickle
import numpy as np
import faiss
import torch
from tqdm import tqdm
from collections import defaultdict
from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding import FlagReranker
from rank_bm25 import BM25Okapi
from huggingface_hub import InferenceClient

# ------------------ Argparse ------------------
parser = argparse.ArgumentParser(description="Dense/Sparse/Hybrid retrieval + Query Rewriting (Multi-Query)")
parser.add_argument("--chunk", type=str, required=True, help="dataset name, e.g., littleItaly")
parser.add_argument("--model", type=str, default="BAAI/bge-m3", help="embedding model, e.g., BAAI/bge-m3")
parser.add_argument("--questions", type=str, default=None, help="questions file (one per line). If None, infer from chunk")
parser.add_argument("--top_k", type=int, default=5, help="top-k to retrieve")
parser.add_argument("--retriever", type=str, default="hybrid", choices=["dense", "sparse", "hybrid"], help="retrieval mode")
parser.add_argument("--alpha", type=float, default=0.5, help="hybrid weight: final = alpha*dense + (1-alpha)*sparse (after per-query min-max)")
parser.add_argument("--index_type", type=str, default="flat", choices=["flat", "hnsw"], help="FAISS index type for dense")
parser.add_argument("--truncate", type=int, default=0, help="truncate retrieved text to N chars (0 = no truncation)")

# === New: Query rewriting ===
parser.add_argument("--rewrite", type=str, default="none", choices=["none", "mq"], help="query rewriting method")
parser.add_argument("--rewrite_n", type=int, default=4, help="number of rewritten queries for multi-query (excluding the original)")
parser.add_argument("--rrf_k", type=int, default=60, help="RRF hyperparam k (larger -> smoother)")
parser.add_argument("--out_dir", type=str, default="result_query_expansion_cross_encoding",
                    help="where to save retrieval results")

parser.add_argument("--rerank", type=str, default="bge",
                    choices=["none", "bge"], help="cross-encoder reranking")
parser.add_argument("--rerank_model", type=str, default="BAAI/bge-reranker-large",
                    help="cross-encoder model id")
parser.add_argument("--candidate_k", type=int, default=200,
                    help="num of candidates before rerank")
parser.add_argument("--rerank_batch_size", type=int, default=64,
                    help="batch size for reranker scoring")
parser.add_argument("--rerank_max_chars", type=int, default=1200,
                    help="truncate doc text for rerank scoring to avoid very long inputs")

args = parser.parse_args()

MODEL = args.model
model_name_simple = MODEL.split("/")[-1].replace(":", "_").replace("/", "_")
OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

# 路径与 embeder.py 保持一致
CHUNKS_PATH = f"data/chunks/chunks_{args.chunk}.jsonl"
EMB_PATH    = f"index/embeddings_{args.chunk}_{model_name_simple}.npy"
IDS_PATH    = f"index/ids_{args.chunk}_{model_name_simple}.npy"
FAISS_PATH  = f"index/faiss_index_{args.chunk}_{model_name_simple}.faiss"
BM25_PATH   = f"index/bm25_{args.chunk}.pkl"  # 仅与数据集相关，模型无关
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
_word_re = re.compile(r"[A-Za-z0-9]+")  # 简单英文 tokenizer；如有中文可接入 jieba
def tokenize(text: str):
    return [w.lower() for w in _word_re.findall(text)]

# ------------------ FAISS (dense) ------------------
def load_or_build_faiss(emb_path, faiss_path, index_type="flat"):
    if os.path.exists(faiss_path):
        index = faiss.read_index(faiss_path)
        print(f"Loaded FAISS index: {faiss_path} (ntotal={index.ntotal})")
        return index
    print("Loading embeddings...")
    embs = np.load(emb_path, mmap_mode="r").astype("float32")
    print(f"Embeddings shape: {embs.shape}")
    faiss.normalize_L2(embs)  # 用 IP 等价余弦
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

_EMBEDDER = None

def get_embedder(model_name):
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            _EMBEDDER = BGEM3FlagModel(model_name, use_fp16=True)
        except Exception:
            _EMBEDDER = BGEM3FlagModel(model_name, use_fp16=False)
    return _EMBEDDER

def embed_queries_bge(model_name, questions):
    model = get_embedder(model_name)
    q = model.encode(questions, batch_size=64)["dense_vecs"].astype("float32")
    faiss.normalize_L2(q)
    return q

def dense_scores_one(index, query_text, ids):
    q_emb = embed_queries_bge(MODEL, [query_text])
    D, I = index.search(q_emb, min(TOP_K*50, max(50, len(ids))))
    id_str = np.array([str(x) for x in ids])
    d = {}
    for s, i in zip(D[0], I[0]):
        if 0 <= i < len(id_str):
            d[id_str[i]] = float(s)
    return d

# ------------------ BM25 (sparse) ------------------
def load_or_build_bm25(ids, chunk_map, bm25_path):
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

def sparse_scores_one(bm25, query_text, ids):
    id_str = np.array([str(x) for x in ids])
    q_tok = tokenize(query_text)
    scores = bm25.get_scores(q_tok)  # np.array, shape=(N_docs,)
    return {id_str[i]: float(scores[i]) for i in range(len(id_str))}

# ------------------ Normalization & Fusion ------------------
def min_max_norm(score_dict):
    if not score_dict:
        return score_dict
    vals = list(score_dict.values())
    vmin, vmax = min(vals), max(vals)
    if vmax <= vmin:
        return {k: 0.0 for k in score_dict}
    return {k: (v - vmin) / (vmax - vmin) for k, v in score_dict.items()}

def hybrid_fuse_dense_sparse(dense_dict, sparse_dict, alpha=0.5):
    dn = min_max_norm(dense_dict)
    sn = min_max_norm(sparse_dict)
    keys = set(dn.keys()) | set(sn.keys())
    fused = {}
    for k in keys:
        dv = dn.get(k, 0.0)
        sv = sn.get(k, 0.0)
        fused[k] = alpha * dv + (1 - alpha) * sv
    return fused

def ranking_from_scores(score_dict):
    # 返回 doc_id 按分数降序的列表mat
    return [cid for cid, _ in sorted(score_dict.items(), key=lambda x: x[1], reverse=True)]

def rrf_fuse(rank_lists, k=60):
    scores = defaultdict(float)
    for rl in rank_lists:
        for r, doc in enumerate(rl):
            scores[doc] += 1.0 / (k + r + 1)
    # 直接返回 dict，后续 materialize_topk 会排序
    return scores

# ------------------ Query Rewriting (Multi-Query) ------------------
def multi_query_expand(
    query: str,
    n: int = 4,
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    timeout: int = 60,
):
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        print("[WARN] HUGGINGFACEHUB_API_TOKEN not set. Fallback to original query only.")
        return [query]

    try:
        client = InferenceClient(model=model_id, token=hf_token, timeout=timeout)

        system_msg = (
            "Rewrite the user's search query into diverse but semantically equivalent queries. "
            "Do not add new facts or change meaning. Reply ONLY with a JSON array of strings."
        )
        user_msg = (
            f"Original query:\n{query}\n\n"
            f"Please produce {n} diverse, short, retrieval-friendly rewrites. "
            "Output JSON only."
        )

        resp = client.chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )

        # 取文本
        text = resp.choices[0].message["content"] if isinstance(resp.choices[0].message, dict) \
               else resp.choices[0].message.content

        # 尝试 JSON 解析，失败则回退逐行
        try:
            l, r = text.rfind("["), text.rfind("]")
            payload = text[l:r+1] if (l != -1 and r != -1 and r > l) else text
            rewrites = json.loads(payload)
        except Exception:
            lines = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
            rewrites = [ln for ln in lines if ln]

        # 清洗与去重
        seen = {query.lower()}
        out = [query]
        for r in rewrites:
            if isinstance(r, str):
                cand = r.strip()
                if cand and cand.lower() not in seen:
                    seen.add(cand.lower())
                    out.append(cand)
            if len(out) >= n + 1:
                break
        return out

    except Exception as e:
        print(f"[WARN] HF chat_completion failed: {e}. Fallback to original query only.")
        return [query]

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

# 从分数字典中取候选列表（不截断文本，避免影响打分）
def candidates_from_scores(score_dict, chunk_map, candidate_k=200):
    items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:candidate_k]
    docs = []
    for cid, s in items:
        ch = chunk_map.get(str(cid))
        if not ch:
            continue
        docs.append({
            "chunk_id": str(cid),
            "pre_score": float(s),          # 融合前的分数（用于对照）
            "source": ch.get("source", ""),
            "text": ch.get("text", "")
        })
    return docs

# 懒加载一个全局 reranker，避免每条 query 都重复载入模型
_RERANKER = None
def get_reranker(model_name: str = "BAAI/bge-reranker-large", use_fp16: bool = True):
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = FlagReranker(model_name, use_fp16=use_fp16)
    return _RERANKER

def bge_cross_encoder_rerank(query: str,
                             docs: list,
                             top_k: int = 5,
                             model_name: str = "BAAI/bge-reranker-large",
                             batch_size: int = 64,
                             max_chars: int = 1200):

    import numpy as np
    reranker = get_reranker(model_name)
    pairs = [[query, d["text"][:max_chars] if d["text"] else ""] for d in docs]
    scores = reranker.compute_score(pairs, batch_size=batch_size)  # list/np.array
    scores = np.array(scores, dtype="float32")
    order = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in order:
        d = docs[int(i)].copy()
        d["ce_score"] = float(scores[int(i)])
        results.append(d)
    return results

# ------------------ Main ------------------
def main():
    # 1) load chunks / ids
    chunk_map = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunk_map)} chunks from {CHUNKS_PATH}")

    ids = np.load(IDS_PATH, allow_pickle=True)
    ids = np.array([str(x) for x in ids])  # 保持为 str

    # 2) prepare retrievers
    index = bm25 = None
    if args.retriever in ("dense", "hybrid"):
        index = load_or_build_faiss(EMB_PATH, FAISS_PATH, index_type=args.index_type)
    if args.retriever in ("sparse", "hybrid"):
        bm25 = load_or_build_bm25(ids, chunk_map, BM25_PATH)

    # 3) load questions
    questions = load_questions(QUESTIONS_PATH)
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH}")

    # 4) iterate queries
    out_txt = os.path.join(
    OUT_DIR,
    f"retrieval_info_{args.retriever}_{args.rewrite}_{args.chunk}.txt"
)
    out_json = os.path.join(OUT_DIR, f"retrieval_{args.chunk}.jsonl")

    with open(out_txt, "w", encoding="utf-8") as f, open(out_json, "w", encoding="utf-8") as jf:
        for qi, q in enumerate(questions, 1):
            # --- 多路查询改写 ---
            if args.rewrite == "mq":
                q_variants = multi_query_expand(q, n=args.rewrite_n)  # [q, q1, q2, ...]
            else:
                q_variants = [q]

            rank_lists = []
            for cq in q_variants:
                if args.retriever == "dense":
                    sdict = dense_scores_one(index, cq, ids)
                elif args.retriever == "sparse":
                    sdict = sparse_scores_one(bm25, cq, ids)
                else:
                    dd = dense_scores_one(index, cq, ids)
                    sd = sparse_scores_one(bm25, cq, ids)
                    sdict = hybrid_fuse_dense_sparse(dd, sd, alpha=args.alpha)
                rank_lists.append(ranking_from_scores(sdict)[:200])

            fused_scores = rrf_fuse(rank_lists, k=args.rrf_k)

            # --- Rerank 阶段 ---
            if args.rerank == "bge":
                doc_candidates = candidates_from_scores(
                    fused_scores, chunk_map, candidate_k=min(args.candidate_k, len(fused_scores))
                )
                reranked = bge_cross_encoder_rerank(
                    q,
                    doc_candidates,
                    top_k=TOP_K,
                    model_name=args.rerank_model,
                    batch_size=args.rerank_batch_size,
                    max_chars=args.rerank_max_chars
                )
                results = []
                for rank, d in enumerate(reranked, 1):
                    text = d["text"]
                    if args.truncate and len(text) > args.truncate:
                        text = text[:args.truncate] + "..."
                    results.append({
                        "rank": rank,
                        "chunk_id": d["chunk_id"],
                        "score": d["ce_score"],
                        "pre_score": d["pre_score"],
                        "source": d["source"],
                        "text": text
                    })
            else:
                results = materialize_topk(fused_scores, chunk_map, top_k=TOP_K, truncate=args.truncate)

            # --- 写入 TXT 文件 ---
            f.write("=" * 80 + "\n")
            f.write(f"[Q{qi}] {q}\n")
            if args.rewrite == "mq":
                rewrites_only = q_variants[1:]
                if rewrites_only:
                    f.write("\n".join(rewrites_only) + "\n\n")
            for r in results:
                if "pre_score" in r:
                    f.write(f"  Rank {r['rank']}: ce_score={r['score']:.4f} | pre_score={r['pre_score']:.4f}\n")
                else:
                    f.write(f"  Rank {r['rank']}: score={r['score']:.4f}\n")
                f.write(f"  chunk_id={r['chunk_id']} | source={r['source']}\n")
                f.write(f"  text: {r['text']}\n\n")

            # --- 写入 JSON 文件（移到循环内部）---
            entry = {
                "qid": qi,
                "question": q,
                "retrieved": results
            }
            jf.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Written results to {out_txt}")
    print(f"Saved JSON results to {out_json}")


if __name__ == "__main__":
    main()
