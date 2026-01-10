import os
import re
import json
import argparse
import pickle
import numpy as np
import faiss
from tqdm import tqdm
from collections import defaultdict, ChainMap
from urllib.parse import urlparse
import requests

from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding import FlagReranker
from rank_bm25 import BM25Okapi
from huggingface_hub import InferenceClient

 

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


_word_re = re.compile(r"[A-Za-z0-9']+", flags=re.UNICODE)

def tokenize(text: str):
    if not text:
        return []
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


_EMBEDDER = None
def get_embedder(model_name: str):
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            _EMBEDDER = BGEM3FlagModel(model_name, use_fp16=True)
        except Exception:
            _EMBEDDER = BGEM3FlagModel(model_name, use_fp16=False)
    return _EMBEDDER


def embed_queries_bge(model_name: str, queries: list[str]):
    model = get_embedder(model_name)
    q = model.encode(queries, batch_size=64)["dense_vecs"].astype("float32")
    faiss.normalize_L2(q)
    return q


def dense_scores_batch(index, queries: list[str], ids: np.ndarray, *, model_name: str, top_k: int):
    """
    Batch embed + batch faiss search
    Return: list[dict] same length as queries, each dict {chunk_id: score}
    """
    if not queries:
        return []

    q_emb = embed_queries_bge(model_name, queries)
    search_k = min(top_k * 50, max(50, len(ids)))
    D, I = index.search(q_emb, search_k)

    id_str = np.array([str(x) for x in ids])

    D = np.clip(D, 0, 1)
    D = np.power(D, 1.5)

    outs = []
    for row in range(len(queries)):
        dct = {}
        for s, i in zip(D[row], I[row]):
            ii = int(i)
            if 0 <= ii < len(id_str):
                dct[id_str[ii]] = float(s)
        outs.append(dct)
    return outs

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


def sparse_scores_one(bm25, query_text: str, ids: np.ndarray):
    id_str = np.array([str(x) for x in ids])
    q_tok = tokenize(query_text)
    scores = bm25.get_scores(q_tok)
    return {id_str[i]: float(scores[i]) for i in range(len(id_str))}


# ------------------- Online Search --------------------------

def online_search_raw(query: str, top_k: int, api_key: str):
    if not api_key:
        print("[WARN] TAVILY_API_KEY not set. Skip online search.")
        return []

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": api_key,
                "query": query,
                "num_results": top_k,
                "include_domains": None,
                "include_answer": False,
                "include_images": False,
                "search_depth": "advanced"
            },
            timeout=20
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])[:top_k]
    except Exception as e:
        print(f"[WARN] Tavily search failed: {e}")
        return []


def online_scores_one(query_text: str, *, top_k: int, api_key: str,
                      cid_prefix: str, source_tag: str = "tavily",
                      dedup_gamma: float = 0.9):
    """
    Keep api_score and apply simple domain-dup penalty.
    Return:
      scores: {cid: score}
      web_map: {cid: {chunk_id/source/text}}
    """
    raw = online_search_raw(query_text, top_k=top_k, api_key=api_key)
    scores = {}
    web_map = {}

    domain_cnt = defaultdict(int)

    for rank, item in enumerate(raw):
        title = item.get("title", "") or ""
        url = item.get("url", "") or ""
        snippet = item.get("content", "") or ""
        api_score = item.get("score", None)

        cid = f"{cid_prefix}_{rank}"
        text = snippet if snippet else title

        # domain dedup penalty
        domain = ""
        if url:
            try:
                domain = urlparse(url).netloc.lower()
            except Exception:
                domain = ""
        if domain:
            domain_cnt[domain] += 1
            penalty = (dedup_gamma ** (domain_cnt[domain] - 1))
        else:
            penalty = 1.0

        base_score = float(api_score) if api_score is not None else (1.0 / (rank + 1))
        final_score = base_score * penalty

        web_map[cid] = {
            "chunk_id": cid,
            "source": url if url else source_tag,
            "text": text,
        }
        scores[cid] = float(final_score)

    return scores, web_map


# ------------------ Normalization & Fusion ------------------

def min_max_norm(score_dict: dict):
    if not score_dict:
        return {}
    vals = list(score_dict.values())
    vmin, vmax = min(vals), max(vals)
    if vmax <= vmin:
        return {k: 0.0 for k in score_dict}
    return {k: (v - vmin) / (vmax - vmin) for k, v in score_dict.items()}


def ranking_from_scores(score_dict):
    return [cid for cid, _ in sorted(score_dict.items(), key=lambda x: x[1], reverse=True)]


def rrf_fuse(rank_lists, k=60):
    scores = defaultdict(float)
    for rl in rank_lists:
        for r, doc in enumerate(rl):
            scores[doc] += 1.0 / (k + r + 1)
    return dict(scores)


def hybrid_fuse_dense_sparse(dense_dict, sparse_dict, alpha=0.8):
    """
    Dense+Sparse fusion (dynamic alpha clamped).
    """
    dn = min_max_norm(dense_dict)
    sn = min_max_norm(sparse_dict)
    keys = set(dn.keys()) | set(sn.keys())

    fused = {}
    for k in keys:
        dv = dn.get(k, 0.0)
        sv = sn.get(k, 0.0)
        # ！！！ need to check if the weight is suitable under common circumstances
        w = alpha + 0.2 * (dv - sv)
        w = max(0.0, min(1.0, w))
        fused[k] = w * dv + (1.0 - w) * sv
    return fused


def fuse_multiquery_rank_and_score(score_dicts: list[dict], rank_lists: list[list[str]], *, rrf_k: int, beta: float):
    """
    Score (min-max) + rank (RRF) mix fusion across query variants.
    beta: weight on score-fusion; (1-beta) on rank-fusion.
    """
    # score fusion
    score_sum = defaultdict(float)
    for sd in score_dicts:
        ns = min_max_norm(sd)
        for k, v in ns.items():
            score_sum[k] += float(v)
    score_sum = dict(score_sum)
    score_sum_norm = min_max_norm(score_sum)

    # rank fusion
    rrf_scores = rrf_fuse(rank_lists, k=rrf_k)
    rrf_norm = min_max_norm(rrf_scores)

    keys = set(score_sum_norm.keys()) | set(rrf_norm.keys())
    out = {}
    for k in keys:
        out[k] = beta * score_sum_norm.get(k, 0.0) + (1.0 - beta) * rrf_norm.get(k, 0.0) 
    return out

# ------------------ Query Rewriting (Multi-Query) ------------------

def multi_query_expand(
    query: str,
    n: int = 4,
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    timeout: int = 60,
):
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("[WARN] HF token not set (HUGGINGFACEHUB_API_TOKEN/HUGGINGFACE_HUB_TOKEN). Fallback to original query only.")
        return [query]

    try:
        client = InferenceClient(model=model_id, token=token, timeout=timeout)

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

        text = resp.choices[0].message["content"] if isinstance(resp.choices[0].message, dict) \
               else resp.choices[0].message.content

        try:
            l, r = text.rfind("["), text.rfind("]")
            payload = text[l:r+1] if (l != -1 and r != -1 and r > l) else text
            rewrites = json.loads(payload)
        except Exception:
            lines = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
            rewrites = [ln for ln in lines if ln]

        seen = {query.lower()}
        out = [query]
        for rr in rewrites:
            if isinstance(rr, str):
                cand = rr.strip()
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
        text = ch.get("text", "") or ""
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


def candidates_from_ids(candidate_ids: list[str], fused_scores: dict, chunk_map, *, truncate: int = 0):
    """
    Build candidate docs list for reranker using an explicit candidate id list.
    pre_score uses fused_scores[cid].
    """
    docs = []
    for cid in candidate_ids:
        ch = chunk_map.get(str(cid))
        if not ch:
            continue
        text = ch.get("text", "") or ""
        if truncate and len(text) > truncate:
            text = text[:truncate] + "..."
        docs.append({
            "chunk_id": str(cid),
            "pre_score": float(fused_scores.get(str(cid), 0.0)),
            "source": ch.get("source", ""),
            "text": text,
        })
    return docs


# ------------------ Reranker ------------------

_RERANKER = None
def get_reranker(model_name: str = "BAAI/bge-reranker-large", use_fp16: bool = True):
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = FlagReranker(model_name, use_fp16=use_fp16)
    return _RERANKER


def bge_cross_encoder_rerank(query: str,
                            docs: list,
                            top_k: int,
                            model_name: str,
                            batch_size: int,
                            max_chars: int):
    if not docs:
        return []

    reranker = get_reranker(model_name)
    pairs = [[query, (d.get("text") or "")[:max_chars]] for d in docs]
    scores = reranker.compute_score(pairs, batch_size=batch_size)
    scores = np.array(scores, dtype="float32")
    order = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in order:
        d = docs[int(i)].copy()
        d["ce_score"] = float(scores[int(i)])
        results.append(d)
    return results


# ------------------ Pipeline ------------------

class RAGPipeline:
    def __init__(self, args):
        self.args = args
        self.model_name_simple = args.model.split("/")[-1].replace(":", "_").replace("/", "_")

        self.paths = {
            "chunks": f"data/chunks/chunks_{args.chunk}.jsonl",
            "emb":    f"index/embeddings_{args.chunk}_{self.model_name_simple}.npy",
            "ids":    f"index/ids_{args.chunk}_{self.model_name_simple}.npy",
            "faiss":  f"index/faiss_index_{args.chunk}_{self.model_name_simple}.faiss",
            "bm25":   f"index/bm25_{args.chunk}.pkl",
            "questions": args.questions or f"data/test/question_{args.chunk}.txt"
        }

        print(f"Loading chunks from {self.paths['chunks']}...")
        self.chunk_map = load_chunks(self.paths["chunks"])
        print(f"Loaded {len(self.chunk_map)} chunks.")

        print(f"Loading IDs from {self.paths['ids']}...")
        self.ids = np.load(self.paths["ids"], allow_pickle=True)
        self.ids = np.array([str(x) for x in self.ids])

        self.index = None
        self.bm25 = None
        self._load_resources()

    def _load_resources(self):
        if self.args.retriever in ("dense", "hybrid", "hybrid_online"):
            self.index = load_or_build_faiss(self.paths["emb"], self.paths["faiss"], index_type=self.args.index_type)

        if self.args.retriever in ("sparse", "hybrid", "hybrid_online"):
            self.bm25 = load_or_build_bm25(self.ids, self.chunk_map, self.paths["bm25"])

        if self.args.rerank == "bge":
            print(f"Reranker enabled: {self.args.rerank_model}")

    def retrieve_single(self, query: str, *, qid: int):
        # Multi-query expansion
        if self.args.rewrite == "mq":
            q_variants = multi_query_expand(query, n=self.args.rewrite_n)
        else:
            q_variants = [query]

        # Prepare per-variant dense/sparse
        dense_list = [None] * len(q_variants)
        sparse_list = [None] * len(q_variants)

        if self.args.retriever in ("dense", "hybrid", "hybrid_online"):
            if self.index is None:
                raise RuntimeError("Dense retriever requested but FAISS index not loaded.")
            dense_list = dense_scores_batch(
                self.index,
                q_variants,
                self.ids,
                model_name=self.args.model,
                top_k=self.args.top_k
            )

        if self.args.retriever in ("sparse", "hybrid", "hybrid_online"):
            if self.bm25 is None:
                raise RuntimeError("Sparse retriever requested but BM25 not loaded.")
            for i, cq in enumerate(q_variants):
                sparse_list[i] = sparse_scores_one(self.bm25, cq, self.ids)

        # For each variant, build score_dict + rank_list, and collect seed candidates
        score_dicts = []    # for score-fusion
        rank_lists = []     # for rrf
        extra_web_map = {}  # online docs
        seed_candidate_ids = set()

        seed_n = max(200, self.args.top_k * 40, self.args.candidate_k)

        for v_idx, cq in enumerate(q_variants):
            dd = dense_list[v_idx] if dense_list[v_idx] is not None else {}
            sd = sparse_list[v_idx] if sparse_list[v_idx] is not None else {}

            # build variant score dict
            if self.args.retriever == "dense":
                sdict = dd

            elif self.args.retriever == "sparse":
                sdict = sd

            elif self.args.retriever == "hybrid":
                sdict = hybrid_fuse_dense_sparse(dd, sd, alpha=self.args.alpha)

            elif self.args.retriever == "online":
                cid_prefix = f"web_q{qid}_v{v_idx}"
                online_scores, web_map = online_scores_one(
                    cq,
                    top_k=200,
                    api_key=self.args.tavily_key,
                    cid_prefix=cid_prefix,
                    dedup_gamma=self.args.online_dedup_gamma
                )
                extra_web_map.update(web_map)
                sdict = online_scores

            elif self.args.retriever == "hybrid_online":
                # local hybrid
                local_scores = hybrid_fuse_dense_sparse(dd, sd, alpha=self.args.alpha)

                # online
                cid_prefix = f"web_q{qid}_v{v_idx}"
                online_scores, web_map = online_scores_one(
                    cq,
                    top_k=100,
                    api_key=self.args.tavily_key,
                    cid_prefix=cid_prefix,
                    dedup_gamma=self.args.online_dedup_gamma
                )
                extra_web_map.update(web_map)

                # Normalize + weighted combination
                ln = min_max_norm(local_scores)
                on = min_max_norm(online_scores)
                w = float(self.args.hybrid_online_w)

                score_level = {}
                keys = set(ln.keys()) | set(on.keys())
                for k in keys:
                    score_level[k] = (1.0 - w) * ln.get(k, 0.0) + w * on.get(k, 0.0)

                # Rank level combination
                local_rank = ranking_from_scores(local_scores)[:200]
                online_rank = ranking_from_scores(online_scores)[:200]
                rank_level = min_max_norm(rrf_fuse([local_rank, online_rank], k=self.args.rrf_k))

                # Mix score + rank at variant level using same beta
                beta = float(self.args.mq_fuse_beta)
                sdict = {}
                keys2 = set(score_level.keys()) | set(rank_level.keys())
                for k in keys2:
                    sdict[k] = beta * score_level.get(k, 0.0) + (1.0 - beta) * rank_level.get(k, 0.0)

            else:
                raise ValueError(f"Unknown retriever type: {self.args.retriever}")

            if not sdict:
                continue

            score_dicts.append(sdict)
            rl = ranking_from_scores(sdict)[:200]
            rank_lists.append(rl)

            seed_candidate_ids.update(rl[:seed_n])

            # if hybrid/hybrid_online, also include top from raw dd/sd for diversity
            if self.args.retriever in ("hybrid", "hybrid_online"):
                if dd:
                    seed_candidate_ids.update(ranking_from_scores(dd)[:seed_n])
                if sd:
                    seed_candidate_ids.update(ranking_from_scores(sd)[:seed_n])

        if not rank_lists:
            return [], q_variants

        # Multi-query fusion = score+rank mix
        fused_scores = fuse_multiquery_rank_and_score(
            score_dicts,
            rank_lists,
            rrf_k=self.args.rrf_k,
            beta=float(self.args.mq_fuse_beta)
        )

        # Avoid huge dict copy using ChainMap
        # merged_map.get(cid) will check extra_web_map first then self.chunk_map, no copying.
        merged_map = ChainMap(extra_web_map, self.chunk_map)

        # Smarter candidate_k (auto grows with top_k)
        # candidate_k now is at least max(args.candidate_k, top_k*20, 50)
        cand_k = min(
            len(seed_candidate_ids) if seed_candidate_ids else len(fused_scores),
            max(50, self.args.candidate_k, self.args.top_k * 20)
        )

        # Build sorted candidate ids by fused score (so reranker sees best pool)
        if seed_candidate_ids:
            cand_sorted = sorted(seed_candidate_ids, key=lambda x: fused_scores.get(str(x), 0.0), reverse=True)
            cand_sorted = cand_sorted[:cand_k]
        else:
            # fallback
            cand_sorted = [cid for cid, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:cand_k]]

        # Rerank or not; unify output schema
        if self.args.rerank == "bge":
            # use truncate for candidates to limit memory
            doc_candidates = candidates_from_ids(
                cand_sorted,
                fused_scores,
                merged_map,
                truncate=self.args.truncate
            )

            reranked = bge_cross_encoder_rerank(
                query,
                doc_candidates,
                top_k=self.args.top_k,
                model_name=self.args.rerank_model,
                batch_size=self.args.rerank_batch_size,
                max_chars=self.args.rerank_max_chars
            )

            results = []
            for rank, d in enumerate(reranked, 1):
                results.append({
                    "rank": rank,
                    "chunk_id": d["chunk_id"],
                    "score": float(d.get("ce_score", 0.0)),
                    "pre_score": float(d.get("pre_score", 0.0)),
                    "source": d.get("source", ""),
                    "text": d.get("text", "")
                })
            return results, q_variants

        else:
            # no rerank: just materialize top_k from fused_scores
            results = materialize_topk(fused_scores, merged_map, top_k=self.args.top_k, truncate=self.args.truncate)
            return results, q_variants

    def run_batch(self):
        questions = load_questions(self.paths["questions"])
        print(f"Loaded {len(questions)} questions from {self.paths['questions']}")

        out_txt = os.path.join(
            self.args.out_dir,
            f"retrieval_info_{self.args.retriever}_{self.args.rewrite}_{self.args.chunk}.txt"
        )
        out_json = os.path.join(self.args.out_dir, f"retrieval_{self.args.chunk}.jsonl")

        with open(out_txt, "w", encoding="utf-8") as f_txt, open(out_json, "w", encoding="utf-8") as f_json:
            for qi, q in enumerate(tqdm(questions, desc="Retrieving"), 1):
                results, q_variants = self.retrieve_single(q, qid=qi)

                # TXT
                f_txt.write("=" * 80 + "\n")
                f_txt.write(f"[Q{qi}] {q}\n")
                if len(q_variants) > 1:
                    f_txt.write("Rewrites:\n" + "\n".join(q_variants[1:]) + "\n\n")

                for r in results:
                    line = f"  Rank {r['rank']}: score={r['score']:.4f}"
                    if "pre_score" in r:
                        line += f" | pre_score={r['pre_score']:.4f}"
                    f_txt.write(line + "\n")
                    f_txt.write(f"  chunk_id={r['chunk_id']} | source={r.get('source','')}\n")
                    f_txt.write(f"  text: {r.get('text','')}\n\n")

                # JSONL
                entry = {
                    "qid": qi,
                    "question": q,
                    "retrieved": results
                }
                f_json.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Written results to {out_txt}")
        print(f"Saved JSON results to {out_json}")


# ------------------ Main ------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Dense/Sparse/Hybrid/Online retrieval + MQ rewriting + Score/Rank fusion + Optional Rerank"
    )

    parser.add_argument("--chunk", type=str, required=True, help="dataset name, e.g., littleItaly")
    parser.add_argument("--model", type=str, default="BAAI/bge-m3", help="embedding model, e.g., BAAI/bge-m3")
    parser.add_argument("--questions", type=str, default=None, help="questions file (one per line). If None, infer from chunk")

    parser.add_argument("--top_k", type=int, default=5, help="top-k to retrieve")
    parser.add_argument("--retriever", type=str, default="hybrid",
                        choices=["dense", "sparse", "hybrid", "online", "hybrid_online"],
                        help="retrieval mode")
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="hybrid weight base alpha (dynamic adjustment inside)")
    parser.add_argument("--index_type", type=str, default="flat", choices=["flat", "hnsw"],
                        help="FAISS index type for dense")
    parser.add_argument("--truncate", type=int, default=0, help="truncate retrieved text to N chars (0 = no truncation)")

    # Query rewriting
    parser.add_argument("--rewrite", type=str, default="none", choices=["none", "mq"], help="query rewriting method")
    parser.add_argument("--rewrite_n", type=int, default=4, help="number of rewritten queries for multi-query (excluding original)")
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF hyperparam k (larger -> smoother)")

    # (4) score/rank mix fusion weight
    parser.add_argument("--mq_fuse_beta", type=float, default=0.55,
                        help="beta in [0,1] for mixing score-fusion and rank(RRF)-fusion across variants. "
                             "final = beta*score + (1-beta)*rrf")

    # Output
    parser.add_argument("--out_dir", type=str, default="result_query_expansion_cross_encoding",
                        help="where to save retrieval results")

    # Rerank
    parser.add_argument("--rerank", type=str, default="bge", choices=["none", "bge"], help="cross-encoder reranking")
    parser.add_argument("--rerank_model", type=str, default="BAAI/bge-reranker-large", help="cross-encoder model id")
    parser.add_argument("--candidate_k", type=int, default=100, help="base num of candidates before rerank (auto grows with top_k)")
    parser.add_argument("--rerank_batch_size", type=int, default=64, help="batch size for reranker scoring")
    parser.add_argument("--rerank_max_chars", type=int, default=1200, help="truncate doc text for rerank scoring")

    # Online
    parser.add_argument("--tavily_key", type=str, default=os.getenv("TAVILY_API_KEY", ""),
                        help="Tavily API key; default from env TAVILY_API_KEY")
    parser.add_argument("--hybrid_online_w", type=float, default=0.35,
                        help="online weight w in [0,1] for hybrid_online score-level fusion: (1-w)*local + w*online")
    parser.add_argument("--online_dedup_gamma", type=float, default=0.7,
                        help="domain duplicate penalty gamma in (0,1]; repeated domain score *= gamma^(count-1)")

    return parser


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pipeline = RAGPipeline(args)
    pipeline.run_batch()


if __name__ == "__main__":

    main()
