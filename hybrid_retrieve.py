import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dense_retrieve import (
    load_chunks, load_questions, build_faiss_index, embed_queries, dense_search
)
from sparse_retrieve import (
    tokenize, build_bm25_index, search_bm25 as sparse_search
)

# ---------- Configuration ----------
CHUNKS_PATH = "data/chunks_littleItaly.jsonl"
EMB_PATH = "index/embeddings_littleItaly.npy"
IDS_PATH = "index/ids_littleItaly.npy"
QUESTIONS_PATH = "data/test/question_test.txt"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
ALPHA = 0.6  # weight coefficient（for weighted average）
# --------------------------


# ===============================================================
#   Weighted Average Fusion
# ===============================================================

def normalize_scores(result_list):
    """Normalize the scores for weighted average"""
    scores = np.array([r["score"] for r in result_list])
    if len(scores) == 0:
        return result_list
    min_s, max_s = scores.min(), scores.max()
    if max_s == min_s:  # avoid division by 0
        norm_scores = np.ones_like(scores)
    else:
        norm_scores = (scores - min_s) / (max_s - min_s)
    for r, s in zip(result_list, norm_scores):
        r["score"] = float(s)
    return result_list



def weighted_average_fusion(dense_res, sparse_res, chunk_map, alpha=0.6, top_k=5):
    """
    conduct weighted average between dense and sparse results
    return the same results as the other two methods
    """
    results = []
    for q_idx in tqdm(range(len(dense_res)), desc="Weighted Fusion"):
        dense_dict = {r["chunk_id"]: r["score"] for r in normalize_scores(dense_res[q_idx])}
        sparse_dict = {r["chunk_id"]: r["score"] for r in normalize_scores(sparse_res[q_idx])}

        # record all ids that have occured
        all_ids = set(dense_dict.keys()) | set(sparse_dict.keys())

        # merge the score
        combined_scores = {}
        for cid in all_ids:
            d_score = dense_dict.get(cid, 0.0)
            s_score = sparse_dict.get(cid, 0.0)
            combined_scores[cid] = alpha * d_score + (1 - alpha) * s_score

        # re rank based on the weighted scores and return the results
        sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        one_query_results = []
        for rank, (cid, score) in enumerate(sorted_ids, 1):
            d = chunk_map[cid]
            one_query_results.append({
                "rank": rank,
                "chunk_id": cid,
                "score": float(score),
                "source": d.get("source", ""),
                "text": d["text"]
            })
        results.append(one_query_results)
    return results


# ===============================================================
#   Reciprocal Rank Fusion (RRF)
# ===============================================================
def reciprocal_rank_fusion(dense_res, sparse_res, chunk_map, k=60, top_k=5):
    """
    Implement Reciprocal Rank Fusion (RRF)
    RRF_score = Σ (1 / (k + rank_i))
    """
    results = []
    for q_idx in tqdm(range(len(dense_res)), desc="RRF Fusion"):
        rr_scores = {}

        # dense results 
        for r in dense_res[q_idx]:
            cid = r["chunk_id"]
            rr_scores[cid] = rr_scores.get(cid, 0.0) + 1.0 / (k + r["rank"])

        # sparse results
        for r in sparse_res[q_idx]:
            cid = r["chunk_id"]
            rr_scores[cid] = rr_scores.get(cid, 0.0) + 1.0 / (k + r["rank"])

        # re rank based on the rrf scores
        sorted_ids = sorted(rr_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        one_query_results = []
        for rank, (cid, score) in enumerate(sorted_ids, 1):
            d = chunk_map[cid]
            one_query_results.append({
                "rank": rank,
                "chunk_id": cid,
                "score": float(score),
                "source": d.get("source", ""),
                "text": d["text"]
            })
        results.append(one_query_results)
    return results


# ===============================================================
#   Example pipeline for testing
# ===============================================================
def main():
    # 1. Load resources
    chunk_map = load_chunks(CHUNKS_PATH)
    questions = load_questions(QUESTIONS_PATH)

    # 2. Dense retrieve
    ids = np.load(IDS_PATH, allow_pickle=True)
    index = build_faiss_index(EMB_PATH)
    model = SentenceTransformer(MODEL_NAME)
    query_embs = embed_queries(model, questions)
    dense_results = dense_search(index, query_embs, ids, chunk_map, TOP_K)

    # 3. Sparse retrieve
    bm25, id_list = build_bm25_index(chunk_map)
    sparse_results = sparse_search(bm25, chunk_map, id_list, questions, TOP_K)

    # 4. merge
    fused_weighted = weighted_average_fusion(dense_results, sparse_results, chunk_map, alpha=ALPHA, top_k=TOP_K)
    fused_rrf = reciprocal_rank_fusion(dense_results, sparse_results, chunk_map, k=60, top_k=TOP_K)

    # 5. output
    for mode, fused in [("weighted", fused_weighted), ("rrf", fused_rrf)]:
        output_path = f"retrieval_info_hybrid_{mode}.txt"
        open(output_path, "w").close()
        with open(output_path, "a", encoding="utf-8") as f:
            for qi, (q, res) in enumerate(zip(questions, fused), 1):
                f.write("=" * 80 + "\n")
                f.write(f"[Q{qi}] {q}\n\n")
                for r in res:
                    f.write(f"  Rank {r['rank']}: score={r['score']:.4f}\n")
                    f.write(f"  chunk_id={r['chunk_id']} | source={r['source']}\n")
                    f.write(f"  text: {r['text']}\n\n")
        print(f"Saved hybrid retrieval results to {output_path}")


if __name__ == "__main__":
    main()
