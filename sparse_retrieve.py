import json
import numpy as np
import re
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from dense_retrieve import load_chunks,load_questions
# ---------- Configuration ----------
CHUNKS_PATH = "data/chunks_littleItaly.jsonl"
QUESTIONS_PATH = "data/test/question_test.txt"
TOP_K = 5  # How many chunks to retrieve per query
# --------------------------

def tokenize(text):
    """Simple tokenizer: lowercase + keep alphanumerics"""
    return re.findall(r"[a-z0-9]+", text.lower())

def build_bm25_index(chunk_map):
    """Build BM25 index using chunk_map values"""
    docs = []
    id_list = []
    for cid, c in chunk_map.items():
        docs.append(tokenize(c["text"]))
        id_list.append(cid)
    bm25 = BM25Okapi(docs)
    print("BM25 index built successfully.")
    return bm25, id_list

def search_bm25(bm25, chunk_map, id_list, questions, top_k=5):
    """Perform BM25 sparse retrieval"""
    results = []
    for q in tqdm(questions, desc="BM25 Searching"):
        query_tokens = tokenize(q)
        scores = bm25.get_scores(query_tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]  # descending order

        one_query_results = []
        for rank, i in enumerate(top_idx, 1):
            cid = id_list[int(i)]
            d = chunk_map[cid]
            one_query_results.append({
                "rank": rank,
                "chunk_id": str(d["chunk_id"]),
                "score": float(scores[i]),
                "source": d.get("source", ""),
                "text": d["text"]
            })
        results.append(one_query_results)
    return results

def main():
    # 1. Load chunk_map and build BM25 index
    chunk_map = load_chunks(CHUNKS_PATH)
    bm25, id_list = build_bm25_index(chunk_map)

    # 2. Load questions
    questions = load_questions(QUESTIONS_PATH)

    # 3. Search
    results = search_bm25(bm25, chunk_map, id_list, questions, top_k=TOP_K)

    # 4. Save results
    output_path = "retrieval_info_sparse.txt"
    open(output_path, "w").close()
    with open(output_path, "a", encoding="utf-8") as f:
        for qi, (q, res) in enumerate(zip(questions, results), 1):
            f.write("=" * 80 + "\n")
            f.write(f"[Q{qi}] {q}\n\n")
            for r in res:
                f.write(f"  Rank {r['rank']}: score={r['score']:.4f}\n")
                f.write(f"  chunk_id={r['chunk_id']} | source={r['source']}\n")
                f.write(f"  text: {r['text']}\n\n")

    print(f"Sparse retrieval results saved to {output_path}")

if __name__ == "__main__":
    main()
