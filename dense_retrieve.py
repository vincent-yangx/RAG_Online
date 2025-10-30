import faiss
import numpy as np
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel


# ---------- Configuration ----------
CHUNKS_PATH = "data/chunks_littleItaly.jsonl"
EMB_PATH = "index/embeddings_littleItaly.npy"
IDS_PATH = "index/ids_littleItaly.npy"
MODEL = "BAAI/bge-m3"
# QUESTIONS_PATH = "data/question.txt"
QUESTIONS_PATH = "data/test/question_littleItaly.txt"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5  # How many answers will be retrieved
# --------------------------

def load_chunks(path):
    """Load chunk data, return a dictionary: chunk_id -> text"""
    chunk_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunk_map[str(d["chunk_id"])] = d
    return chunk_map

def build_faiss_index(emb_path):
    """load embedding and build FAISS indexing"""
    print("Loading embeddings...")
    embs = np.load(emb_path)
    print(f"Embeddings shape: {embs.shape}")
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product indexing

    # vector has been normalized, inner product is equal to cosine similarity 
    index.add(embs)
    print("FAISS index built. Total vectors:", index.ntotal)
    return index

def embed_queries(model, questions):
    """Turn the question into embedding"""
    # sentence transformers
    if isinstance(model, SentenceTransformer):
        q_embs = model.encode(questions,convert_to_numpy=True,normalize_embeddings=True).astype("float32")
    # BAAI
    elif isinstance(model, BGEM3FlagModel):
        q_embs = model.encode(questions, batch_size=64)["dense_vecs"].astype("float32")  # [N,1024]
    faiss.normalize_L2(q_embs)
    return q_embs

def dense_search(index, query_embs, ids, chunk_map, top_k=5):
    """Search similar text using FAISS"""
    D, I = index.search(query_embs, top_k)
    results = []
    for q_idx, (scores, idxs) in enumerate(zip(D, I)):
        one_query_results = []
        for score, i in zip(scores, idxs):
            chunk_id = ids[i]
            chunk = chunk_map[str(chunk_id)]
            one_query_results.append({
                "rank": len(one_query_results)+1,
                "chunk_id": str(chunk_id),
                "score": float(score),
                "source": chunk.get("source", ""),
                # "text": chunk["text"][:300] + "..."  # truncation
                "text": chunk["text"]
            })
        results.append(one_query_results)
    return results

def load_questions(path="data/test/question.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    # 1. load the chunk mapping
    chunk_map = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunk_map)} chunks.")

    # 2. load ID and vectors
    ids = np.load(IDS_PATH, allow_pickle=True)

    # 3. construct FAISS indexing
    index = build_faiss_index(EMB_PATH)

    # 4. load the query from question
    # with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    #     questions = [line.strip() for line in f if line.strip()]
    questions = load_questions(QUESTIONS_PATH)
    # 5. Embed the query
    # model = SentenceTransformer(MODEL_NAME)
    model = BGEM3FlagModel(MODEL, use_fp16=True)
    query_embs = embed_queries(model, questions)

    # 6. Index
    results = dense_search(index, query_embs, ids, chunk_map, TOP_K)

    # 7. output
    output_path = "retrieval_info.txt"
    open(output_path,"w").close()
    with open(output_path, "a", encoding="utf-8") as f:
    # Record the retrived information

        for qi, (q, res) in enumerate(zip(questions, results), 1):
            # print("="*80)
            # print(f"[Q{qi}] {q}")
            f.write("=" * 80 + "\n")
            f.write(f"[Q{qi}] {q}\n\n")

            for r in res:
                # print(f"  Rank {r['rank']}: score={r['score']:.4f}")
                # print(f"  chunk_id={r['chunk_id']} | source={r['source']}")
                # print(f"  text: {r['text']}\n")
                f.write(f"  Rank {r['rank']}: score={r['score']:.4f}\n")
                f.write(f"  chunk_id={r['chunk_id']} | source={r['source']}\n")
                f.write(f"  text: {r['text']}\n\n") 

if __name__ == "__main__":
    main()
