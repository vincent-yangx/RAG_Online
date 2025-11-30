import json
import numpy as np
import os
from tqdm import tqdm
import argparse
from FlagEmbedding import BGEM3FlagModel
import faiss


# ================= Parse Arguments ========================
parser = argparse.ArgumentParser(description="Embed WikiQA corpus into dense vectors & build FAISS index")

parser.add_argument(
    "--corpus",
    type=str,
    default="./wikiqa_test/wikiqa_corpus.json",
    help="Path to wikiqa_corpus.json (e.g., ./wikiqa_test/wikiqa_corpus.json)"
)

parser.add_argument(
    "--model",
    type=str,
    default="BAAI/bge-m3",
    help="Embedding model to use (default: BAAI/bge-m3)"
)

parser.add_argument(
    "--out_dir",
    type=str,
    default="./wikiqa_test_index",
    help="Directory to save embeddings & index"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Batch size for embedding"
)

args = parser.parse_args()

CORPUS_PATH = args.corpus
MODEL_NAME = args.model
OUT_DIR = args.out_dir
BATCH_SIZE = args.batch_size

model_name_simple = MODEL_NAME.split("/")[-1]

# Output files
OUT_EMB = f"wikiqa_dense_emb_{model_name_simple}.npy"
OUT_IDS = f"wikiqa_doc_ids_{model_name_simple}.npy"
OUT_FAISS = f"wikiqa_faiss_index_{model_name_simple}.faiss"


# ================= Load WikiQA corpus ======================
def load_corpus(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    return corpus


# ================= Build Embeddings ========================
def build_wikiqa_embeddings():

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load corpus
    corpus = load_corpus(CORPUS_PATH)
    doc_ids = list(corpus.keys())
    texts = [corpus[doc_id]["text"] for doc_id in doc_ids]

    print(f"[INFO] Loaded {len(texts)} documents from WikiQA corpus.")

    # Load model
    print(f"[INFO] Loading embedding model: {MODEL_NAME}")
    model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)

    # Start embedding
    all_embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding"):
        batch_texts = texts[i:i + BATCH_SIZE]
        emb = model.encode(batch_texts, batch_size=64)["dense_vecs"].astype("float32")

        # Normalize embedding
        faiss.normalize_L2(emb)
        all_embs.append(emb)

    embs = np.vstack(all_embs).astype("float32")
    doc_ids = np.array(doc_ids)

    print(f"[INFO] Embedding completed. Shape = {embs.shape}")

    # Save embeddings + ids
    emb_path = os.path.join(OUT_DIR, OUT_EMB)
    ids_path = os.path.join(OUT_DIR, OUT_IDS)

    np.save(emb_path, embs)
    np.save(ids_path, doc_ids)

    print(f"[INFO] Saved embeddings to: {emb_path}")
    print(f"[INFO] Saved doc ids to: {ids_path}")

    # Build FAISS index
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss_path = os.path.join(OUT_DIR, OUT_FAISS)
    faiss.write_index(index, faiss_path)

    print(f"[INFO] FAISS index saved to: {faiss_path}")

    # Verification
    print("=" * 70)
    print("[VERIFY]")

    saved_embs = np.load(emb_path)
    saved_ids = np.load(ids_path)

    print(f"embedding shape = {saved_embs.shape}")
    print(f"ids shape = {saved_ids.shape}")

    print("Embeddings consistency:", np.array_equal(embs, saved_embs))
    print("IDs consistency:", np.array_equal(doc_ids, saved_ids))

    print("[DONE]")


# ================= Execute ==============================
if __name__ == "__main__":
    build_wikiqa_embeddings()
