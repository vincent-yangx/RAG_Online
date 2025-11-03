import json
import numpy as np
import os
from tqdm import tqdm
import argparse
from FlagEmbedding import BGEM3FlagModel
import faiss 

# =============== Get Args =========================
parser = argparse.ArgumentParser(description="Please enter the retrieve mode to use and dataset to test")
parser.add_argument("--chunk", type=str, required=True, help="Please select which chunk data to embed")
parser.add_argument("--model", type=str, default="BAAI/bge-m3", help="Model to use for embedding")
args = parser.parse_args()

MODEL = args.model


CHUNK_PATH = f"data/chunks/chunks_{args.chunk}.jsonl"
OUT_DIR = "index"
model_name_simple = MODEL.split('/')[-1]
OUT_EMB = f"embeddings_{args.chunk}_{model_name_simple}.npy"
OUT_IDX = f"ids_{args.chunk}_{model_name_simple}.npy"

def load_chunks(jsonl_path):
    """read chunks.jsonl line by line"""
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            chunks.append(data)
    return chunks

def build_embeddings(
        chunks_path=CHUNK_PATH,
        model_name=MODEL,
        out_dir=OUT_DIR,
        batch_size=128,
        normalize=True):
    
    os.makedirs(out_dir, exist_ok=True)

    # Load the model
    print(f"Loading embedding model: {model_name}")
    model = BGEM3FlagModel(model_name, use_fp16=True)

    # Read the chunks
    chunks = load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    print(f"Total chunks to encode: {len(texts)}")

    # Encode in Batch
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        emb = model.encode(batch_texts, batch_size=64)["dense_vecs"].astype("float32")
        
        if normalize:
            faiss.normalize_L2(emb)  # Normalize the embeddings
        all_embs.append(emb)

    embs = np.vstack(all_embs).astype("float32")
    ids = np.array(ids)

    # Save the results
    emb_path = os.path.join(out_dir, OUT_EMB)
    idx_path = os.path.join(out_dir, OUT_IDX)
    
    np.save(emb_path, embs)
    np.save(idx_path, ids)
    print(f"Saved {embs.shape[0]} embeddings to {emb_path}")
    print(f"Saved {len(ids)} ids to {idx_path}")

    index = faiss.IndexFlatIP(embs.shape[1])     
    index.add(embs)                           
    faiss_index_path = os.path.join(out_dir, f"faiss_index_{args.chunk}_{model_name_simple}.faiss")
    faiss.write_index(index, faiss_index_path)

    print("=" * 80)
    print("VERIFY:")
    
    saved_embeddings = np.load(emb_path)
    saved_ids = np.load(idx_path)
    
    print(f"embedding has shape: {saved_embeddings.shape}")
    print(f"index has shape: {saved_ids.shape}")
    
    print(f"Embeddings consistency check: {np.array_equal(embs, saved_embeddings)}")
    print(f"IDs consistency check: {np.array_equal(ids, saved_ids)}")

if __name__ == "__main__":
    build_embeddings()