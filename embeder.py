import json
import numpy as np
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import argparse
from FlagEmbedding import BGEM3FlagModel
import faiss 

# from generate import EMB_PATH

# =============== Get Args =========================
parser = argparse.ArgumentParser(description="Please enter the retrieve mode to use and dataset to test")
parser.add_argument("--chunk", type=str, required=True,help="Pleaes select which chunk data to embed")
parser.add_argument("--model",default="sentence-transformers",type=str,help="Please enter the embedding model to use")
args = parser.parse_args()

if args.model == "sentence-transformers":
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"
elif args.model == "BAAI":
    MODEL = "BAAI/bge-m3"
else:
    raise ValueError("Invalid Embedding Model!")

print("*"*50)
print(f"The embedding model used is {MODEL}")
print("*"*50)

# ============== Configurations =====================
CHUNK_PATH = f"data/chunks/chunks_{args.chunk}.jsonl"
# MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# MODEL = "BAAI/bge-m3"
OUT_DIR = "index"
OUT_EMB = f"embeddings_{args.chunk}_{args.model}.npy"
OUT_IDX = f"ids_{args.chunk}_{args.model}.npy"


def load_chunks(jsonl_path):
    """read chunks.jsonl line by line"""
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            chunks.append(data)
    return chunks

def build_embeddings(
        # chunks_path="data/chunks_normal.jsonl",
        chunks_path=CHUNK_PATH,
        model_name=MODEL,
        out_dir=OUT_DIR,
        batch_size=128,
        normalize=True):
    
    """Embedding the chunk text using sentence transformer"""
    os.makedirs(out_dir, exist_ok=True)

    # Load the model
    print(f"Loading embedding model: {model_name}")

    if MODEL == "sentence-transformers/all-MiniLM-L6-v2":
        model = SentenceTransformer(model_name)
    elif MODEL == "BAAI/bge-m3":
        model = BGEM3FlagModel(MODEL, use_fp16=True)


    # Read the chunks
    chunks = load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]
    ids   = [c["chunk_id"] for c in chunks]

    print(f"Total chunks to encode: {len(texts)}")

    # Encode in Batch
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]

        if MODEL == "sentence-transformers/all-MiniLM-L6-v2":
            emb = model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize  # Normalize for cosine similarity
            )
        elif MODEL == "BAAI/bge-m3":
            emb = model.encode(batch_texts, batch_size=64)["dense_vecs"].astype("float32")  # [N,1024]    
        
        faiss.normalize_L2(emb) # Normalize the embeddings
        all_embs.append(emb)

    embs = np.vstack(all_embs).astype("float32")
    ids  = np.array(ids)

    # Save the results
    np.save(os.path.join(out_dir, OUT_EMB), embs)
    np.save(os.path.join(out_dir, OUT_IDX), ids)
    print(f"Saved {embs.shape[0]} embeddings to {out_dir}/")

    embeddings = np.load(f"index/embeddings_{args.chunk}_{args.model}.npy")
    index = np.load(f"index/ids_{args.chunk}_{args.model}.npy")


    print("="*80)
    print("VERIFY:")
    print(f"embedding has shape: {embeddings.shape}")
    print(f"index has shape: {index.shape}")    

if __name__ == "__main__":
    build_embeddings(
        chunks_path=CHUNK_PATH,          # File Path
        model_name=MODEL,
        out_dir=OUT_DIR,                     # Output Directory
        batch_size=128,
        normalize=True
    )
