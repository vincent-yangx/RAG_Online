from ctypes.wintypes import LANGID
import json
from typing import Required
from numpy import require
import torch
import transformers
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline 
from dense_retrieve import build_faiss_index, load_chunks, embed_queries, dense_search, load_questions  # import utility function as needed
from sparse_retrieve import build_bm25_index, search_bm25
from hybrid_retrieve import weighted_average_fusion, reciprocal_rank_fusion
import json
import argparse
from FlagEmbedding import BGEM3FlagModel


# ================= Get the argument =============================
parser = argparse.ArgumentParser(description="Please enter the retrieve mode to use and dataset to test")
parser.add_argument("--mode", type=str, required=True,help="Specify retrieve mode: spare, dense, weighted, rrf")
parser.add_argument("--dataset", type=str, required=True,help="Please enter the dataset to test")
parser.add_argument("--topk", type=int, required=True, help="Please enter Top K you want to use")
parser.add_argument("--embed", type=str, required=True,help="Specify the embedding model")
args = parser.parse_args()

print(f"We will be using {args.mode} for retrieving!\n")
print(f"We will be testing {args.dataset}!\n")
print(f"We will be using {args.embed} for embedding\n")



#  ================== Configuration ==================================
MODEL_ID= "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MAX_CONTEXT_CHARS = 10000                        # Control the length of context
TOP_K = args.topk                                      # Many top answers will be used
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_PATH = f"data/chunks/chunks_{args.dataset}.jsonl"
IDX_PATH = f"index/ids_{args.dataset}_{args.embed}.npy"
EMB_PATH = f"index/embeddings_{args.dataset}_{args.embed}.npy"
EMBED_MODEL_ID = "BAAI/bge-m3"
QUESTION_PATH = f"data/test/question_{args.dataset}.txt"
ALPHA = 0.6 # Weight coefficient for weighted averaging
REFERENCE_PATH = f"data/reference/reference_{args.dataset}.json"

if args.embed == "sentence-transformers":
    EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
elif args.embed == "BAAI":
    EMBED_MODEL_ID = "BAAI/bge-m3"
else:
    raise ValueError("Invalid Embedding Model!")


# ===================================================================

# Construct Template Prompt
# PROMPT_TEMPLATE = """
# Answer the question based only on the CONTEXT below. If the answer cannot be found in the context, say "N/A".
# Keep your answer short (within 30 words).
# QUESTION:{question}
# CONTEXT: {context}
# """

PROMPT_TEMPLATE = """
Your task:
1. Carefully read the question and the retrieved information below.
2. Determine whether the retrieved information contains relevant or correct answers.
3. If it does, use it to support your answer and cite it briefly.
4. If it does not, rely on your own knowledge to answer accurately.
5. Do not mix irrelevant facts from the retrieved text.
Question:
{question}
Retrieved Information:
{context}
Answer (clearly indicate if your answer is based on retrieval or your own knowledge):
Please show me your final answer, do not show your reasoning process. 
"""

# role_message = "You are a concise and factual assistant."
role_message = "You are an expert assistant with access to external retrieved documents."


def load_reference_answers(path):
    """Load reference answers from JSON"""
    with open(path, "r", encoding="utf-8") as f:
        refs = json.load(f)
    ref_map = {}
    for d in refs:
        ref_map.update(d)
    return ref_map


def build_llm_pipeline(model_id=MODEL_ID, device=DEVICE):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    return pipeline



def generate_answer(llm_pipe, question, retrieved_chunks):
    # Concatenate context
    ctxs, total_len = [], 0
    for item in retrieved_chunks:
        t = item["text"].strip()
        if total_len + len(t) > MAX_CONTEXT_CHARS:
            break
        ctxs.append(t)
        total_len += len(t)
    context = "\n\n".join(ctxs)

    prompt = PROMPT_TEMPLATE.format(question=question, context=context)

    message = [
        {"role": "system", "content": role_message },
        {"role":"user", "content":prompt}
    ]
    
    llm_pipe(message)

    outputs = llm_pipe(message, max_new_tokens=1000, do_sample=False) # Call the model to generate output
    content = outputs[0]["generated_text"][-1]['content']
    if "</think>" in content:  
        parts = content.split("\n</think>\n\n")
        think = parts[0]
        answer = parts[1]
    return think,answer


def main():
    # ===============  1. Load necessary components ====================
    chunk_map = load_chunks(CHUNK_PATH)
    ids = np.load(IDX_PATH, allow_pickle=True)
    questions = load_questions(QUESTION_PATH)
    llm = build_llm_pipeline(MODEL_ID, DEVICE)
    results = {}
    reference_answers = load_reference_answers(REFERENCE_PATH)


    
    # ===============  2. Initialzie retrieval models ====================
    if args.mode in ["dense", "weighted", "rrf"]:
        print("→ Loading dense embeddings...")
        index = build_faiss_index(EMB_PATH)
        ids = np.load(IDX_PATH, allow_pickle=True)
        
        # Instantiate the embedding model
        if args.embed == "sentence-transformers":
            embed_model = SentenceTransformer(EMBED_MODEL_ID) 
        elif args.embed == "BAAI":
            embed_model = BGEM3FlagModel(EMBED_MODEL_ID, use_fp16=True)
        else: 
            raise ValueError("Invalid Embedding Model!")

    if args.mode in ["sparse", "weighted", "rrf"]:
        print("→ Building BM25 index...")
        bm25, bm25_index = build_bm25_index(chunk_map)

    # ===============  3. retrieval loops ====================
    for qi, q in enumerate(questions, 1):
        print("="*80)
        print(f"[Q{qi}] {q}")


        # =======  dense ============
        if args.mode == "dense":
            query_embs = embed_queries(embed_model, [q])
            retrieved = dense_search(index, query_embs, ids, chunk_map, top_k=TOP_K)[0]

        # ======= sparse ============
        elif args.mode == "sparse":
            retrieved = search_bm25(bm25, chunk_map, bm25_index, [q],top_k=TOP_K)[0]

        # === weighted average fusion ===
        elif args.mode == "weighted":
            query_embs = embed_queries(embed_model, [q])
            dense_res = dense_search(index, query_embs, ids, chunk_map, top_k=TOP_K)
            sparse_res = search_bm25(bm25, chunk_map, bm25_index, [q],top_k=TOP_K)
            retrieved = weighted_average_fusion(dense_res, sparse_res, chunk_map, alpha=ALPHA, top_k=TOP_K)[0]

        # === reciprocal rank fusion ===
        elif args.mode == "rrf":
            query_embs = embed_queries(embed_model, [q])
            dense_res = dense_search(index, query_embs, ids, chunk_map, top_k=TOP_K)
            sparse_res = search_bm25(bm25, chunk_map, bm25_index, [q],top_k=TOP_K)
            retrieved = reciprocal_rank_fusion(dense_res, sparse_res, chunk_map, top_k=TOP_K)[0]

        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
        print(f"retrieved information is: {retrieved}")
  
        # get the answers

        think, answer = generate_answer(llm, q, retrieved)
        results[str(qi)] = answer


        print(f"→ LLM Reasoning: {think}\n")
        print(f"→ LLM Answer: {answer}\n")

        ref_ans = reference_answers.get(str(qi), "(No reference found)")
        print(f"→ Reference Answer: {ref_ans}\n")
        print("=" * 80)
        

    # 5. save the results
    output_file = f"system_outputs/system_output_{args.embed}_{args.mode}_{args.dataset}_{args.topk}_deepseek.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved{output_file}")

if __name__ == "__main__":
    import numpy as np
    main()
