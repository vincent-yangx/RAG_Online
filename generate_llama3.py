import json
from huggingface_hub import InferenceClient
from tqdm import tqdm
import os
import argparse

# ------------------ Argparse ------------------
parser = argparse.ArgumentParser(description="Generate answers using LLaMA 3 8B Instruct")
parser.add_argument("--chunk", type=str, required=True, help="dataset name, e.g., about_cmu_refined_hybrid")
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                    help="generator model name (default: LLaMA3 8B Instruct)")
parser.add_argument("--top_k", type=int, default=5, help="number of passages to include in context")
parser.add_argument("--temperature", type=float, default=0.3, help="generation temperature")
parser.add_argument("--max_tokens", type=int, default=300, help="maximum tokens for generation")
args = parser.parse_args()

# ------------------ Paths ------------------
INPUT_PATH = f"result_query_expansion_cross_encoding/retrieval_{args.chunk}.jsonl"
OUT_DIR = "Llama_output"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_JSON = os.path.join(OUT_DIR, f"{args.chunk}.jsonl")
OUTPUT_TXT = os.path.join(OUT_DIR, f"{args.chunk}.txt")

# ------------------ Model ------------------
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
client = InferenceClient(model=args.model, token=hf_token)

def build_prompt(question, passages, top_k=5):
    """Construct the retrieval-augmented prompt."""
    context = "\n\n".join(
        [f"[Passage {i+1}]\n{p['text']}" for i, p in enumerate(passages[:top_k])]
    )
    return f"""
You are a helpful assistant. Use the given passages to answer the user's question.
Make sure that the answer is precise and conceise. Don't include repeated information.
If the answer cannot be found in the passages, say "I don't know".

Question: {question}

Passages:
{context}

Answer:
""".strip()

# ------------------ Main ------------------
with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
     open(OUTPUT_JSON, "w", encoding="utf-8") as fout_json, \
     open(OUTPUT_TXT, "w", encoding="utf-8") as fout_txt:

    for line in tqdm(fin, desc="Generating answers"):
        data = json.loads(line)
        qid = data.get("qid")
        q = data["question"]
        retrieved = data["retrieved"]

        prompt = build_prompt(q, retrieved, top_k=args.top_k)

        try:
            resp = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            answer = resp.choices[0].message["content"].strip()
        except Exception as e:
            answer = f"[ERROR] {e}"

        # --- 写入 JSON ---
        output_entry = {
            "qid": qid,
            "question": q,
            "answer_llama3_8b": answer
        }
        fout_json.write(json.dumps(output_entry, ensure_ascii=False) + "\n")

        # --- 写入简洁 TXT ---
        q_label = f"[Q{qid}]" if qid is not None else "[Q]"
        fout_txt.write(f"{q_label} {q}\n")
        fout_txt.write(f"Answer: {answer}\n\n")

print("✅ Answers saved to:")
print(f"   - JSONL: {OUTPUT_JSON}")
print(f"   - TXT:   {OUTPUT_TXT}")