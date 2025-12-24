# Actually even worse

import json
from huggingface_hub import InferenceClient
from tqdm import tqdm
import os
import argparse
import re

# ------------------ Argparse ------------------
parser = argparse.ArgumentParser(description="Generate short, normalized answers using LLaMA 3 8B Instruct (extractive JSON mode)")
parser.add_argument("--chunk", type=str, required=True, help="dataset name, e.g., about_cmu_refined_hybrid")
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                    help="generator model name (default: LLaMA3 8B Instruct)")
parser.add_argument("--top_k", type=int, default=5, help="number of passages to include in context")
parser.add_argument("--temperature", type=float, default=0.0, help="generation temperature (suggest 0.0 for extraction)")
parser.add_argument("--max_tokens", type=int, default=300, help="maximum tokens for generator pass")
# Validator options
parser.add_argument("--validate", action="store_true",
                    help="enable a second-pass JSON validator to auto-fix format/evidence issues")
parser.add_argument("--validator_max_tokens", type=int, default=160,
                    help="max tokens for validator pass")
args = parser.parse_args()

# ------------------ Paths ------------------
INPUT_PATH = f"result_query_expansion_cross_encoding/retrieval_{args.chunk}.jsonl"
OUT_DIR = "Llama_output_new_prompt"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_JSON = os.path.join(OUT_DIR, f"{args.chunk}_new_prompt.jsonl")
OUTPUT_TXT = os.path.join(OUT_DIR, f"{args.chunk}_new_prompt.txt")

# ------------------ Model ------------------
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
client = InferenceClient(model=args.model, token=hf_token)

# ------------------ Prompts ------------------
SYSTEM_PROMPT = """
You are a precise fact extractor. Use ONLY the provided passages.

TASK
- Decide the answer_type from:
  ["date","date_range","money","integer_count","time_24h","yes_no","location_name","price_range","free_text_short"]
- Return a SINGLE JSON object wrapped in <final>...</final> with fields:
  {
    "qid": <int>,
    "answer_type": <string>,
    "normalized_answer": <string>,
    "verbatim_evidence": <short exact quote 3-30 words>,
    "passage_id": <int>,
    "confidence": <float 0.0-1.0>,
    "note": <optional short reason or 'conflict'>
  }

NORMALIZATION
- date: YYYY-MM-DD
- date_range: YYYY-MM-DD/YYYY-MM-DD
- money: $X or $X.XX (no commas)
- integer_count: digits only
- time_24h: HH:MM
- yes_no: "yes" or "no"
- location_name: concise proper name
- price_range: $A-$B
- free_text_short: <= 15 words

CONSTRAINTS
- Output ONLY the <final>{JSON}</final>. No extra text.
- If there is no explicit evidence, or passages conflict, set normalized_answer to "I don't know" and note="conflict" if applicable.
- Do NOT use outside knowledge.
"""

VALIDATOR_SYSTEM = """
You are a strict JSON answer validator and fixer.
- If the candidate JSON's verbatim_evidence does not literally appear in the cited passage,
  or normalized_answer violates normalization, or conflicts exist, REVISE the JSON.
- Otherwise return it unchanged.
- Output ONLY the revised JSON wrapped in <final>...</final>. No extra text.
"""

# ------------------ Prompt Builders ------------------
def _format_passages(passages, top_k=5):
    lines = []
    for i, p in enumerate(passages[:top_k], 1):
        src = p.get("source") or p.get("url") or p.get("doc_id") or ""
        src_str = f" ({src})" if src else ""
        text = p.get("text", "")
        lines.append(f"[P{i}]{src_str}\n{text}")
    return "\n\n".join(lines)

def build_user_prompt(qid, question, passages, top_k=5):
    context = _format_passages(passages, top_k=top_k)
    return f"""
qID: {qid}
Question: {question}

Passages:
{context}

OUTPUT
<final>{{...}}</final>
""".strip()

def build_validator_prompt(question, passages, candidate_obj, top_k=5):
    context = _format_passages(passages, top_k=top_k)
    cand = json.dumps(candidate_obj, ensure_ascii=False)
    return f"""
Question: {question}

Passages:
{context}

Candidate:
{cand}

OUTPUT
<final>{{...}}</final>
""".strip()

# ------------------ Parsing & Post-processing ------------------
def extract_final_json(text):
    raw = text.strip()
    m = re.search(r"<final>\s*(\{.*?\})\s*</final>", raw, flags=re.DOTALL)
    if not m:
        return None, raw
    try:
        obj = json.loads(m.group(1))
        return obj, raw
    except json.JSONDecodeError:
        return None, raw

def clamp_normalized(obj):
    """Light normalization without changing semantics drastically."""
    if not isinstance(obj, dict):
        return obj
    t = (obj.get("answer_type") or "").strip()
    s = (obj.get("normalized_answer") or "").strip()
    if t == "integer_count":
        m = re.search(r"\d+", s)
        obj["normalized_answer"] = m.group(0) if m else "I don't know"
    elif t == "yes_no":
        s2 = s.lower().strip()
        obj["normalized_answer"] = "yes" if s2.startswith("y") else ("no" if s2.startswith("n") else "I don't know")
    elif t == "money":
        s2 = s.replace(",", "").strip()
        if not s2.startswith("$"):
            m = re.search(r"\d+(?:\.\d{1,2})?", s2)
            s2 = f"${m.group(0)}" if m else "I don't know"
        obj["normalized_answer"] = s2
    elif t == "price_range":
        s2 = s.replace(" ", "")
        obj["normalized_answer"] = s2
    # Other types: keep model output
    return obj

def get_normalized_answer(obj):
    if isinstance(obj, dict):
        ans = obj.get("normalized_answer")
        if isinstance(ans, str) and ans.strip():
            return ans.strip()
    return "I don't know"

# ------------------ Main ------------------
with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
     open(OUTPUT_JSON, "w", encoding="utf-8") as fout_json, \
     open(OUTPUT_TXT, "w", encoding="utf-8") as fout_txt:

    for line in tqdm(fin, desc="Generating answers"):
        data = json.loads(line)
        qid = data.get("qid")
        q = data["question"]
        retrieved = data["retrieved"]

        user_prompt = build_user_prompt(qid, q, retrieved, top_k=args.top_k)

        try:
            resp = client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            raw_out = resp.choices[0].message["content"].strip()
        except Exception as e:
            raw_out = f"<final>{{\"qid\": {qid}, \"answer_type\":\"free_text_short\", \"normalized_answer\":\"I don't know\", \"verbatim_evidence\":\"\", \"passage_id\": 0, \"confidence\": 0.0, \"note\":\"exception: {str(e)}\"}}</final>"

        obj, raw_text = extract_final_json(raw_out)

        # Optional: second-pass validator
        if args.validate:
            candidate = obj if obj is not None else {
                "qid": qid, "answer_type":"free_text_short", "normalized_answer":"I don't know",
                "verbatim_evidence":"", "passage_id":0, "confidence":0.0, "note":"no-json"
            }
            v_prompt = build_validator_prompt(q, retrieved, candidate, top_k=args.top_k)
            try:
                v_resp = client.chat_completion(
                    messages=[
                        {"role": "system", "content": VALIDATOR_SYSTEM},
                        {"role": "user", "content": v_prompt},
                    ],
                    max_tokens=args.validator_max_tokens,
                    temperature=0.0
                )
                v_out = v_resp.choices[0].message["content"].strip()
                v_obj, v_raw = extract_final_json(v_out)
                if v_obj:
                    obj, raw_text = v_obj, v_raw
            except Exception:
                pass

        # Light post-process
        if obj:
            obj = clamp_normalized(obj)
        normalized = get_normalized_answer(obj)

        # --- write JSON ---
        output_entry = {
            "qid": qid,
            "question": q,
            "answer_llama3_8b": normalized,          
            "answer_json": obj if isinstance(obj, dict) else None,
            "raw": raw_text
        }
        fout_json.write(json.dumps(output_entry, ensure_ascii=False) + "\n")

        # --- write into txt file ---
        q_label = f"[Q{qid}]" if qid is not None else "[Q]"
        fout_txt.write(f"{q_label} {q}\n")
        fout_txt.write(f"Answer: {normalized}\n\n")

print("✅ Answers saved to:")
print(f"   - JSONL: {OUTPUT_JSON}")
print(f"   - TXT:   {OUTPUT_TXT}")
