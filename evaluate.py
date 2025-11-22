import json
import re
from collections import Counter
import argparse
import os
from tqdm import tqdm

from huggingface_hub import InferenceClient

HF_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
LLM_CORRECT_THRESHOLD = 0.7
_hf_client = None  


JUDGE_PROMPT_TEMPLATE = """
You are an impartial judge for a question answering task.

You will be given:
- a question (may be empty)
- a reference answer
- a model-generated answer

Your job:

1. First, identify the core factual content in the reference answer:
   - key entities (people, places, events, organizations),
   - key attributes (dates, time ranges, counts, locations, names),
   - and their relationships.

2. Then compare the model answer to the reference answer and decide whether it is:
   - correct,
   - partially correct,
   - or incorrect.

3. Focus on whether the model answer preserves the core facts and relationships
   from the reference answer. Wording may be different; that is fine.

4. Extra details in the model answer are allowed as long as they do NOT clearly
   contradict the reference answer or the question.
   - If the reference answer omits some details (for example, the year of a date),
     but the model answer fills them in while keeping the core information consistent
     (same date range, same location, same entities), you should still treat the
     answer as correct or at least partially correct, NOT incorrect.
   - Only treat extra information as incorrect if it clearly conflicts with the
     reference (e.g., different year, different place, different person) or with
     the question.

5. Use the following guidelines:
   - "correct": the model answer captures all of the important information in the
     reference answer, and any extra details do not conflict with it.
   - "partially_correct": the model answer gets the main idea roughly right but
     misses some important details OR contains some minor inaccuracies.
   - "incorrect": the model answer fails to capture the main facts, gives a
     fundamentally wrong answer, or clearly contradicts the reference.

Output a single line of valid JSON with the following fields:
- "label": one of ["correct", "partially_correct", "incorrect"]
- "score": a float between 0 and 1
    - 1.0 for fully correct
    - 0.5 for partially correct
    - 0.0 for incorrect
- "reason": a short English explanation

Do NOT include any extra text outside the JSON.

Now judge the following:

Question: {question}

Reference answer: {reference}

Model answer: {prediction}
""".strip()



def get_hf_client():
    global _hf_client
    if _hf_client is not None:
        return _hf_client

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HUGGINGFACEHUB_API_TOKEN is not set in environment. "
            "Please export it before running with --use_llm_judge."
        )

    print(f"[INFO] Initializing HF InferenceClient with model: {HF_MODEL_NAME}")
    _hf_client = InferenceClient(HF_MODEL_NAME, token=hf_token)
    return _hf_client


def _parse_json_from_text(text: str):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, flags=re.DOTALL)
        if not m:
            return {"label": "invalid", "score": 0.0, "reason": "no JSON found"}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {"label": "invalid", "score": 0.0, "reason": "JSON parse error"}


def llm_judge_one(question: str, gold: str, pred: str):
    """
    Use HF Llama-3-8B-Instruct 
    return: {label, score, reason}
    """
    client = get_hf_client()

    question = question or ""
    gold = gold or ""
    pred = pred or ""

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        reference=gold,
        prediction=pred,
    )

    resp = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
    )

    content = resp.choices[0].message.content
    data = _parse_json_from_text(content)

    label = data.get("label", "invalid")
    try:
        score = float(data.get("score", 0.0))
    except Exception:
        score = 0.0
    reason = data.get("reason", "")

    if label not in ["correct", "partially_correct", "incorrect"]:
        label = "invalid"
    if not (0.0 <= score <= 1.0):
        score = 0.0

    return {"label": label, "score": score, "reason": reason}


# -----------------------------
# Utils
# -----------------------------
def normalize(text):
    """Lowercase, remove punctuation and extra spaces."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def exact_match(pred, gold):
    return int(normalize(pred) == normalize(gold))

def token_f1(pred, gold):
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def parse_items(text):
    """Split a multi-item answer into normalized set of items."""
    parts = re.split(r'[\n;•-]', text)
    items = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r'^\d+[\.\)]\s*', '', p)
        items.append(normalize(p))
    return set(items)

def list_f1(pred, gold):
    pred_items = parse_items(pred)
    gold_items = parse_items(gold)
    if not pred_items and not gold_items:
        return 1.0
    if not pred_items or not gold_items:
        return 0.0
    inter = pred_items & gold_items
    precision = len(inter) / len(pred_items)
    recall = len(inter) / len(gold_items)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# -----------------------------
# Load QA files
# -----------------------------

def load_reference_json(path):
    """
    {
      "3": "The House of Bernarda Alba",
      "4": "..."
    }
    dict[int_qid] -> {"qid": int_qid, "answer": str}
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    data = {}
    for k, v in obj.items():
        try:
            qid = int(k)
        except ValueError:
            continue
        data[qid] = {
            "qid": qid,
            "answer": v
        }
    return data


def load_output_jsonl(path):
    """
    {"qid": 1, "question": "...", "answer_llama3_8b": "..."}
    """
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = item.get("qid")
            if qid is None:
                continue
            data[int(qid)] = item
    return data


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(reference_path, output_path, list_qids=None,
             use_llm_judge=False, llm_threshold=0.9,
             details_output_path=None):
    ref_data = load_reference_json(reference_path)
    out_data = load_output_jsonl(output_path)

    print(f"Loaded reference items: {len(ref_data)}")
    print(f"Loaded output items: {len(out_data)}")

    total_em, total_f1, n = 0, 0, 0
    list_f1_val, list_count = 0, 0

    llm_total_score = 0.0
    llm_correct_count = 0
    llm_n = 0

    details = []

    for qid, ref_item in tqdm(
        ref_data.items(),
        total=len(ref_data),
        desc="Evaluating QA pairs",
        ncols=100
    ):
        if qid not in out_data:
            print(f"[WARN] Missing output for qid={qid}")
            continue

        out_item = out_data[qid]

        gold = ref_item.get("answer")
        pred = (
            out_item.get("answer_llama3_8b")
            or out_item.get("pred")
            or out_item.get("answer")
        )

        if not gold or not pred:
            continue

        em = exact_match(pred, gold)
        f1 = token_f1(pred, gold)

        if list_qids and qid in list_qids:
            f1 = list_f1(pred, gold)
            list_f1_val += f1
            list_count += 1

        total_em += em
        total_f1 += f1
        n += 1

        if use_llm_judge:
            question = ref_item.get("question") or out_item.get("question", "")
            judge_res = llm_judge_one(question, gold, pred)
            llm_score = judge_res["score"]
            llm_total_score += llm_score
            llm_n += 1
            if llm_score >= llm_threshold:
                llm_correct_count += 1
            sample = {
            "qid": qid,
            "question": out_item.get("question") or ref_item.get("question", ""),
            "gold": gold,
            "pred": pred,
            "em": em,
            "f1": f1,
        }

            if list_qids and qid in list_qids:
                sample["is_list_question"] = True
                sample["list_f1"] = f1
            else:
                sample["is_list_question"] = False

            if use_llm_judge:
                sample["llm_label"] = judge_res["label"]
                sample["llm_score"] = llm_score
                sample["llm_reason"] = judge_res["reason"]

            details.append(sample)

    print("=" * 70)
    print(f"Evaluated {n} QA pairs.")
    print(f"Overall EM: {total_em / n:.3f}")
    print(f"Overall F1: {total_f1 / n:.3f}")
    if list_count > 0:
        print(f"List Qs ({list_count}): F1 = {list_f1_val / list_count:.3f}")

    if use_llm_judge and llm_n > 0:
        print("-" * 70)
        print(f"LLM Judge Model (HF Inference): {HF_MODEL_NAME}")
        print(f"LLM_AvgScore: {llm_total_score / llm_n:.3f}")
        print(f"LLM_CorrectRate (score >= {llm_threshold}): {llm_correct_count / llm_n:.3f}")

        if details_output_path is not None:
            with open(details_output_path, "w", encoding="utf-8") as f:
                for row in details:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[INFO] Saved per-question details to {details_output_path}")

    print("=" * 70)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QA model output against golden reference.")
    parser.add_argument("--chunk", type=str, required=True,
                        help="Name of chunk; reference will be data/reference/reference_<chunk>.json")
    parser.add_argument("--list_qids", type=str, default="",
                        help="Comma-separated qids treated as list-type questions (e.g. '4,5')")
    parser.add_argument("--use_llm_judge", action="store_true",
                        help="Use HF Llama-3-8B-Instruct as LLM judge.")
    parser.add_argument("--llm_threshold", type=float, default=LLM_CORRECT_THRESHOLD,
                        help="Threshold of judge score to treat as correct (default 0.9).")
    parser.add_argument("--llm_model", type=str, default=HF_MODEL_NAME,
                        help="HuggingFace model id for judge (default meta-llama/Meta-Llama-3-8B-Instruct).")

    args = parser.parse_args()

    # reference：JSON；output：JSONL
    reference_path = os.path.join("data/reference", "reference_" + args.chunk + ".json")
    output_path = os.path.join("Llama_output_all28", args.chunk + ".jsonl")

    if args.list_qids.strip() == "":
        list_qids = []
    else:
        list_qids = [int(x) for x in args.list_qids.split(",")]

    HF_MODEL_NAME = args.llm_model
    LLM_CORRECT_THRESHOLD = args.llm_threshold

    details_output_path = f"eval_{args.chunk}_details.jsonl"
    evaluate(
        reference_path,
        output_path,
        list_qids,
        use_llm_judge=args.use_llm_judge,
        llm_threshold=LLM_CORRECT_THRESHOLD,
        details_output_path=details_output_path,
    )
