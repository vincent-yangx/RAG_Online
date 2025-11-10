import json
import re
from collections import Counter
import argparse
import os

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
def load_jsonl(path):
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
def evaluate(reference_path, output_path, list_qids=None):
    ref_data = load_jsonl(reference_path)
    out_data = load_jsonl(output_path)

    print(f"Loaded reference items: {len(ref_data)}")
    print(f"Loaded output items: {len(out_data)}")


    total_em, total_f1, n = 0, 0, 0
    list_f1_val, list_count = 0, 0

    for qid, ref_item in ref_data.items():
        if qid not in out_data:
            print(f"[WARN] Missing output for qid={qid}")
            continue

        gold = ref_item.get("answer") or ref_item.get("gold") or ref_item.get("reference")
        pred = out_data[qid].get("answer_llama3_8b") or out_data[qid].get("pred") or out_data[qid].get("answer")

        if not gold or not pred:
            continue

        em = exact_match(pred, gold)
        f1 = token_f1(pred, gold)

        # 列表题用 list F1
        if list_qids and qid in list_qids:
            f1 = list_f1(pred, gold)
            list_f1_val += f1
            list_count += 1

        total_em += em
        total_f1 += f1
        n += 1

    print("=" * 70)
    print(f"Evaluated {n} QA pairs.")
    print(f"Overall EM: {total_em / n:.3f}")
    print(f"Overall F1: {total_f1 / n:.3f}")
    if list_count > 0:
        print(f"List Qs ({list_count}): F1 = {list_f1_val / list_count:.3f}")
    print("=" * 70)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QA model output against golden reference.")
    parser.add_argument("--chunk", type=str, required=True, help="Path to golden reference JSONL file (under data/reference/).")
    parser.add_argument("--list_qids", type=str, default="", help="Comma-separated qids treated as list-type questions (e.g. '4,5')")
    args = parser.parse_args()
    reference_path = os.path.join("data/reference","reference_" + args.chunk + ".jsonl")
    output_path = os.path.join("Llama_output", args.chunk + ".jsonl")
    if args.list_qids.strip() == "":
        with open(reference_path, "r", encoding="utf-8") as f:
            list_qids = [json.loads(line).get("qid") for line in f if "qid" in line]
            list_qids = [int(q) for q in list_qids if q is not None]
    else:
        list_qids = [int(x) for x in args.list_qids.split(",")]
    evaluate(reference_path, output_path, list_qids)
