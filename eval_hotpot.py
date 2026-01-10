#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_hotpot.py

Evaluate your retrieval outputs on HotpotQA prepared by prepare_hotpot.py.

Inputs
------
1) Retrieval output JSONL (from rag_pipeline.py):
   {"qid": 1, "question": "...", "retrieved": [{"rank":1,"chunk_id":"...","source":"...","text":"...","score":...}, ...]}

2) Gold JSONL (from prepare_hotpot.py):
   {"qid":"<hotpot_id>", "question":"...", "supporting_facts":[[title, sent_id],...],
    "gold_titles":[...], "gold_chunk_ids":[...], "context_titles":[...]}

Metrics (per K)
---------------
- title_recall@K: fraction of gold_titles covered by top-K retrieved titles
- title_pair_recall@K: 1 if ALL gold_titles are in top-K titles, else 0
- chunk_recall@K: fraction of gold_chunk_ids covered by top-K retrieved chunk_ids
- mrr_title: reciprocal rank of first gold title (min rank among retrieved titles), averaged
- mrr_chunk: reciprocal rank of first gold chunk_id, averaged

Alignment
---------
Default is by LINE ORDER (pred line i vs gold line i). This is correct if you ran rag_pipeline.py
on data/test/question_<chunk>.txt produced by prepare_hotpot.py.

You can use --align question to align by exact question string (safer if files got reordered),
but it assumes questions are unique (or it will take the first match).

Usage
-----
python evaluate_hotpot.py \
  --pred results/retrieval_hotpot_dev.jsonl \
  --gold data/test/hotpot_gold_hotpot_dev.jsonl \
  --ks 1 2 5 10 20 \
  --out results/eval_hotpot_dev.json
"""

from __future__ import annotations
import argparse
import json
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import os

def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def normalize_title(t: str) -> str:
    return " ".join((t or "").strip().split())

def safe_set(xs: List[str]) -> set:
    return set([normalize_title(x) for x in xs if (x or "").strip()])

def extract_titles_from_pred(retrieved: List[dict]) -> List[str]:
    titles: List[str] = []
    for r in retrieved:
        src = r.get("source", "")
        if src:
            titles.append(normalize_title(str(src)))
    return titles

def extract_chunk_ids_from_pred(retrieved: List[dict]) -> List[str]:
    cids: List[str] = []
    for r in retrieved:
        cid = r.get("chunk_id", "")
        if cid:
            cids.append(str(cid))
    return cids

def recall_at_k(gold: List[str], pred: List[str], k: int) -> float:
    g = safe_set(gold)
    if not g:
        return 0.0
    topk = safe_set(pred[:k])
    return len(g & topk) / float(len(g))

def pair_recall_at_k(gold: List[str], pred: List[str], k: int) -> float:
    g = safe_set(gold)
    if not g:
        return 0.0
    topk = safe_set(pred[:k])
    return 1.0 if g.issubset(topk) else 0.0

def mrr_title(gold: List[str], pred: List[str]) -> float:
    g = safe_set(gold)
    if not g:
        return 0.0
    for i, p in enumerate(pred):
        if normalize_title(p) in g:
            return 1.0 / float(i + 1)
    return 0.0

def mrr_exact(gold: List[str], pred: List[str]) -> float:
    g = set([str(x) for x in gold if x is not None])
    if not g:
        return 0.0
    for i, p in enumerate(pred):
        if str(p) in g:
            return 1.0 / float(i + 1)
    return 0.0

def align_by_question(pred_rows: List[dict], gold_rows: List[dict]) -> List[Tuple[dict, dict]]:
    gold_map = defaultdict(list)
    for g in gold_rows:
        q = (g.get("question", "") or "").strip()
        gold_map[q].append(g)

    pairs: List[Tuple[dict, dict]] = []
    missing = 0
    multi = 0
    for p in pred_rows:
        q = (p.get("question", "") or "").strip()
        cand = gold_map.get(q, [])
        if not cand:
            missing += 1
            continue
        if len(cand) > 1:
            multi += 1
        pairs.append((p, cand[0]))

    if missing:
        print(f"[WARN] align=question: {missing} pred questions not found in gold.")
    if multi:
        print(f"[WARN] align=question: {multi} questions are not unique in gold; using first match.")
    return pairs

def align_by_order(pred_rows: List[dict], gold_rows: List[dict]) -> List[Tuple[dict, dict]]:
    n = min(len(pred_rows), len(gold_rows))
    if len(pred_rows) != len(gold_rows):
        print(f"[WARN] align=order: pred has {len(pred_rows)} lines, gold has {len(gold_rows)} lines. Using first {n}.")
    return list(zip(pred_rows[:n], gold_rows[:n]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=str, required=True, help="Retrieval output jsonl (from rag_pipeline.py)")
    ap.add_argument("--gold", type=str, required=True, help="Gold jsonl (from prepare_hotpot.py)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1,2,5,10,20], help="K values for Recall@K")
    ap.add_argument("--align", type=str, default="order", choices=["order","question"],
                    help="How to align pred rows with gold rows (default: order)")
    ap.add_argument("--out", type=str, default="", help="Optional: write summary + per-q metrics as JSON")
    args = ap.parse_args()

    pred_rows = read_jsonl(args.pred)
    gold_rows = read_jsonl(args.gold)

    if args.align == "question":
        pairs = align_by_question(pred_rows, gold_rows)
    else:
        pairs = align_by_order(pred_rows, gold_rows)

    if not pairs:
        raise SystemExit("[ERROR] No aligned pairs to evaluate. Check your paths and --align option.")

    # Accumulators
    agg_title_recall = {k: 0.0 for k in args.ks}
    agg_title_pair = {k: 0.0 for k in args.ks}
    agg_chunk_recall = {k: 0.0 for k in args.ks}
    agg_mrr_title = 0.0
    agg_mrr_chunk = 0.0

    per_q: List[dict] = []

    for p, g in pairs:
        retrieved = p.get("retrieved", []) or []
        pred_titles = extract_titles_from_pred(retrieved)
        pred_cids = extract_chunk_ids_from_pred(retrieved)

        gold_titles = g.get("gold_titles") or []
        gold_cids = g.get("gold_chunk_ids") or []

        row = {
            "qid_pred": p.get("qid"),
            "qid_gold": g.get("qid"),
            "question": p.get("question", g.get("question", "")),
            "gold_titles": gold_titles,
            "gold_chunk_ids": gold_cids,
            "metrics": {}
        }

        for k in args.ks:
            tr = recall_at_k(gold_titles, pred_titles, k)
            tpr = pair_recall_at_k(gold_titles, pred_titles, k)
            cr = recall_at_k([str(x) for x in gold_cids], [str(x) for x in pred_cids], k)

            agg_title_recall[k] += tr
            agg_title_pair[k] += tpr
            agg_chunk_recall[k] += cr

            row["metrics"][f"title_recall@{k}"] = tr
            row["metrics"][f"title_pair_recall@{k}"] = tpr
            row["metrics"][f"chunk_recall@{k}"] = cr

        mt = mrr_title(gold_titles, pred_titles)
        mc = mrr_exact([str(x) for x in gold_cids], [str(x) for x in pred_cids])
        agg_mrr_title += mt
        agg_mrr_chunk += mc
        row["metrics"]["mrr_title"] = mt
        row["metrics"]["mrr_chunk"] = mc

        per_q.append(row)

    n = float(len(per_q))
    summary = {"n": int(n)}
    for k in args.ks:
        summary[f"title_recall@{k}"] = agg_title_recall[k] / n
        summary[f"title_pair_recall@{k}"] = agg_title_pair[k] / n
        summary[f"chunk_recall@{k}"] = agg_chunk_recall[k] / n
    summary["mrr_title"] = agg_mrr_title / n
    summary["mrr_chunk"] = agg_mrr_chunk / n

    print("=== Hotpot Retrieval Evaluation ===")
    print(f"pred:  {args.pred}")
    print(f"gold:  {args.gold}")
    print(f"align: {args.align}")
    print(f"N:     {summary['n']}")
    print()
    for k in args.ks:
        print(f"Title Recall@{k}:      {summary[f'title_recall@{k}']:.4f}")
        print(f"Title Pair Recall@{k}: {summary[f'title_pair_recall@{k}']:.4f}")
        print(f"Chunk Recall@{k}:      {summary[f'chunk_recall@{k}']:.4f}")
        print("-" * 40)
    print(f"MRR Title: {summary['mrr_title']:.4f}")
    print(f"MRR Chunk: {summary['mrr_chunk']:.4f}")

    if args.out:
        out_obj = {"summary": summary, "per_question": per_q}
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(f"\nWrote: {args.out}")

if __name__ == "__main__":
    main()
