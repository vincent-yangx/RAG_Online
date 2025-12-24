import json
import argparse
from collections import defaultdict

def normalize_question(q: str) -> str:
    return " ".join(q.strip().lower().split())

def load_queries(queries_path):
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)  # {qid: question_text}
    qtext2qid = {}
    for qid, qtext in queries.items():
        key = normalize_question(qtext)
        qtext2qid[key] = qid
    return queries, qtext2qid

def load_qrels(qrels_path):
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels = json.load(f)  # {qid: {doc_id: 1, ...}}
    return qrels

def load_retrieval_results(retrieval_path):
    """
    load the retrieved result into data ana wait for future use
    """
    data = []
    with open(retrieval_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    return data

def evaluate(retrieval_data, qtext2qid, qrels, k=5):
    total = 0
    hit_at_k = 0
    mrr_sum = 0.0

    skipped = 0

    for entry in retrieval_data:
        q_text = entry["question"]
        key = normalize_question(q_text)

        if key not in qtext2qid:
            # question not in qrels
            skipped += 1
            continue

        qid = qtext2qid[key]
        gold_dict = qrels.get(qid, {})
        if not gold_dict:
            # no positive answer
            skipped += 1
            continue

        gold_docs = set(gold_dict.keys())

        retrieved_docs = [r["chunk_id"] for r in entry.get("retrieved", [])]

        total += 1

        # Recall@k
        topk = retrieved_docs[:k]
        if gold_docs & set(topk):
            hit_at_k += 1

        # MRR@k
        rr = 0.0
        for rank, doc_id in enumerate(topk, start=1):
            if doc_id in gold_docs:
                rr = 1.0 / rank
                break
        mrr_sum += rr

    recall_k = hit_at_k / total if total > 0 else 0.0
    mrr_k = mrr_sum / total if total > 0 else 0.0

    return {
        "total_evaluated": total,
        "skipped": skipped,
        "recall@k": recall_k,
        "mrr@k": mrr_k,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=str, required=True,
                        help="path to retrieval_wikiqa.jsonl")
    parser.add_argument("--queries", type=str, required=True,
                        help="path to wikiqa_queries.json")
    parser.add_argument("--qrels", type=str, required=True,
                        help="path to wikiqa_qrels.json")
    parser.add_argument("--k", type=int, default=5,
                        help="cutoff for Recall@k / MRR@k")
    args = parser.parse_args()

    queries, qtext2qid = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    retrieval_data = load_retrieval_results(args.retrieval)

    metrics = evaluate(retrieval_data, qtext2qid, qrels, k=args.k)

    print("===== WikiQA Retrieval Evaluation =====")
    print(f"Eval @k = {args.k}")
    print(f"Total evaluated queries: {metrics['total_evaluated']}")
    print(f"Skipped (no gold / no mapping): {metrics['skipped']}")
    print(f"Recall@{args.k}: {metrics['recall@k']:.4f}")
    print(f"MRR@{args.k}:    {metrics['mrr@k']:.4f}")

if __name__ == "__main__":
    main()
