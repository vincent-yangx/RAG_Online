from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Tuple, Any, Iterable, Optional

def read_hotpot(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def safe_str(x: Any) -> str:
    return "" if x is None else str(x)

def join_sents(sents: List[str]) -> str:
    return " ".join([safe_str(s).strip() for s in sents if safe_str(s).strip()]).strip()

def make_chunk_id(prefix: str, qid: str, title: str, *, per_example: bool, sent_idx: Optional[int] = None) -> str:
    if per_example:
        base = f"{prefix}::{qid}::{title}"
    else:
        base = f"{prefix}::{title}"
    if sent_idx is None:
        return base
    return f"{base}::s{sent_idx}" 

def iter_context(ctx: Any) -> Iterable[Tuple[str, List[str]]]:
    if not isinstance(ctx, list):
        return
    for item in ctx:
        if not isinstance(item, list) or len(item) != 2:
            continue
        title = safe_str(item[0])
        sents = item[1] if isinstance(item[1], list) else []
        sents = [safe_str(s) for s in sents]
        yield title, sents

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="hotpot_dev_distractor_v1.json", help="Path to HotpotQA JSON (e.g., hotpot_dev_distractor_v1.json)")
    ap.add_argument("--chunk", type=str, default="hotpot_dev", help="Output chunk name used by your pipeline (e.g., hotpot_dev)")
    ap.add_argument("--out_base", type=str, default="data", help="Base output dir (default: data/)")
    ap.add_argument("--granularity", type=str, default="paragraph", choices=["paragraph", "sentence"],
                    help="Chunk granularity: paragraph (title-level) or sentence (title+sent_id)")
    ap.add_argument("--per_example", action="store_true",
                    help="If set, chunk_id includes qid (hp::qid::title...). Safer for sentence-level alignment, but larger corpus.")
    ap.add_argument("--max_samples", type=int, default=0, help="Optional limit for quick tests (0 = no limit)")
    ap.add_argument("--prefix", type=str, default="hp", help="Chunk id prefix (default: hp)")
    args = ap.parse_args()

    # For sentence granularity, per-example is strongly recommended.
    if args.granularity == "sentence" and not args.per_example:
        print("[WARN] You selected sentence granularity without --per_example. "
              "This may break supporting_facts alignment if titles repeat across examples. "
              "Consider adding --per_example.")

    out_chunks_dir = os.path.join(args.out_base, "chunks")
    out_test_dir = os.path.join(args.out_base, "test")
    os.makedirs(out_chunks_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)

    chunks_path = os.path.join(out_chunks_dir, f"chunks_{args.chunk}.jsonl")
    qtxt_path = os.path.join(out_test_dir, f"question_{args.chunk}.txt")
    qjsonl_path = os.path.join(out_test_dir, f"questions_{args.chunk}.jsonl")
    gold_path = os.path.join(out_test_dir, f"hotpot_gold_{args.chunk}.jsonl")

    data = read_hotpot(args.input)
    if args.max_samples and args.max_samples > 0:
        data = data[:args.max_samples]

    # Chunk storage
    # If not per_example, we may deduplicate by chunk_id; keep the first seen.
    chunk_store: Dict[str, dict] = {}
    dup_count = 0

    # Outputs
    questions_txt: List[str] = []
    questions_jsonl: List[dict] = []
    gold_jsonl: List[dict] = []

    # Stats
    n_ctx_titles = 0
    n_ctx_sents = 0

    for ex_i, ex in enumerate(data):
        qid = safe_str(ex.get("_id") or ex.get("id") or ex.get("qid") or ex_i)
        question = safe_str(ex.get("question")).strip()
        answer = safe_str(ex.get("answer")).strip()
        sf = ex.get("supporting_facts", [])
        ctx = ex.get("context", [])

        if not question:
            continue

        questions_txt.append(question)
        questions_jsonl.append({"qid": qid, "question": question})

        # Build a map: title -> list of sents for THIS example (needed for sentence granularity & gold mapping)
        ex_context_titles: List[str] = []
        for title, sents in iter_context(ctx):
            ex_context_titles.append(title)
            n_ctx_titles += 1
            n_ctx_sents += len(sents)

            if args.granularity == "paragraph":
                cid = make_chunk_id(args.prefix, qid, title, per_example=args.per_example, sent_idx=None)
                text = join_sents(sents)
                doc = {"chunk_id": cid, "source": title, "text": text}
                if args.per_example:
                    doc["qid"] = qid
                if cid in chunk_store:
                    dup_count += 1
                else:
                    chunk_store[cid] = doc

            else:  # sentence granularity
                for si, sent in enumerate(sents):
                    cid = make_chunk_id(args.prefix, qid, title, per_example=args.per_example, sent_idx=si)
                    text = safe_str(sent).strip()
                    doc = {"chunk_id": cid, "source": title, "text": text}
                    if args.per_example:
                        doc["qid"] = qid
                    if cid in chunk_store:
                        dup_count += 1
                    else:
                        chunk_store[cid] = doc

        # Derive gold chunk ids from supporting_facts
        gold_titles = []
        gold_chunk_ids = []
        sf_pairs: List[Tuple[str, int]] = []
        if isinstance(sf, list):
            for pair in sf:
                if isinstance(pair, list) and len(pair) == 2:
                    t = safe_str(pair[0])
                    try:
                        si = int(pair[1])
                    except Exception:
                        continue
                    sf_pairs.append((t, si))

        for (t, si) in sf_pairs:
            gold_titles.append(t)
            if args.granularity == "paragraph":
                gcid = make_chunk_id(args.prefix, qid, t, per_example=args.per_example, sent_idx=None)
            else:
                gcid = make_chunk_id(args.prefix, qid, t, per_example=args.per_example, sent_idx=si)
            gold_chunk_ids.append(gcid)

        gold_entry = {
            "qid": qid,
            "question": question,
            "answer": answer,
            "type": ex.get("type", None),
            "level": ex.get("level", None),
            "supporting_facts": [[t, si] for (t, si) in sf_pairs],
            "gold_titles": sorted(set(gold_titles)),
            "gold_chunk_ids": gold_chunk_ids,
            "context_titles": ex_context_titles,
        }
        gold_jsonl.append(gold_entry)

    # Write outputs
    with open(chunks_path, "w", encoding="utf-8") as f:
        for _, doc in chunk_store.items():
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    with open(qtxt_path, "w", encoding="utf-8") as f:
        for q in questions_txt:
            f.write(q.replace("\n", " ").strip() + "\n")

    with open(qjsonl_path, "w", encoding="utf-8") as f:
        for row in questions_jsonl:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(gold_path, "w", encoding="utf-8") as f:
        for row in gold_jsonl:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=== HotpotQA preparation done ===")
    print(f"Input: {args.input}")
    print(f"Examples processed: {len(questions_txt)}")
    print(f"Chunks written: {len(chunk_store)}  (dup skipped: {dup_count})")
    print(f"Avg context titles / example: {n_ctx_titles / max(1, len(questions_txt)):.2f}")
    print(f"Avg context sentences / example: {n_ctx_sents / max(1, len(questions_txt)):.2f}")
    print(f"Wrote: {chunks_path}")
    print(f"Wrote: {qtxt_path}")
    print(f"Wrote: {qjsonl_path}")
    print(f"Wrote: {gold_path}")
    print()
    print("Next steps:")
    print(f"1) Build embeddings + indexes for chunk='{args.chunk}' (your existing embedding/index script).") 
    print(f"2) Run retrieval: python rag_pipeline.py --chunk {args.chunk} --retriever hybrid --rewrite mq --rerank bge --out_dir results/")
    print()
    print("Tip: start with --max_samples 200 to sanity-check before generating the full corpus.")

if __name__ == "__main__":
    main()
