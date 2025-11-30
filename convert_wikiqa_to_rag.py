import json
import os

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def convert_wikiqa_split_grouped(jsonl_path, corpus, queries, qrels, start_doc_idx=0, start_q_idx=0):
    data = load_jsonl(jsonl_path)

    # same question to the same qid
    question2qid = {}
    doc_idx = start_doc_idx
    q_idx = start_q_idx

    for item in data:
        question = item["question"]
        answer   = item["answer"]
        title    = item["document_title"]
        label    = item["label"]  # 0/1

        # if the question appears for the first time, assign a new qid
        if question not in question2qid:
            qid = f"q_{q_idx}"
            question2qid[question] = qid
            queries[qid] = question
            q_idx += 1
        else:
            qid = question2qid[question]

        doc_id = f"d_{doc_idx}"
        doc_idx += 1

        corpus[doc_id] = {
            "title": title,
            "text": answer
        }

        if label == 1:
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = 1

    return doc_idx, q_idx

def build_wikiqa_rag_format(base_path="./wikiqa"):
    corpus = {}
    queries = {}
    qrels = {}

    doc_idx = 0
    q_idx = 0

    # 目前只用 test 集做评测
    for split in ["test"]:
        jsonl_path = os.path.join(base_path, f"{split}.jsonl")
        doc_idx, q_idx = convert_wikiqa_split_grouped(
            jsonl_path, corpus, queries, qrels,
            start_doc_idx=doc_idx, start_q_idx=q_idx
        )

    return corpus, queries, qrels

if __name__ == "__main__":
    corpus, queries, qrels = build_wikiqa_rag_format("./wikiqa")
    print("Corpus size:", len(corpus))
    print("Queries size:", len(queries))
    print("Qrels size:", len(qrels))

    # 看看某个 qid 对应多少个候选答案
    some_qid = list(queries.keys())[0]
    print("Example qid:", some_qid)
    print("Question:", queries[some_qid])
    print("Positive doc_ids:", qrels.get(some_qid, {}))

    output_dir = "./wikiqa_test"
    os.makedirs(output_dir, exist_ok=True)

    # 保存 corpus
    with open(os.path.join(output_dir, "wikiqa_corpus.json"), "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    # 保存 queries
    with open(os.path.join(output_dir, "wikiqa_queries.json"), "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    # 保存 qrels
    with open(os.path.join(output_dir, "wikiqa_qrels.json"), "w", encoding="utf-8") as f:
        json.dump(qrels, f, ensure_ascii=False, indent=2)
    
    print("Saved output the folder wikiqa_test")
