import json
import os

with open("./wikiqa_test/wikiqa_corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

os.makedirs("wikiqa_data/chunks", exist_ok=True)
out_path = "wikiqa_data/chunks/chunks_wikiqa.jsonl"

with open(out_path, "w", encoding="utf-8") as f:
    for doc_id, doc in corpus.items():
        line = {
            "chunk_id": doc_id,                      
            "source": doc.get("title", "wikiqa"), 
            "text": doc["text"]              
        }
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

print("Saved chunks to", out_path)
