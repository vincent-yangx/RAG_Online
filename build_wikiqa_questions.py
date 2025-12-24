'''
The script is useless  
'''
import json
import os

with open("./wikiqa_test/wikiqa_queries.json", "r", encoding="utf-8") as f:
    queries = json.load(f)

with open("./wikiqa_test/wikiqa_qrels.json", "r", encoding="utf-8") as f:
    qrels = json.load(f)

os.makedirs("wikiqa_data/test", exist_ok=True)
out_path = "wikiqa_data/test/question_wikiqa.txt"

with open(out_path, "w", encoding="utf-8") as f:
    for qid, qtext in queries.items():
        if qid in qrels and len(qrels[qid]) > 0:
            f.write(qtext.strip() + "\n")

print("Saved questions to", out_path)
