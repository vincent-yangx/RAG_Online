from datasets import load_dataset
import os
import json

wiki = load_dataset("wiki_qa")

save_path = "./wikiqa/"
os.makedirs(save_path, exist_ok=True)

for split in ["train", "validation", "test"]:
    data = wiki[split]
    with open(os.path.join(save_path, f"{split}.jsonl"), "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

print("WikiQA saved to:", save_path)
