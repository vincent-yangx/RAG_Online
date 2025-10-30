import json
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser(description="Please enter the retrieve mode to use and dataset to test")
parser.add_argument("--exe", type=str, required=True,help="Specify what execution to do ")
args = parser.parse_args()



def renumber_jsonl(input_path, output_path, width=3, prefix=""):
    """
    renumber the chunk_id from 000
    """
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        counter = 0
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"skip invalid JSON row: {line!r} error: {e}")
                continue
            
            new_id = f"{prefix}{counter:0{width}d}"
            obj["chunk_id"] = new_id
            
            fout.write(json.dumps(obj, ensure_ascii=False))
            fout.write("\n")
            
            counter += 1



def generate_reference(txt_path, json_path):
    txt_path = txt_path
    json_path = json_path

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()] 

    
    data = {str(i + 1): line for i, line in enumerate(lines)}


    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([data], f, ensure_ascii=False, indent=4)

    print("Generated json file!:", json_path)


def merge_question_files(input_dir, output_file):
    """
    merge all question text file
    """
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Delete: {output_file}")

    total_q = 0

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in sorted(os.listdir(input_dir)):
            if filename.endswith(".txt"):
                file_path = os.path.join(input_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    lines = [line.strip() for line in infile if line.strip()]
                    outfile.write("\n".join(lines) + "\n")
                    print(f"This file has {len(lines)} questions")
                    total_q += len(lines)
                print(f"merge file: {filename}")
    print(f"\n merging complete: {output_file}")
    print(f"Total questions merged: {total_q}\n")



def merge_reference_files(input_dir, output_file):
    """
    merge all reference json files
    """

    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Delete: {output_file}")
    all_refs = {}
    current_id = 1

    for filename in sorted(os.listdir(input_dir)):
        if filename.startswith("reference_") and filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as infile:
                data = json.load(infile)
                answers = data[0]
                print(f"This file has{len(answers)} answers")
                for _, answer in answers.items():
                    all_refs[str(current_id)] = answer
                    current_id += 1
            print(f"merge file: {filename}")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(all_refs, outfile, indent=4, ensure_ascii=False)
    
    print(f"\n merge complete，total {len(all_refs)} answers，output file: {output_file}")


def merge_jsonl_texts(input_dir, output_file):
    """
    merge all jsonl files
    """
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Delete: {output_file}")
    
    merged_data = []
    chunk_id = 0
    fail_count = 0


    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if "text" in obj:
                            merged_data.append({
                                "chunk_id": f"{chunk_id:04d}",
                                "text": obj["text"].strip()
                            })
                            chunk_id += 1
                    except json.JSONDecodeError:
                        print(f"skip invalid row: {line[:80]}...")
            print(f"processed file: {filename}")

    with open(output_file, "w", encoding="utf-8") as outfile:
        for item in merged_data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write("\n")

    print(f"\n 合并完成，共 {len(merged_data)} 个 chunk，输出文件为: {output_file}")


def convert():
    # 输入 CSV 文件路径
    csv_path = "test_set_day3.csv"

    # 输出 TXT 文件路径
    txt_path = "data/test/test_set_day3.txt"

    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 提取 'Question' 列并去掉空值
    questions = df["Question"].dropna().astype(str).tolist()

    # 写入 TXT 文件
    with open(txt_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(q.strip() + "\n")

    print(f"✅ 已成功写入 {len(questions)} 条问题到 {txt_path}")





if __name__ == "__main__":

    if args.exe == "renumber":
        input_path = "data/chunks/chunks_thefrick_2.jsonl"
        output_path = "data/chunks/chunks_thefrick.jsonl"
        renumber_jsonl(input_path, output_path, width=3, prefix="")
        print("total row number：", sum(1 for _ in open(output_path, 'r', encoding='utf-8')))

    elif args.exe == "reference":
        txt_path = "data/reference/reference_picksburgh.txt"
        json_path = "data/reference/reference_picklesburgh.json"
        generate_reference(txt_path,json_path)

    elif args.exe == "merge_all":
        # input_dir = "data/chunks" 
        # output_file = "data/chunks/chunks_all28.jsonl"
        # merge_jsonl_texts(input_dir, output_file)

        input_dir = "data/test"
        output_file = "data/test/question_all28.txt"
        merge_question_files(input_dir, output_file)


        input_dir = "data/reference" 
        output_file = "data/reference/reference_all28.json"
        merge_reference_files(input_dir, output_file)
    elif args.exe == "convert":
        convert()
