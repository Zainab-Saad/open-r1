import json, os
os.makedirs("dataset/sft", exist_ok=True)
kept = 0
with open("dataset/Qwen3_32B_teacher_demos.jsonl") as fin, open("dataset/sft/Qwen3_32B_teacher_demos_for_sft.jsonl", "w") as fout:
    for line in fin:
        d = json.loads(line)
        if not d["is_correct"]:
            continue
        if "</think>" not in d["model_response"]:
            continue
        fout.write(json.dumps({
            "prompt":     [{"role": "user", "content": d["question"]}],
            "completion": [{"role": "assistant", "content": d["model_response"]}],
        }) + "\n")
        kept += 1
print(kept, "examples")
