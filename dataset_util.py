import json, os

SYSTEM_COT_PROMPT = (
    "You are a careful math problem solver. "
    "Solve problems using a clear step by step reasoning process. "
    "Follow the requested final answer format exactly."
)

MATH_COT_PROMPT = (
    "Solve the following math problem step by step. Show all your reasoning clearly. "
    "At the end, write your final answer inside \\boxed{{}}.\n\n"
    "Problem: {question}\n\n"
    "Solution:"
)

DATASET_PROMPTS = {"math": MATH_COT_PROMPT}

def format_prompt(prompt: str, question: str) -> str:
    return prompt.format(question=question)

os.makedirs("dataset/sft", exist_ok=True)
kept = 0
skipped=0
with open("dataset/Qwen3_32B_teacher_demos.jsonl") as fin, open("dataset/sft/Qwen3_32B_teacher_demos_for_sft.jsonl", "w") as fout:
    for line in fin:
        d = json.loads(line)
        if not d["is_correct"]:
            skipped+=1
            continue
        if "</think>" not in d["model_response"]:
            skipped+=1
            continue
        fout.write(json.dumps({
            "prompt": [
                {"role": "system", "content": SYSTEM_COT_PROMPT},
                {"role": "user", "content": format_prompt(DATASET_PROMPTS[d["dataset"]], d["question"])},
            ],
            "completion": [{"role": "assistant", "content": d["model_response"]}],
        }) + "\n")
        kept += 1
print(kept, "examples")
print(skipped, "incorrect skipped")