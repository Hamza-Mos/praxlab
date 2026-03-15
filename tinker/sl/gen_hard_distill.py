"""Generate self-distillation traces on HARD problems only (level 4-5).

Same as gen_self_distill.py but filtered to level 4-5 problems.

Usage: python gen_hard_distill.py <sampler_weights_path> [target]
"""

import json
import re
import sys
import tinker
from tinker import types
from transformers import AutoTokenizer
from datasets import load_dataset
import random

MODEL = "Qwen/Qwen3-8B"
MAX_TOKENS = 4096
TEMPERATURE = 0.5
NUM_SAMPLES = 5
TARGET_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 200


def find_boxed(text):
    spans = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        depth = 0
        j = idx + 6
        while j < len(text):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    spans.append((idx, j + 1, text[idx+7:j]))
                    break
            j += 1
        i = j + 1 if j < len(text) else len(text)
    return spans


def normalize_number(s):
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def answers_match(predicted, expected):
    p, e = normalize_number(predicted), normalize_number(expected)
    if p is not None and e is not None:
        return abs(p - e) < 1e-6
    return predicted.strip() == expected.strip()


def main():
    sampler_path = sys.argv[1]
    print(f"Using sampler: {sampler_path}", flush=True)

    existing_prompts = set()
    with open("data.jsonl") as f:
        for line in f:
            item = json.loads(line.strip())
            existing_prompts.add(re.sub(r'\s+', ' ', item["prompt"].strip()))
    for path in ["../../tinker/rl/eval_prompts.jsonl", "../../tinker/rl/prompts.jsonl"]:
        with open(path) as f:
            for line in f:
                item = json.loads(line.strip())
                existing_prompts.add(re.sub(r'\s+', ' ', item["prompt"].strip()))

    # Load ONLY level 4-5 MATH problems
    subjects = ['algebra', 'counting_and_probability', 'geometry',
                'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    problems = []
    for subj in subjects:
        ds = load_dataset('EleutherAI/hendrycks_math', subj, split='train')
        for ex in ds:
            try:
                level = int(ex['level'].replace('Level ', ''))
            except ValueError:
                continue
            if level < 4:  # ONLY level 4-5
                continue
            pn = re.sub(r'\s+', ' ', ex['problem'].strip())
            if pn in existing_prompts:
                continue
            boxed = find_boxed(ex['solution'])
            if not boxed:
                continue
            problems.append({
                "problem": ex["problem"],
                "level": level,
                "ground_truth": boxed[-1][2],
            })

    random.seed(999)
    random.shuffle(problems)
    selected = problems[:int(TARGET_NEW * 2)]
    print(f"Selected {len(selected)} hard (L4-5) problems", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    sc = tinker.ServiceClient()
    sampling_client = sc.create_sampling_client(base_model=MODEL, model_path=sampler_path)

    stop_seqs = [tokenizer.eos_token] if tokenizer.eos_token else []
    for st in ["<|im_end|>", "<|eot_id|>", "</s>"]:
        if st not in stop_seqs:
            stop_seqs.append(st)

    sp = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, stop=stop_seqs)

    # Submit all
    futures = []
    for item in selected:
        msgs = [{"role": "user", "content": item["problem"]}]
        toks = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        mi = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=toks)])
        f = sampling_client.sample(prompt=mi, num_samples=NUM_SAMPLES, sampling_params=sp)
        futures.append((f, item))

    traces = []
    wrong = 0
    for i, (f, item) in enumerate(futures):
        if len(traces) >= TARGET_NEW:
            break
        try:
            result = f.result()
        except Exception as e:
            continue
        for seq in result.sequences:
            text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            boxed = find_boxed(text)
            if boxed and answers_match(boxed[-1][2], item["ground_truth"]):
                traces.append({"prompt": item["problem"], "response": text})
                break
        else:
            wrong += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(futures)}] verified: {len(traces)}, wrong: {wrong}", flush=True)

    print(f"\nGenerated {len(traces)} hard self-distillation traces")

    with open("hard_traces.jsonl", "w") as f:
        for item in traces:
            f.write(json.dumps(item) + "\n")
    print(f"Saved to hard_traces.jsonl")


if __name__ == "__main__":
    main()
