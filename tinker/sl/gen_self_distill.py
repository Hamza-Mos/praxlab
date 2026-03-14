"""Generate self-distillation traces: use the SFT model to solve NEW problems.

Uses the best SFT model's sampler weights to generate traces on problems
not in the current training set, verifies answers against ground truth,
and adds verified traces to data.jsonl.

Usage: python gen_self_distill.py <sampler_weights_path>
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
MAX_TOKENS = 2048
TEMPERATURE = 0.7
NUM_SAMPLES = 5  # Sample 5 per problem, keep best verified
TARGET_NEW = 400  # Target number of new verified traces


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
    pred_num = normalize_number(predicted)
    exp_num = normalize_number(expected)
    if pred_num is not None and exp_num is not None:
        return abs(pred_num - exp_num) < 1e-6
    return predicted.strip() == expected.strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python gen_self_distill.py <sampler_weights_path>")
        sys.exit(1)

    sampler_path = sys.argv[1]
    print(f"Using sampler weights: {sampler_path}", flush=True)

    # Load existing training prompts to exclude
    existing_prompts = set()
    with open("data.jsonl") as f:
        for line in f:
            item = json.loads(line.strip())
            existing_prompts.add(re.sub(r'\s+', ' ', item["prompt"].strip()))

    # Also exclude RL eval/train prompts
    for path in ["../../tinker/rl/eval_prompts.jsonl", "../../tinker/rl/prompts.jsonl"]:
        with open(path) as f:
            for line in f:
                item = json.loads(line.strip())
                existing_prompts.add(re.sub(r'\s+', ' ', item["prompt"].strip()))

    print(f"Excluding {len(existing_prompts)} existing prompts", flush=True)

    # Load NEW MATH problems
    subjects = ['algebra', 'counting_and_probability', 'geometry',
                'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

    new_problems = []
    for subj in subjects:
        ds = load_dataset('EleutherAI/hendrycks_math', subj, split='train')
        for ex in ds:
            try:
                level_num = int(ex['level'].replace('Level ', ''))
            except ValueError:
                continue
            if level_num < 2:
                continue
            pn = re.sub(r'\s+', ' ', ex['problem'].strip())
            if pn in existing_prompts:
                continue
            boxed = find_boxed(ex['solution'])
            if not boxed:
                continue
            new_problems.append({
                "problem": ex["problem"],
                "level": level_num,
                "ground_truth": boxed[-1][2],
            })

    random.seed(123)
    random.shuffle(new_problems)
    # Take enough to get TARGET_NEW verified (expect ~80% pass rate with our model)
    selected = new_problems[:int(TARGET_NEW * 1.5)]
    print(f"Selected {len(selected)} new problems for self-distillation", flush=True)

    # Setup sampler
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        base_model=MODEL, model_path=sampler_path
    )

    stop_sequences = []
    if tokenizer.eos_token:
        stop_sequences.append(tokenizer.eos_token)
    for stop_tok in ["<|im_end|>", "<|eot_id|>", "</s>"]:
        if stop_tok not in stop_sequences:
            stop_sequences.append(stop_tok)

    sampling_params = types.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=stop_sequences if stop_sequences else None,
    )

    # Submit all sampling requests as futures (parallel)
    print("Submitting sampling requests...", flush=True)
    futures_list = []
    for item in selected:
        messages = [{"role": "user", "content": item["problem"]}]
        token_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        model_input = types.ModelInput(
            chunks=[types.EncodedTextChunk(tokens=token_ids)]
        )
        future = sampling_client.sample(
            prompt=model_input, num_samples=NUM_SAMPLES,
            sampling_params=sampling_params
        )
        futures_list.append((future, item))

    print(f"Submitted {len(futures_list)} requests, collecting results...", flush=True)

    # Collect results
    new_traces = []
    wrong = 0
    no_answer = 0

    for i, (future, item) in enumerate(futures_list):
        if len(new_traces) >= TARGET_NEW:
            break

        try:
            result = future.result()
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr, flush=True)
            continue

        # Check each sample for correctness
        found = False
        for seq in result.sequences:
            text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            boxed = find_boxed(text)
            if not boxed:
                continue
            predicted = boxed[-1][2]
            if answers_match(predicted, item["ground_truth"]):
                new_traces.append({
                    "prompt": item["problem"],
                    "response": text,
                })
                found = True
                break

        if not found:
            had_answer = any(find_boxed(tokenizer.decode(seq.tokens, skip_special_tokens=True))
                          for seq in result.sequences)
            if had_answer:
                wrong += 1
            else:
                no_answer += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(futures_list)}] verified: {len(new_traces)}, "
                  f"wrong: {wrong}, no_answer: {no_answer}", flush=True)

    print(f"\nGenerated {len(new_traces)} self-distillation traces")
    print(f"Wrong: {wrong}, No answer: {no_answer}")

    # Combine with existing data
    existing = []
    with open("data.jsonl") as f:
        for line in f:
            if line.strip():
                existing.append(json.loads(line))

    combined = existing + new_traces
    random.shuffle(combined)

    with open("data.jsonl", "w") as f:
        for item in combined:
            f.write(json.dumps(item) + "\n")

    print(f"Combined: {len(existing)} existing + {len(new_traces)} new = {len(combined)} total")

    # Show samples
    for item in new_traces[:2]:
        print(f"\nPROMPT: {item['prompt'][:100]}")
        print(f"RESPONSE: {item['response'][:400]}")
        print("---")


if __name__ == "__main__":
    main()
