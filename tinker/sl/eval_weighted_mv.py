"""Weighted majority vote: penalize answers from short/unverified reasoning.

Instead of raw MV, weight each answer by reasoning quality:
- Longer reasoning = higher weight (more thought = more reliable)
- Contains verification ("check", "verify") = bonus weight
- Contains boxed answer = bonus weight

Usage: python eval_weighted_mv.py <sampler_path> [num_samples] [temp]
"""

import json
import re
import sys
from collections import defaultdict
import tinker
from tinker import types
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
MAX_TOKENS = 4096
EVAL_PROMPTS_PATH = "../../tinker/rl/eval_prompts.jsonl"

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _normalize_number(s):
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_answer(completion):
    boxed = _BOXED_RE.findall(completion)
    if boxed:
        return boxed[-1].strip()
    nums = _NUM_RE.findall(completion.replace(",", ""))
    if nums:
        return nums[-1].strip()
    return None


def check_correct(answer, gt):
    if answer is None:
        return False
    exp, pred = _normalize_number(gt), _normalize_number(answer)
    if exp is not None and pred is not None:
        return abs(pred - exp) < 1e-6
    return answer.strip() == gt.strip()


def compute_weight(text):
    """Weight a completion by reasoning quality."""
    w = 1.0
    # Length bonus (longer = more reasoning)
    w += min(len(text) / 2000, 2.0)  # cap at 3x for very long
    # Verification bonus
    text_lower = text.lower()
    if any(v in text_lower for v in ["check", "verify", "let me confirm", "indeed"]):
        w += 0.5
    # Boxed answer bonus (formatted correctly)
    if _BOXED_RE.search(text):
        w += 0.5
    return w


def main():
    sampler_path = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    temperature = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3

    eval_prompts = []
    with open(EVAL_PROMPTS_PATH) as f:
        for line in f:
            if line.strip():
                eval_prompts.append(json.loads(line))

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    sc = tinker.ServiceClient()
    sampling_client = sc.create_sampling_client(base_model=MODEL, model_path=sampler_path)

    stop_seqs = [tokenizer.eos_token] if tokenizer.eos_token else []
    for st in ["<|im_end|>", "<|eot_id|>", "</s>"]:
        if st not in stop_seqs:
            stop_seqs.append(st)

    sp = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=temperature, stop=stop_seqs)

    futures = []
    for item in eval_prompts:
        msgs = [{"role": "user", "content": item["prompt"]}]
        toks = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        mi = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=toks)])
        f = sampling_client.sample(prompt=mi, num_samples=num_samples, sampling_params=sp)
        futures.append((f, item))

    correct_raw = 0
    correct_weighted = 0
    correct_any = 0

    for i, (f, item) in enumerate(futures):
        result = f.result()

        # Collect answers with weights
        raw_answers = []
        weighted_answers = defaultdict(float)
        any_correct = False

        for seq in result.sequences:
            text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            answer = extract_answer(text)
            if answer is None:
                continue
            n = _normalize_number(answer)
            key = f"{n:.6f}" if n is not None else answer.strip()
            raw_answers.append(key)
            weighted_answers[key] += compute_weight(text)
            if check_correct(answer, item["ground_truth"]):
                any_correct = True

        # Raw MV
        if raw_answers:
            from collections import Counter
            raw_mv = Counter(raw_answers).most_common(1)[0][0]
            if check_correct(raw_mv, item["ground_truth"]):
                correct_raw += 1

        # Weighted MV
        if weighted_answers:
            weighted_mv = max(weighted_answers, key=weighted_answers.get)
            if check_correct(weighted_mv, item["ground_truth"]):
                correct_weighted += 1

        if any_correct:
            correct_any += 1

        if (i + 1) % 10 == 0:
            n_done = i + 1
            print(f"  [{n_done}/{len(eval_prompts)}] raw={correct_raw/n_done:.1%} "
                  f"weighted={correct_weighted/n_done:.1%} any={correct_any/n_done:.1%}")

    n = len(eval_prompts)
    print(f"\n{'='*60}")
    print(f"WEIGHTED MV ({num_samples} samples, temp={temperature})")
    print(f"{'='*60}")
    print(f"raw_mv: {correct_raw/n:.4f} ({correct_raw}/{n})")
    print(f"weighted_mv: {correct_weighted/n:.4f} ({correct_weighted}/{n})")
    print(f"any_correct: {correct_any/n:.4f} ({correct_any}/{n})")


if __name__ == "__main__":
    main()
