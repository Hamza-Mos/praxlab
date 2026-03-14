"""Evaluate SFT model accuracy on 50 MATH eval problems.

Three-way comparison: base model vs SFT vs RL (83.13%).

Usage: python eval_accuracy.py <sampler_weights_path>
Example: python eval_accuracy.py 'tinker://...:train:0/sampler_weights/final'
"""

import json
import re
import sys
import tinker
from tinker import types
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
MAX_TOKENS = 4096
TEMPERATURE = 0.0  # Greedy for deterministic eval
EVAL_PROMPTS_PATH = "../../tinker/rl/eval_prompts.jsonl"

# Reuse RL reward function logic
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _normalize_number(s: str) -> float | None:
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def check_correct(completion: str, ground_truth: str) -> bool:
    expected = _normalize_number(ground_truth)
    if expected is None:
        return False
    boxed_matches = _BOXED_RE.findall(completion)
    if boxed_matches:
        predicted = _normalize_number(boxed_matches[-1])
        if predicted is not None and abs(predicted - expected) < 1e-6:
            return True
    # Fallback: last number
    num_matches = _NUM_RE.findall(completion.replace(",", ""))
    if num_matches:
        predicted = _normalize_number(num_matches[-1])
        if predicted is not None and abs(predicted - expected) < 1e-6:
            return True
    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_accuracy.py <sampler_weights_path>")
        sys.exit(1)

    sampler_path = sys.argv[1]
    print(f"Model: {MODEL}")
    print(f"Sampler weights: {sampler_path}")

    # Load eval prompts
    eval_prompts = []
    with open(EVAL_PROMPTS_PATH) as f:
        for line in f:
            if line.strip():
                eval_prompts.append(json.loads(line))
    print(f"Eval problems: {len(eval_prompts)}")

    # Setup
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

    # System prompt (must match training if used)
    system_prompt = sys.argv[2] if len(sys.argv) > 2 else None

    # Submit all sampling requests
    futures = []
    for item in eval_prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": item["prompt"]})
        token_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        model_input = types.ModelInput(
            chunks=[types.EncodedTextChunk(tokens=token_ids)]
        )
        future = sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=sampling_params
        )
        futures.append((future, item))

    # Collect results
    correct = 0
    boxed_count = 0
    samples = []
    for i, (future, item) in enumerate(futures):
        result = future.result()
        text = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
        is_correct = check_correct(text, item["ground_truth"])
        has_boxed = bool(_BOXED_RE.search(text))

        if is_correct:
            correct += 1
        if has_boxed:
            boxed_count += 1

        if len(samples) < 5:
            samples.append((item["prompt"][:100], text[:300], item["ground_truth"], is_correct))

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(eval_prompts)}] running accuracy: {correct/(i+1):.1%}")

    accuracy = correct / len(eval_prompts)
    boxed_rate = boxed_count / len(eval_prompts)

    print(f"\n{'='*60}")
    print(f"RESULTS: SFT Reasoning Trace Distillation")
    print(f"{'='*60}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"correct: {correct}/{len(eval_prompts)}")
    print(f"boxed_rate: {boxed_rate:.2f}")
    print(f"\nThree-way comparison:")
    print(f"  Base Qwen3-8B:     ~15% (estimated)")
    print(f"  SFT (this model):  {accuracy:.1%}")
    print(f"  RL (exp 7):        83.13%")

    print(f"\n--- Sample completions ---")
    for prompt, completion, gt, correct in samples:
        print(f"PROMPT: {prompt}...")
        print(f"COMPLETION: {completion}")
        print(f"GROUND TRUTH: {gt} | CORRECT: {correct}")
        print("---")


if __name__ == "__main__":
    main()
