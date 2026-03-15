"""Oracle reranking: use Claude to select the best answer from N samples.

Generates N completions per problem, groups by answer, then asks Claude
to judge which answer group has the best reasoning. This exploits the
fact that any_correct@64=100% — the right answer IS in the samples.

Usage: python eval_oracle_rerank.py <sampler_path> [num_samples]
"""

import json
import re
import sys
from collections import Counter, defaultdict
import anthropic
import tinker
from tinker import types
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
MAX_TOKENS = 4096
TEMPERATURE = 0.5
EVAL_PROMPTS_PATH = "../../tinker/rl/eval_prompts.jsonl"

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _normalize_number(s):
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_answer(text):
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    nums = _NUM_RE.findall(text.replace(",", ""))
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


def rerank_with_claude(problem, answer_groups):
    """Ask Claude to pick the best-reasoned answer group."""
    if len(answer_groups) == 1:
        return list(answer_groups.keys())[0]

    client = anthropic.Anthropic()

    # Build prompt showing top answer groups with example reasoning
    prompt = f"Problem: {problem}\n\nSeveral attempts produced these different answers:\n\n"
    for i, (answer, completions) in enumerate(sorted(answer_groups.items(),
                                                      key=lambda x: -len(x[1]))):
        count = len(completions)
        best = max(completions, key=len)[:800]  # Longest reasoning
        prompt += f"Answer {i+1}: {answer} (appeared {count} times)\n"
        prompt += f"Sample reasoning:\n{best}\n\n"

    prompt += "Which answer is correct? Reply with ONLY the answer value (e.g., '42' or '\\frac{1}{2}'). No explanation."

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        claude_pick = resp.content[0].text.strip()
        # Find which answer group Claude picked
        for answer in answer_groups:
            if check_correct(claude_pick, answer) or claude_pick in answer:
                return answer
        # Fallback: majority vote
        return max(answer_groups, key=lambda x: len(answer_groups[x]))
    except Exception:
        return max(answer_groups, key=lambda x: len(answer_groups[x]))


def main():
    sampler_path = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 32

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

    sp = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, stop=stop_seqs)

    # Generate all samples
    futures = []
    for item in eval_prompts:
        msgs = [{"role": "user", "content": item["prompt"]}]
        toks = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        mi = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=toks)])
        f = sampling_client.sample(prompt=mi, num_samples=num_samples, sampling_params=sp)
        futures.append((f, item))

    correct_mv = 0
    correct_oracle = 0
    correct_any = 0

    for i, (f, item) in enumerate(futures):
        result = f.result()
        answer_groups = defaultdict(list)
        any_correct = False

        for seq in result.sequences:
            text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            answer = extract_answer(text)
            if answer is None:
                continue
            n = _normalize_number(answer)
            key = f"{n:.6f}" if n is not None else answer.strip()
            answer_groups[key].append(text)
            if check_correct(answer, item["ground_truth"]):
                any_correct = True

        # Raw MV
        if answer_groups:
            mv_answer = max(answer_groups, key=lambda x: len(answer_groups[x]))
            if check_correct(mv_answer, item["ground_truth"]):
                correct_mv += 1

        # Oracle reranking (only for problems where MV might be wrong)
        if answer_groups:
            # Only call Claude if there's disagreement
            if len(answer_groups) > 1:
                oracle_answer = rerank_with_claude(item["prompt"], answer_groups)
            else:
                oracle_answer = list(answer_groups.keys())[0]
            if check_correct(oracle_answer, item["ground_truth"]):
                correct_oracle += 1

        if any_correct:
            correct_any += 1

        if (i + 1) % 10 == 0:
            n_done = i + 1
            print(f"  [{n_done}/{len(eval_prompts)}] mv={correct_mv/n_done:.1%} "
                  f"oracle={correct_oracle/n_done:.1%} any={correct_any/n_done:.1%}", flush=True)

    n = len(eval_prompts)
    print(f"\n{'='*60}")
    print(f"ORACLE RERANKING ({num_samples} samples)")
    print(f"{'='*60}")
    print(f"raw_mv: {correct_mv/n:.4f} ({correct_mv}/{n})")
    print(f"oracle_rerank: {correct_oracle/n:.4f} ({correct_oracle}/{n})")
    print(f"any_correct: {correct_any/n:.4f} ({correct_any}/{n})")


if __name__ == "__main__":
    main()
