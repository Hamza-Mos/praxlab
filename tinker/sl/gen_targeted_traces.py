"""Generate targeted Claude traces for specific problem types.

Focuses on complex numbers, golden ratio, polynomial remainders,
and Fibonacci-related problems — the types the model fails on.

Usage: python gen_targeted_traces.py [num_per_type]
"""

import json
import os
import re
import sys
import random
from datasets import load_dataset
import anthropic

NUM_PER_TYPE = int(sys.argv[1]) if len(sys.argv) > 1 else 15
MODEL = "claude-sonnet-4-20250514"  # Sonnet for cost efficiency

SYSTEM = """You are solving a math competition problem. Think through it step by step.
Show ALL intermediate steps. Verify your answer. End with \\boxed{answer}."""

# Problem type keywords
TYPE_KEYWORDS = {
    "complex_numbers": ["complex number", "unit circle", "|z|", "z +", "z^", "Re(z)", "Im(z)"],
    "fibonacci_golden": ["fibonacci", "golden ratio", "\\frac{1+\\sqrt{5}}{2}", "F_n", "recurrence"],
    "polynomial_remainder": ["remainder", "divided by", "polynomial", "p(x)", "monic"],
}


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
                    spans.append(text[idx+7:j])
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


def matches_type(problem_text, keywords):
    text_lower = problem_text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def main():
    client = anthropic.Anthropic()

    # Load existing prompts to exclude
    existing = set()
    if os.path.exists("data.jsonl"):
        with open("data.jsonl") as f:
            for line in f:
                item = json.loads(line.strip())
                existing.add(re.sub(r'\s+', ' ', item["prompt"].strip()))
    for path in ["../../tinker/rl/eval_prompts.jsonl", "../../tinker/rl/prompts.jsonl"]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    item = json.loads(line.strip())
                    existing.add(re.sub(r'\s+', ' ', item["prompt"].strip()))

    # Load MATH problems filtered by type
    subjects = ['intermediate_algebra', 'algebra', 'number_theory', 'precalculus',
                'counting_and_probability']
    type_problems = {t: [] for t in TYPE_KEYWORDS}

    for subj in subjects:
        ds = load_dataset('EleutherAI/hendrycks_math', subj, split='train')
        for ex in ds:
            try:
                level = int(ex['level'].replace('Level ', ''))
            except ValueError:
                continue
            if level < 3:
                continue
            pn = re.sub(r'\s+', ' ', ex['problem'].strip())
            if pn in existing:
                continue
            boxed = find_boxed(ex['solution'])
            if not boxed:
                continue
            for t, kws in TYPE_KEYWORDS.items():
                if matches_type(ex['problem'] + ' ' + ex['solution'], kws):
                    type_problems[t].append({
                        "problem": ex["problem"],
                        "level": level,
                        "ground_truth": boxed[-1],
                    })

    for t, probs in type_problems.items():
        print(f"Type '{t}': {len(probs)} problems found", flush=True)

    # Generate traces for each type
    all_traces = []
    for type_name, problems in type_problems.items():
        random.seed(789 + hash(type_name))
        random.shuffle(problems)
        selected = problems[:int(NUM_PER_TYPE * 2)]
        traces = []
        wrong = 0

        for i, item in enumerate(selected):
            if len(traces) >= NUM_PER_TYPE:
                break
            try:
                resp = client.messages.create(
                    model=MODEL, max_tokens=4096, system=SYSTEM,
                    messages=[{"role": "user", "content": item["problem"]}],
                )
                text = resp.content[0].text
                boxed = find_boxed(text)
                if boxed and answers_match(boxed[-1], item["ground_truth"]):
                    formatted = f"<think>\n{text}\n</think>\n\n\\boxed{{{boxed[-1]}}}"
                    traces.append({"prompt": item["problem"], "response": formatted})
                else:
                    wrong += 1
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)

        print(f"  {type_name}: {len(traces)} verified, {wrong} wrong", flush=True)
        all_traces.extend(traces)

    # Save to separate file
    with open("targeted_traces.jsonl", "w") as f:
        for item in all_traces:
            f.write(json.dumps(item) + "\n")

    print(f"\nTotal: {len(all_traces)} targeted traces saved to targeted_traces.jsonl")


if __name__ == "__main__":
    main()
