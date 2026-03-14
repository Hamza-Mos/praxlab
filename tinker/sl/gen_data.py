"""Generate reasoning trace data from MATH dataset for SFT training.

Loads MATH level 2-5 problems, excludes RL eval problems to prevent leakage,
and writes data.jsonl with step-by-step solutions.

Run: python gen_data.py
"""

import json
import random
import re
from datasets import load_dataset

# Config
TARGET_TOTAL = 800
LEVEL_23_FRAC = 0.4  # 40% level 2-3, 60% level 4-5
SEED = 42

# Load eval problems to exclude (prevent data leakage)
eval_problems = set()
with open("../../tinker/rl/eval_prompts.jsonl") as f:
    for line in f:
        item = json.loads(line.strip())
        # Normalize whitespace for matching
        eval_problems.add(re.sub(r'\s+', ' ', item["prompt"].strip()))

# Also exclude RL training prompts
with open("../../tinker/rl/prompts.jsonl") as f:
    for line in f:
        item = json.loads(line.strip())
        eval_problems.add(re.sub(r'\s+', ' ', item["prompt"].strip()))

print(f"Excluding {len(eval_problems)} RL problems from training data")

# Load all MATH subjects
subjects = ['algebra', 'counting_and_probability', 'geometry',
            'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

level_23 = []  # Level 2-3 problems
level_45 = []  # Level 4-5 problems

for subj in subjects:
    ds = load_dataset('EleutherAI/hendrycks_math', subj, split='train')
    for ex in ds:
        try:
            level_num = int(ex['level'].replace('Level ', ''))
        except ValueError:
            continue
        problem_norm = re.sub(r'\s+', ' ', ex['problem'].strip())

        # Skip eval/RL problems
        if problem_norm in eval_problems:
            continue

        # Skip level 1 (too easy) and problems without \boxed{}
        if level_num < 2 or '\\boxed' not in ex['solution']:
            continue

        item = {"prompt": ex['problem'], "response": ex['solution']}

        if level_num <= 3:
            level_23.append(item)
        else:
            level_45.append(item)

print(f"Available: {len(level_23)} level 2-3, {len(level_45)} level 4-5")

# Sample according to mix
random.seed(SEED)
n_23 = int(TARGET_TOTAL * LEVEL_23_FRAC)
n_45 = TARGET_TOTAL - n_23

sampled_23 = random.sample(level_23, min(n_23, len(level_23)))
sampled_45 = random.sample(level_45, min(n_45, len(level_45)))

all_data = sampled_23 + sampled_45
random.shuffle(all_data)

print(f"Selected: {len(sampled_23)} level 2-3, {len(sampled_45)} level 4-5")
print(f"Total: {len(all_data)} examples")

# Write data.jsonl
with open("data.jsonl", "w") as f:
    for item in all_data:
        f.write(json.dumps(item) + "\n")

print(f"Wrote {len(all_data)} examples to data.jsonl")

# Show a few examples
print("\n--- Sample examples ---")
for item in all_data[:3]:
    print(f"PROMPT: {item['prompt'][:150]}")
    print(f"RESPONSE: {item['response'][:200]}")
    print("---")
