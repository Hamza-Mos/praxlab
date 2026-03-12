# posttrainer

Autonomous training harness inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Point an AI agent at a task, and it builds, trains, and iterates forever — modifying model architectures, reward functions, training data, and hyperparameters to continuously improve. Supports both pre-training (from scratch) and post-training (fine-tuning existing models).

Each directory is a self-contained workspace with its own `program.md` (agent instructions), mutable training scripts, and experiment tracking. The agent reads `program.md`, builds the setup, then loops: modify one thing → run → evaluate → keep or discard → repeat. It never stops.

## Quick Start

| I want to... | Go here | Run |
|--------------|---------|-----|
| **Pre-train from scratch (architecture research)** | `pretrain/` | `cd pretrain && ./run.sh` |
| **RL (GRPO) with Tinker SDK** | `tinker/rl/` | `cd tinker/rl && ./run.sh` |
| **SFT with Tinker SDK** | `tinker/sl/` | `cd tinker/sl && ./run.sh` |
| **RL with Prime Intellect** | `prime/` | `cd prime && ./run.sh` |

### Step-by-step

1. **Pick a directory** based on your training method and backend
2. **Edit `program.md`** — fill in Section 1 with your task description, model, and cost budget
3. **Set credentials** — `export TINKER_API_KEY=...`, `prime login`, or `modal setup`
4. **Launch:** `cd <directory> && ./run.sh` (default: Claude; also `./run.sh --agent codex` or `./run.sh --agent opencode`)

The agent will:
- Create an experiment branch (main stays clean as the starter template)
- Build the training setup (reward function, data, config) from your task description
- Form hypotheses with causal mechanisms before each experiment
- Run experiments autonomously, confirm/refute mechanisms after each result
- Track structured research memory via `lab` CLI (queryable failures, insights, syntheses)
- Keep improvements, discard failures (via git commits on the experiment branch)
- Synthesize learnings every 3-5 experiments
- Never stop until you tell it to (or it hits your cost budget)

To reset after an experiment: `git checkout main` (or run `./clean.sh` to also remove generated files).

## Architecture

```
posttrainer/
├── README.md              ← you are here
├── lab                    ← structured experiment tracking CLI (SQLite-backed)
├── rules.md               ← hard rules from 70+ real experiments
├── clean.sh               ← reset generated files
├── data/                  ← experiment database (gitignored)
│   ├── experiments.db
│   └── log.jsonl
├── pretrain/              ← pre-training from scratch (autoresearch)
│   ├── program.md         ← agent instructions (edit Section 1)
│   ├── campaign.yaml      ← metric + config for this leaf
│   ├── run.sh             ← one-command agent launcher
│   ├── train.py           ← model + optimizer + training loop (agent modifies)
│   ├── prepare.py         ← data prep + eval (fixed, read-only)
│   ├── modal_run.py       ← Modal cloud GPU execution (fixed)
│   ├── notes.md           ← agent's lab notebook (optional)
│   └── results.tsv        ← experiment log
├── tinker/
│   ├── rl/                ← GRPO with Tinker SDK
│   │   ├── program.md     ← agent instructions (edit Section 1)
│   │   ├── campaign.yaml  ← metric + config
│   │   ├── run.sh         ← agent launcher
│   │   ├── train.py       ← training loop (agent modifies)
│   │   ├── reward.py      ← reward function (agent modifies)
│   │   ├── prompts.jsonl  ← training data (agent modifies)
│   │   ├── eval_prompts.jsonl
│   │   ├── notes.md
│   │   └── results.tsv
│   └── sl/                ← SFT with Tinker SDK
│       ├── program.md
│       ├── campaign.yaml
│       ├── run.sh
│       ├── train.py
│       ├── data.jsonl
│       ├── notes.md
│       └── results.tsv
└── prime/                 ← Prime Intellect hosted RL
    ├── program.md
    ├── campaign.yaml
    ├── run.sh
    ├── notes.md
    └── results.tsv
```

## Tips for Writing a Good program.md

The quality of your `program.md` Section 1 directly determines the quality of results. Here are tips:

### Be specific about the task
Bad: "Make the model better at math"
Good: "Train Qwen3-8B to solve GSM8K-style word problems. The model should show its work step-by-step and put the final numeric answer in \\boxed{}. Correct means the number inside \\boxed{} matches the ground truth."

### Specify your model
For **post-training** (tinker, prime), choose based on your needs:
- **Small + fast iteration**: `Qwen/Qwen3-8B`, `meta-llama/Llama-3.2-3B`
- **Efficient MoE**: `Qwen/Qwen3-30B-A3B` (30B params, only 3B active — great quality/cost)
- **Maximum capability**: `Qwen/Qwen3-235B-A22B`, `deepseek-ai/DeepSeek-V3-0324`

For **pre-training** (`pretrain/`), the model is defined in `train.py` and trained from scratch. Default: ~50M param GPT (12 layers, 768 dim). The agent modifies the architecture directly.

### Set a cost budget
Include a line like "Stop if total cost exceeds $10" to prevent runaway spending. Tinker RL experiments typically cost $0.50-$2.00 each. Prime experiments are currently free during beta. Pre-training experiments on Modal cost ~$0.10-$0.30 per 5-minute run on H100.

### Define what "good" looks like
The agent needs to know what success is. Be explicit about output format, correctness criteria, and quality bar.

### Choose the right approach

| Task type | Best approach | Directory |
|-----------|--------------|-----------|
| Architecture research (attention, FFN, scaling) | Pre-train | `pretrain/` |
| Optimizer research (Muon, SOAP, schedules) | Pre-train | `pretrain/` |
| Verifiable answers (math, code, classification) | GRPO (RL) | `tinker/rl/` |
| Subjective quality (writing, conversation) | SFT | `tinker/sl/` |
| Multi-turn / agentic (tool use, games, dialogue) | Prime RL | `prime/` |
| Complex environments (sandboxed code, browser) | Prime RL | `prime/` |

## Hard Rules

All training is governed by `rules.md` — 22 hard rules derived from 70+ real experiments. The most critical:

1. **NEVER use cosine LR scheduling for RL** — it collapses to zero. Use constant.
2. **Temperature 1.0 for GRPO** — lower temperatures cause model collapse.
3. **batch_size >= 128** — below this, training is too noisy.
4. **One change per experiment** — so you know what caused the effect.
5. **Save checkpoint before changing the reward function** — reward changes reset progress.
6. **Read 3 sample completions per experiment** — guards against reward hacking.

See `rules.md` for the complete list.

## How It Works

This project follows the [autoresearch](https://github.com/karpathy/autoresearch) pattern:

1. **Human writes `program.md`** — strategic decisions (what task, what model, what approach)
2. **Agent executes the loop** — tactical decisions (what hyperparams, what reward tweaks, what data to add)
3. **Git tracks everything** — every experiment is a commit. Improvements are kept. Failures are reverted.
4. **`results.tsv` is the scoreboard** — one number to optimize (val_bpb, eval_reward_mean, or eval_loss)
5. **`lab` CLI is the research memory** — structured hypotheses, mechanism tracking, failure avoidance, periodic synthesis. Queryable across sessions via SQLite.

The key insight from [the blog post](https://hamzamostafa.com/blog/agents-training-their-own-models): agents are good at *execution* within constraints but poor at *judgment*. So we make the human decisions strategic and the agent decisions tactical. The constraints (rules.md, program.md) are what make it work.

## Contributing

This project is in active development. If you add a new backend or training method, create a new directory with its own `program.md`, mutable scripts, and tracking files. Follow the existing pattern.

## References

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — The original autonomous research pattern
- [Tinker SDK](https://tinker-docs.thinkingmachines.ai) — Tinker API documentation
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) — Official training recipes
- [Prime Intellect](https://docs.primeintellect.ai) — Prime Intellect documentation
- [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) — Prime Intellect training framework
- [verifiers](https://github.com/PrimeIntellect-ai/verifiers) — Prime Intellect environment library
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — GRPO for reasoning
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — The speedrun leaderboard for GPT architecture improvements
- [Modal](https://modal.com) — Cloud GPU platform for pre-training experiments
- [Blog: AI Agents Training Models](https://hamzamostafa.com/blog/agents-training-their-own-models) — Lessons from 100+ experiments
