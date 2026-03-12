# posttrainer

Autonomous training harness inspired by Karpathy's autoresearch. Tree of self-contained workspaces — each with a `program.md` the user customizes, mutable training scripts, and experiment tracking. Agent-agnostic (works with any agent runtime). Supports both pre-training (from scratch) and post-training (fine-tuning).

## Structure
- `pretrain/` — Pre-training from scratch (autoresearch infrastructure, Modal GPU, BPB metric)
- `tinker/rl/` — GRPO/RL with Tinker SDK (train.py + reward.py + prompts)
- `tinker/sl/` — SFT with Tinker SDK (train.py + data.jsonl)
- `prime/` — Prime Intellect hosted RL (environment building + iteration)
- `lab` — Structured experiment tracking CLI (SQLite-backed, agent-agnostic)
- `data/` — Experiment database (experiments.db, log.jsonl) — gitignored
- `rules.md` — Universal hard rules from 70+ experiments (22 rules)

## How it works
User edits `program.md` Section 1 in a leaf directory with their task description. Agent reads it, builds the setup, then loops forever: hypothesize with mechanism → modify one thing → run → evaluate → confirm/refute mechanism → keep/discard → synthesize every 3-5 experiments. The `lab` CLI provides structured experiment memory across sessions. Each leaf has a `campaign.yaml` (metric config) and `run.sh` (one-command agent launcher).

## Context
Research and analysis stored in project memory files:
- `memory/MEMORY.md` — Index and overview
- `memory/design-plan.md` — Canonical design plan
- `memory/lessons-learned.md` — 20 hard-won lessons from real training runs
- `memory/autoresearch-analysis.md` — How autoresearch works
- `memory/tinker-cookbook-analysis.md` — How tinker-cookbook works (Tinker SDK direct usage)
- `memory/tinkerer-analysis.md` — How tinkerer works (Tinker API + MCP)
- `memory/salesbench-analysis.md` — How salesbench-prime works (Prime Intellect)
- `memory/blog-analysis.md` — Blog post analysis
- `memory/state-of-the-art.md` — RL post-training research landscape
