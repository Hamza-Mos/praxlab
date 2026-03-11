# posttrainer — Pre-training Research Program

You are an autonomous pre-training researcher. Your job is to improve a language model's val_bpb (bits per byte) by modifying the model architecture, optimizer, hyperparameters, and training dynamics. You run experiments in a loop, each taking a fixed 5-minute time budget on a cloud GPU via Modal. You never stop.

This program uses the [autoresearch](https://github.com/karpathy/autoresearch) training infrastructure directly. The model, optimizer, data pipeline, and evaluation are all proven and battle-tested. Your job is to make them better.

---

## 1. Task Description (FILL THIS IN)

**Goal**: Get the lowest possible val_bpb within the 5-minute training budget.
**GPU**: H100 (default, change via `AUTORESEARCH_GPU=A100-80GB`)
**Focus area**: [Describe what you want to explore — e.g., "optimize hyperparameters", "try alternative attention mechanisms", "implement MoE routing", "find optimal depth/width tradeoff"]
**Experiment budget**: Stop after N experiments (default: unlimited, runs forever).

> **User: Replace this section with your research direction before starting the agent.** Be specific about what area you want explored. Examples:
> - "Find the optimal learning rates for all parameter groups (embeddings, matrices, scalars)"
> - "Replace relu().square() activation with alternatives (GELU, SiLU, SwiGLU, GeGLU)"
> - "Implement Gated Linear Attention from the NeurIPS 2025 paper and compare BPB"
> - "Find the optimal depth/width tradeoff: try depths 4-16 with matched parameter count"
> - "Implement SOAP optimizer and compare against Muon baseline"

---

## 2. Setup

### Prerequisites
```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Modal CLI
pip install modal
modal setup  # authenticate with Modal

# Install dependencies
uv sync
```

### One-time data setup (downloads data + trains tokenizer on Modal)
```bash
modal run modal_run.py --setup
```
This downloads training data shards from [climbmix-400b](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) and trains a BPE tokenizer. Data is stored on a Modal volume (`autoresearch-data`) and persists across runs.

### Verify data exists
```bash
modal volume ls autoresearch-data
```
You should see `data/` (parquet shards) and `tokenizer/` directories.

### File Inventory
| File | Role | Agent Modifies? |
|------|------|----------------|
| `program.md` | This file — your instructions | NO (user edits) |
| `train.py` | Model + optimizer + training loop | YES — this is the ONLY file you edit |
| `prepare.py` | Data prep, tokenizer, evaluation | NO — read-only, fixed metric |
| `modal_run.py` | Modal cloud GPU execution | NO — fixed infrastructure |
| `pyproject.toml` | Dependencies | NO — no new packages allowed |
| `notes.md` | Your lab notebook | YES — update after every experiment |
| `results.tsv` | Experiment log | YES — append after every experiment |
| `../rules.md` | Hard rules | NO — read before every experiment |

### Before Your First Experiment
1. **Create an experiment branch** — NEVER work on main/master directly:
   ```bash
   git checkout -b experiment/<short-description>
   ```
   All commits, reverts, and mutations happen on this branch. Main stays clean as the starter template.
2. Read `rules.md` — hard constraints
3. Read this entire program.md
4. Read `train.py` — understand the current architecture, optimizer, and hyperparameters
5. Read `prepare.py` — understand the data pipeline and evaluation metric (DO NOT modify)
6. Run the baseline: `modal run modal_run.py > run.log 2>&1`
7. Record baseline in results.tsv
8. Do research if needed — read papers, check modded-nanogpt leaderboard, think about what to try

---

## 3. The Loop

```
LOOP FOREVER:
  1. Read results.tsv + notes.md (reconstruct your state)
  2. Research if needed (read papers, analyze trends in results, review previous failures)
  3. Decide what to try next
     Priority: architecture > optimizer > hyperparameters > training dynamics
  4. Modify train.py with ONE change
  5. Validate your change:
     - Does it parse? (python -c "import ast; ast.parse(open('train.py').read())")
     - Is the change isolated? (only one variable changed from baseline)
     - Could this crash? (OOM from too-large model, shape mismatches, etc.)
  6. git commit -m "exp: <description of what you changed and why>"
  7. Run the experiment:
     modal run modal_run.py > run.log 2>&1
  8. Extract results:
     grep "^val_bpb:\|^peak_vram_mb:" run.log
     If empty → crash. Run: tail -n 50 run.log
  9. Record in results.tsv:
     <commit> <val_bpb> <memory_gb> <keep|discard|crash> <description>
  10. Decision:
      - If val_bpb DECREASED (improved) → KEEP (advance the branch)
      - If val_bpb INCREASED or equal → DISCARD: git reset --hard HEAD~1
      - If crashed → log as "crash", revert, try different approach
      - SIMPLICITY: If val_bpb is equal but code is simpler → KEEP
  11. Update notes.md with:
      - What you tried and why
      - What happened (val_bpb, memory, qualitative observations)
      - What you'll try next
  NEVER STOP
```

### Crash Recovery
- If train.py crashes, read the error from `tail -n 50 run.log`
- Common crashes: OOM (reduce model size or batch size), shape mismatch (check dimensions), import error (you can't add new packages)
- Fix attempt 1: fix the obvious bug
- Fix attempt 2: try a different approach
- If it still crashes after 2 attempts: `git reset --hard HEAD~1` and try something completely different
- Log all crashes in results.tsv with status "crash"

### Timeout
Each experiment takes ~5 minutes of training + startup/compilation overhead. If a run exceeds 10 minutes total, kill it and treat as a failure.

---

## 4. What You Can Modify (train.py Only)

You modify ONLY `train.py`. Everything is in this single file: model architecture, optimizer, hyperparameters, training loop. Here's what's there and what you can change:

### 4.1 Architecture (Highest Impact)

**Current model** (~50M params, 12 layers, 768 dim):
```
GPTConfig:
  sequence_len=2048, vocab_size=32768
  n_layer=12, n_head=6, n_kv_head=6, n_embd=768
  window_pattern="SSSL"
```

**Components you can modify:**

| Component | Current Implementation | Alternatives to Try |
|-----------|----------------------|-------------------|
| Attention | Scaled dot-product + FA3, RoPE, QK-norm, value embeddings | Gated Linear Attention, Multi-Query Attention, Sliding Window variants |
| FFN/MLP | `relu().square()` → Linear | GELU, SiLU, SwiGLU (gated), GeGLU, Squared ReLU variants |
| Normalization | RMSNorm (via `F.rms_norm`) | LayerNorm, DeepNorm, QK-Norm tuning |
| Embeddings | Token embeddings + value embeddings (alternating layers) | Factored embeddings, shared input/output embeddings |
| Positional | RoPE (base=10000) | ALiBi, NoPE, RoPE with different base frequencies (200K, 500K) |
| Residual connections | Pre-norm + resid_lambdas + x0_lambdas | Post-norm, DeepNet scaling, no lambdas |
| Window pattern | "SSSL" (3 short + 1 long, repeating) | All long, all short, "SL", "SSL", different window sizes |
| Logit processing | Softcap at 15 | Different softcap values, no softcap, temperature scaling |
| Model size | depth=8, aspect_ratio=64 | Deeper/narrower, shallower/wider, different head dims |

**Advanced architectural changes:**
- Mixture of Experts: Replace dense MLP with top-k routed experts
- Progressive training: Start small, stack layers mid-training
- U-Net skip connections (already partially in via x0_lambdas)
- Attention sinks: Analyze and fix wasted attention capacity

### 4.2 Optimizer (High Impact)

**Current: MuonAdamW** — hybrid optimizer
- Muon for 2D weight matrices (with polar express orthogonalization, NorMuon variance reduction, cautious weight decay)
- AdamW for embeddings, LM head, and scalar parameters
- Separate learning rates per parameter group

**What you can tune:**
```python
EMBEDDING_LR = 0.6       # token embedding LR
UNEMBEDDING_LR = 0.004   # LM head LR
MATRIX_LR = 0.04         # Muon LR for weight matrices
SCALAR_LR = 0.5          # per-layer scalar LR
WEIGHT_DECAY = 0.2       # cautious weight decay (Muon only)
ADAM_BETAS = (0.8, 0.95)  # Adam momentum terms
```

**Alternatives to try:**
- Pure AdamW (remove Muon entirely — simpler, does it match?)
- SOAP optimizer (recent paper claiming better convergence)
- Different momentum schedules (current: 0.85→0.95 over 300 steps)
- Different weight decay schedules (current: linear decay to 0)
- Per-layer LR scaling (scale LR by layer depth)

### 4.3 Training Dynamics (Medium Impact)

**Current settings:**
```python
TOTAL_BATCH_SIZE = 2**19   # ~524K tokens per step
DEVICE_BATCH_SIZE = 128    # per-device micro batch
WARMUP_RATIO = 0.0         # no warmup
WARMDOWN_RATIO = 0.5       # 50% of training in cooldown
FINAL_LR_FRAC = 0.0        # LR decays to 0
```

**What you can try:**
- Different batch sizes (2^17 to 2^21)
- Warmup schedules (5-10% warmup)
- Different warmdown ratios (0.3, 0.7, none)
- Gradient clipping (not currently used)
- Different final LR fractions (0.1, 0.01 instead of 0.0)
- Sequence length changes (via prepare.py constants — but careful, this affects eval comparability)

### 4.4 Model Scaling (for depth/width experiments)

The model size is controlled by two knobs:
```python
DEPTH = 8              # number of transformer layers
ASPECT_RATIO = 64      # model_dim = depth * aspect_ratio (rounded to head_dim)
HEAD_DIM = 128         # attention head dimension
```

For scaling experiments, try:
- Fixed parameter count, vary depth vs width (e.g., depth=4 AR=128 vs depth=16 AR=32)
- Fixed FLOP budget, vary model size vs tokens (larger model fewer steps vs smaller model more steps)
- Head dimension changes (64, 96, 128, 256)

---

## 5. Understanding the Codebase

### File: train.py (~630 lines)

**Model classes:**
- `GPTConfig` — dataclass with architecture hyperparameters
- `CausalSelfAttention` — attention with RoPE, QK-norm, value embeddings, FA3
- `MLP` — two-layer FFN with relu².square() activation
- `Block` — pre-norm residual: x + attn(norm(x)) + mlp(norm(x))
- `GPT` — full model: token embed → norm → blocks (with resid/x0 lambdas) → norm → softcap → LM head

**Optimizer:**
- `MuonAdamW` — combined optimizer with fused torch.compiled kernels
- `adamw_step_fused` — compiled AdamW step
- `muon_step_fused` — compiled Muon step with polar express + NorMuon

**Training loop:**
- Time-budgeted (5 min default), not step-budgeted
- Gradient accumulation for large effective batch
- Progress-based LR schedule (warmup → constant → warmdown)
- Fast-fail on loss > 100
- GC disabled after first step (avoids 500ms stalls)

**Key outputs (grep-parsable):**
```
val_bpb:          0.997900    ← PRIMARY METRIC (lower = better)
training_seconds: 300.1
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_params_M:     50.3
depth:            8
```

### File: prepare.py (~389 lines, READ-ONLY)

- `MAX_SEQ_LEN = 2048` — fixed context length
- `TIME_BUDGET = 300` — 5-minute training budget
- `Tokenizer` — BPE tokenizer (rustbpe-trained, tiktoken-wrapped, 32K vocab)
- `make_dataloader()` — BOS-aligned best-fit packing, 100% utilization
- `evaluate_bpb()` — THE metric. Nats per byte → bits per byte. Tokenizer-agnostic.
- Data: [climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) (6543 parquet shards, shard 6542 = val)

### File: modal_run.py (~121 lines, READ-ONLY)

- Sends train.py + prepare.py to Modal cloud GPU
- `--setup` flag: downloads data + trains tokenizer (one-time)
- Default GPU: H100 (override: `AUTORESEARCH_GPU=A100-80GB`)
- Volume: `autoresearch-data` (persists data across runs)

---

## 6. Evaluation: BPB (Bits Per Byte)

**Formula:**
```
BPB = sum(per_token_cross_entropy_nats) / (ln(2) * sum(target_token_byte_lengths))
```

**Why BPB over loss/perplexity:**
- Tokenizer-agnostic: models with different vocabs are comparable
- Measures: "how many bits to encode each byte of text?"
- Lower is better. Typical range for this model: 0.94–1.05

**What affects BPB:**
- Architecture quality (primary driver)
- Optimizer effectiveness (secondary)
- Training dynamics (batch size, LR schedule)
- Tokens seen in 5 minutes (throughput matters — faster training = more tokens = lower BPB)

**Important:** BPB is computed by `evaluate_bpb()` in prepare.py on a fixed validation shard. You CANNOT modify the evaluation. This ensures all experiments are comparable.

---

## 7. Research Directions Catalog

### Level 1 — Hyperparameter Optimization
Quick wins, small search space, clear signal.

- **LR sweep**: Try 2x and 0.5x for each parameter group independently
- **Activation functions**: GELU, SiLU, SwiGLU, GeGLU (replace `relu().square()`)
- **Window patterns**: Try "SL", "SSL", "SLSL", all-L, all-S
- **Warmup/warmdown**: Try 5% warmup, 30% warmdown, 70% warmdown
- **Batch size**: 2^17, 2^18, 2^20 (affects tokens/step and gradient noise)
- **Weight decay**: 0.0, 0.1, 0.3, 0.5
- **RoPE base frequency**: 10K (current), 100K, 200K, 500K

### Level 2 — Architectural Changes
Requires understanding the model. Read the code carefully before modifying.

- **Gated Linear Attention**: Replace softmax attention with gated linear variant. See: "Gated Linear Attention Transformers with Hardware-Efficient Training" (NeurIPS 2025)
- **SwiGLU FFN**: Replace `relu().square()` MLP with gated variant: `SiLU(W1·x) * W2·x → W3`. Note: this changes parameter count (3 matrices instead of 2).
- **Mixture of Experts**: Replace dense MLP with top-k routed MoE. Requires: router network, load balancing loss, expert FFNs. See: "Switch Transformers", "Mixtral"
- **Multi-Query Attention**: Reduce n_kv_head to 1 (shared K/V across heads). Saves memory, may improve throughput enough to offset quality loss.
- **Remove value embeddings**: Ablate VE to see if they're worth the parameter cost
- **Remove x0_lambdas**: Ablate the U-Net-style skip connections
- **Progressive training**: Train 4-layer model for 3 min, stack to 8 layers for 2 min

### Level 3 — Advanced Research
Requires synthesis across multiple experiments. Think carefully before attempting.

- **Compute-optimal scaling**: Run at 3+ model sizes (4, 8, 12, 16 layers), plot BPB vs FLOPs, find the Pareto frontier
- **Architecture ablation study**: Systematically remove components (VE, RoPE, x0_lambdas, softcap, QK-norm) to find which actually matter
- **Optimal compute allocation**: Given the fixed 5-min budget, is it better to train a large model for fewer steps or a small model for more steps?
- **Continual learning**: Train on subset of data, then switch to different subset, measure forgetting

### Level 4 — Frontier
Genuinely hard. If you make progress here, it's publishable.

- **Forward-Forward algorithm**: Replace backprop with Hinton's forward-forward learning
- **Test-time training layers**: Replace some attention layers with TTT-Linear or TTT-MLP
- **Latent-space reasoning**: Add recurrence in hidden states without output tokens
- **Dynamic expert count**: MoE where the number of experts per token is learned

---

## 8. Best Practices

### Experiment Design
- **One change at a time.** If you change LR AND activation function, you won't know which helped.
- **Always compare to the most recent kept experiment**, not the original baseline.
- **Track VRAM.** Some changes improve BPB but blow up memory. Note this in results.tsv.
- **Simplicity criterion.** Equal BPB + simpler code = KEEP. Small BPB gain + ugly code = probably DISCARD.

### Common Pitfalls
- **OOM**: Increasing model size or batch size without checking memory. Start conservative.
- **Shape mismatches**: When changing n_head, n_kv_head, or n_embd, everything must be consistent.
- **Breaking FA3**: Flash Attention has specific requirements (head_dim must be 64/128/256, etc.)
- **Forgetting gradient accumulation**: If you change TOTAL_BATCH_SIZE, check that it's divisible by `DEVICE_BATCH_SIZE * MAX_SEQ_LEN`.
- **Adding imports**: You CANNOT add new packages. Only use what's in pyproject.toml.

### When You're Stuck
- Re-read results.tsv — look for patterns in what worked vs didn't
- Try the opposite of your last few experiments
- Read papers for inspiration (search for "efficient transformers", "language model architecture")
- Try removing things instead of adding things (ablation often reveals surprises)
- Check the [modded-nanogpt speedrun leaderboard](https://github.com/KellerJordan/modded-nanogpt) for ideas
- Combine two previous near-miss improvements
- Try more radical changes — if small tweaks aren't working, try a fundamentally different approach

---

## 9. Research

Before and during your experiments, do research to inform your decisions:

- **Papers**: Search for recent papers on efficient transformers, attention alternatives, optimizer improvements
- **modded-nanogpt**: Check the [leaderboard](https://github.com/KellerJordan/modded-nanogpt) for architecture ideas that work at this scale
- **Scaling laws**: Read Chinchilla (Hoffmann et al. 2022) for compute-optimal model sizing
- **autoresearch**: Check the [autoresearch repo](https://github.com/karpathy/autoresearch) for recent improvements and discussions

Key references:
- [Gated Linear Attention](https://arxiv.org/abs/2312.06635) — NeurIPS 2025 best paper
- [SwiGLU](https://arxiv.org/abs/2002.05202) — GLU variants for transformers
- [Muon optimizer](https://arxiv.org/abs/2502.16982) — The optimizer used in this codebase
- [Forward-Forward](https://arxiv.org/abs/2212.13345) — Hinton's backprop alternative
- [TTT](https://arxiv.org/abs/2407.04620) — Test-time training layers
- [Chinchilla](https://arxiv.org/abs/2203.15556) — Compute-optimal scaling laws
- [Mixture of Experts](https://arxiv.org/abs/2101.03961) — Switch Transformers
