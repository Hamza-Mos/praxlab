"""
Reward function for GRPO training.

CONTRACT:
    compute_reward(completion: str, ground_truth: str) -> float
    - Must return a float (typically 0.0 to 1.0)
    - Must be deterministic
    - Must NEVER crash (catch all exceptions, return 0.0)

EXPERIMENT: Pure binary reward for emergent reasoning (DeepSeek-R1-Zero style).
Extracts \\boxed{N} answers. No partial credit, no format reward — just 1.0 or 0.0.
The model must discover its own reasoning strategy from pure reward signal.
"""

import re

# Match \boxed{...} — content can be a number (integer or decimal, possibly negative)
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
# Fallback: extract last number in the completion
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _normalize_number(s: str) -> float | None:
    """Try to parse a string as a number, stripping whitespace/commas."""
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def compute_reward(completion: str, ground_truth: str) -> float:
    """
    Pure binary reward: 1.0 if correct, 0.0 otherwise.

    Extraction priority:
      1. Last \\boxed{N} in the completion
      2. Fallback: last number in the completion

    Comparison: numeric equality (float comparison with tolerance).
    """
    try:
        expected = _normalize_number(ground_truth)
        if expected is None:
            return 0.0

        # Try \boxed{} first (prefer last match — final answer)
        boxed_matches = _BOXED_RE.findall(completion)
        if boxed_matches:
            predicted = _normalize_number(boxed_matches[-1])
            if predicted is not None and abs(predicted - expected) < 1e-6:
                return 1.0
            return 0.0

        # Fallback: last number in completion
        num_matches = _NUM_RE.findall(completion.replace(",", ""))
        if num_matches:
            predicted = _normalize_number(num_matches[-1])
            if predicted is not None and abs(predicted - expected) < 1e-6:
                return 1.0

        return 0.0
    except Exception:
        return 0.0
