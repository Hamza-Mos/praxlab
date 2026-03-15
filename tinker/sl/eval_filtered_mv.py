"""Filtered MV: reject low-quality samples before majority voting.

Filters: must have \\boxed{}, reasoning > 200 chars, contains verification.

Usage: python eval_filtered_mv.py <sampler_path> [num_samples] [temp]
"""
import json, re, sys
from collections import Counter
import tinker
from tinker import types
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
MAX_TOKENS = 4096
EVAL_PROMPTS_PATH = "../../tinker/rl/eval_prompts.jsonl"

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _norm(s):
    s = s.strip().replace(",","").replace(" ","")
    try: return float(s)
    except: return None

def extract(text):
    b = _BOXED_RE.findall(text)
    if b: return b[-1].strip()
    return None  # STRICT: only boxed answers

def check(ans, gt):
    if ans is None: return False
    e,p = _norm(gt), _norm(ans)
    if e is not None and p is not None: return abs(p-e)<1e-6
    return ans.strip()==gt.strip()

def quality_filter(text):
    """Return True if sample passes quality filter."""
    if not _BOXED_RE.search(text):
        return False
    if len(text) < 200:
        return False
    return True

def main():
    path = sys.argv[1]
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    temp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3

    prompts = []
    with open(EVAL_PROMPTS_PATH) as f:
        for l in f:
            if l.strip(): prompts.append(json.loads(l))

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    sc = tinker.ServiceClient()
    samp = sc.create_sampling_client(base_model=MODEL, model_path=path)
    stops = [tok.eos_token] if tok.eos_token else []
    for s in ["<|im_end|>","<|eot_id|>","</s>"]:
        if s not in stops: stops.append(s)
    sp = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=temp, stop=stops)

    futures = []
    for item in prompts:
        msgs = [{"role":"user","content":item["prompt"]}]
        toks_list = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        mi = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=toks_list)])
        f = samp.sample(prompt=mi, num_samples=num, sampling_params=sp)
        futures.append((f, item))

    correct_raw = 0
    correct_filt = 0
    correct_any = 0
    total_kept = 0
    total_rejected = 0

    for i, (f, item) in enumerate(futures):
        r = f.result()
        raw_answers = []
        filt_answers = []
        any_ok = False

        for seq in r.sequences:
            text = tok.decode(seq.tokens, skip_special_tokens=True)
            ans = extract(text)

            # Raw (all boxed answers)
            if ans:
                n = _norm(ans)
                key = f"{n:.6f}" if n is not None else ans.strip()
                raw_answers.append(key)

                # Filtered (quality check)
                if quality_filter(text):
                    filt_answers.append(key)
                    total_kept += 1
                else:
                    total_rejected += 1

            if check(ans, item["ground_truth"]):
                any_ok = True

        # Raw MV
        if raw_answers:
            mv = Counter(raw_answers).most_common(1)[0][0]
            if check(mv, item["ground_truth"]): correct_raw += 1

        # Filtered MV
        if filt_answers:
            mv = Counter(filt_answers).most_common(1)[0][0]
            if check(mv, item["ground_truth"]): correct_filt += 1
        elif raw_answers:
            # Fallback to raw if all filtered
            mv = Counter(raw_answers).most_common(1)[0][0]
            if check(mv, item["ground_truth"]): correct_filt += 1

        if any_ok: correct_any += 1
        if (i+1)%10==0:
            print(f"  [{i+1}/50] raw={correct_raw/(i+1):.1%} filt={correct_filt/(i+1):.1%}", flush=True)

    n = len(prompts)
    print(f"\nraw_mv: {correct_raw/n:.4f} ({correct_raw}/{n})")
    print(f"filtered_mv: {correct_filt/n:.4f} ({correct_filt}/{n})")
    print(f"any_correct: {correct_any/n:.4f} ({correct_any}/{n})")
    print(f"kept: {total_kept}, rejected: {total_rejected} ({total_rejected/(total_kept+total_rejected)*100:.0f}% rejected)")

if __name__=="__main__": main()
