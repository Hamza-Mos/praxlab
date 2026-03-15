"""Mixed-temperature MV: combine low-temp (accurate) + high-temp (diverse) samples.

Usage: python eval_mixed_temp.py <sampler_path>
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
    n = _NUM_RE.findall(text.replace(",",""))
    if n: return n[-1].strip()
    return None

def check(ans, gt):
    if ans is None: return False
    e,p = _norm(gt), _norm(ans)
    if e is not None and p is not None: return abs(p-e)<1e-6
    return ans.strip()==gt.strip()

def main():
    path = sys.argv[1]
    eval_prompts = []
    with open(EVAL_PROMPTS_PATH) as f:
        for l in f:
            if l.strip(): eval_prompts.append(json.loads(l))

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    sc = tinker.ServiceClient()
    samp = sc.create_sampling_client(base_model=MODEL, model_path=path)
    stops = [tok.eos_token] if tok.eos_token else []
    for s in ["<|im_end|>","<|eot_id|>","</s>"]:
        if s not in stops: stops.append(s)

    # Two temperature configs
    configs = [(96, 0.1), (32, 0.7)]  # (num_samples, temp)

    futures_all = []
    for item in eval_prompts:
        msgs = [{"role":"user","content":item["prompt"]}]
        toks = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        mi = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=toks)])
        item_futures = []
        for n, t in configs:
            sp = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=t, stop=stops)
            f = samp.sample(prompt=mi, num_samples=n, sampling_params=sp)
            item_futures.append(f)
        futures_all.append((item_futures, item))

    correct_mixed = 0
    correct_any = 0
    for i, (futs, item) in enumerate(futures_all):
        answers = []
        any_ok = False
        for f in futs:
            r = f.result()
            for seq in r.sequences:
                text = tok.decode(seq.tokens, skip_special_tokens=True)
                a = extract(text)
                if a:
                    n = _norm(a)
                    answers.append(f"{n:.6f}" if n is not None else a.strip())
                if check(a, item["ground_truth"]):
                    any_ok = True
        if answers:
            mv = Counter(answers).most_common(1)[0][0]
            if check(mv, item["ground_truth"]): correct_mixed += 1
        if any_ok: correct_any += 1
        if (i+1)%10==0:
            print(f"  [{i+1}/50] mixed={correct_mixed/(i+1):.1%} any={correct_any/(i+1):.1%}",flush=True)

    n = len(eval_prompts)
    print(f"\nmixed_mv: {correct_mixed/n:.4f} ({correct_mixed}/{n})")
    print(f"any_correct: {correct_any/n:.4f} ({correct_any}/{n})")

if __name__=="__main__": main()
