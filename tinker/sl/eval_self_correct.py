"""Self-correction: generate answer, then ask model to verify and correct.

For each problem:
1. Generate initial answer at temp=0 (greedy)
2. Feed back: "My answer is X. Is this correct? Solve again carefully."
3. Take the corrected answer
4. MV over corrected answers from multiple samples

Usage: python eval_self_correct.py <sampler_path> [num_samples]
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
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 5

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

    # Step 1: Initial greedy solve
    sp0 = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, stop=stops)
    init_futures = []
    for item in prompts:
        msgs = [{"role":"user","content":item["prompt"]}]
        toks_list = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        mi = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=toks_list)])
        f = samp.sample(prompt=mi, num_samples=1, sampling_params=sp0)
        init_futures.append((f, item))

    # Collect initial answers
    initial_answers = []
    for f, item in init_futures:
        r = f.result()
        text = tok.decode(r.sequences[0].tokens, skip_special_tokens=True)
        ans = extract(text)
        initial_answers.append(ans or "unknown")

    # Step 2: Self-correction — generate N corrected samples
    sp1 = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.5, stop=stops)
    corr_futures = []
    for idx, item in enumerate(prompts):
        correction_prompt = (
            f"{item['prompt']}\n\n"
            f"A student's answer was: {initial_answers[idx]}. "
            f"Carefully solve this problem from scratch. Show all work. "
            f"End with \\boxed{{answer}}."
        )
        msgs = [{"role":"user","content":correction_prompt}]
        toks_list = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        mi = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=toks_list)])
        f = samp.sample(prompt=mi, num_samples=num, sampling_params=sp1)
        corr_futures.append((f, item))

    correct_init = 0
    correct_corr = 0
    correct_any = 0
    for i, (f, item) in enumerate(corr_futures):
        # Initial
        if check(initial_answers[i], item["ground_truth"]):
            correct_init += 1

        # Corrected MV
        r = f.result()
        answers = []
        any_ok = False
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
            if check(mv, item["ground_truth"]): correct_corr += 1
        if any_ok: correct_any += 1

        if (i+1)%10==0:
            print(f"  [{i+1}/50] init={correct_init/(i+1):.1%} corr={correct_corr/(i+1):.1%}", flush=True)

    n = len(prompts)
    print(f"\ninitial_greedy: {correct_init/n:.4f} ({correct_init}/{n})")
    print(f"corrected_mv5: {correct_corr/n:.4f} ({correct_corr}/{n})")
    print(f"any_correct: {correct_any/n:.4f} ({correct_any}/{n})")

if __name__=="__main__": main()
