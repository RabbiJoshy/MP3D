import sys, re, pandas as pd
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
MODEL = sys.argv[1]
model, tok = load(MODEL)
k = pd.read_json("claims_clustered.jsonl", lines=True)
s = pd.read_parquet("claim_sample.parquet").set_index("passage_id").text
two = k[k.claim.str.contains("two-child|benefit cap", case=False)]
bad = k[k.claim.str.contains("poorest pensioners|cut £12 billion|Benefits should be cut for", case=False)]
test = pd.concat([two, bad]).drop_duplicates("passage_id")
SYS = ("You analyse excerpts of UK House of Commons speeches. Step 1: name the main specific policy the speaker discusses. "
       "Step 2: decide whether the SPEAKER THEMSELVES wants that policy, or opposes it. Careful: speakers often describe a policy, "
       "or another party's plan, in order to attack it. Sarcasm and criticism of opponents mean the speaker is AGAINST that thing. "
       "Step 3: write the speaker's own position as one plain sentence of at most 15 words. "
       "Answer in exactly this format:\nPOLICY: ...\nSPEAKER: FOR or AGAINST\nPOSITION: ...\nIf there is no policy position, answer POSITION: NONE.")
rows = []
for r in test.itertuples():
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": s[r.passage_id][:2500]}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **({"enable_thinking": False} if "Qwen3-8B" in MODEL else {}))
    o = generate(model, tok, prompt=prompt, max_tokens=80, sampler=make_sampler(temp=0.0))
    pos = re.search(r"POSITION:\s*(.*)", o); pol = re.search(r"POLICY:\s*(.*)", o); sp = re.search(r"SPEAKER:\s*(\w+)", o)
    rows.append({"mp": r.mp, "party": r.party, "old": r.claim, "policy": pol and pol[1].strip(), "side": sp and sp[1], "new": pos and pos[1].strip()})
    print(f"{r.party:5} {r.mp[:20]:20} | {rows[-1]['side']} | {rows[-1]['new']}", flush=True)
pd.DataFrame(rows).to_json(f"claims2_{MODEL.split('/')[-1]}.jsonl", orient="records", lines=True)
