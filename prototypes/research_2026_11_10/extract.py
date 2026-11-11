"""Generic step A: claims from a topic sample. python extract.py Housing"""
import sys, re, time, pandas as pd
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
tag = sys.argv[1]
model, tok = load("mlx-community/Qwen3-4B-Instruct-2507-4bit")
t = pd.read_parquet(f"{tag}_sample.parquet")
SYS = ("You analyse excerpts of UK House of Commons speeches. Step 1: name the main specific policy the speaker discusses. "
       "Step 2: decide whether the SPEAKER THEMSELVES wants that policy, or opposes it. Careful: speakers often describe a policy, "
       "or another party's plan, in order to attack it. Sarcasm and criticism of opponents mean the speaker is AGAINST that thing. "
       "Step 3: write the speaker's own position as one plain sentence of at most 15 words. "
       "Answer in exactly this format:\nPOLICY: ...\nSPEAKER: FOR or AGAINST\nPOSITION: ...\nIf there is no policy position, answer POSITION: NONE.")
rows, t0 = [], time.time()
for i, r in enumerate(t.itertuples()):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": r.text[:2500]}]
    o = generate(model, tok, prompt=tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), max_tokens=80, sampler=make_sampler(temp=0.0))
    g = lambda k: (m := re.search(k + r":\s*(.*)", o)) and m[1].strip()
    rows.append({"passage_id": r.passage_id, "member_id": int(r.member_id), "mp": r.name, "party": r.party_abbr, "role": r.role,
                 "debate": r.major or r.minor, "policy": g("POLICY"), "side": g("SPEAKER"), "position": g("POSITION"), "text": r.text})
    if i % 100 == 0: print(i, f"{time.time()-t0:.0f}s |", rows[-1]["position"], flush=True)
pd.DataFrame(rows).to_json(f"{tag}_claims.jsonl", orient="records", lines=True)
print("DONE", len(rows), f"{time.time()-t0:.0f}s")
