import json, sys, time, pandas as pd
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
model, tok = load("mlx-community/Qwen3-4B-Instruct-2507-4bit")
t = pd.read_parquet("claim_sample.parquet")
SYS = ("You read short excerpts of UK House of Commons speeches. If the speaker argues for or against a specific policy, "
       "write the position THEY argue as one plain-English sentence of at most 15 words, e.g. 'The two-child benefit limit should be scrapped.' "
       "State the policy itself, not who supports it. If they only ask a question, give thanks, describe a case, or attack a party without a policy position, write NONE. Output only the sentence.")
out, t0 = [], time.time()
for i, r in enumerate(t.itertuples()):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": r.text[:2500]}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    c = generate(model, tok, prompt=prompt, max_tokens=40, sampler=make_sampler(temp=0.0)).strip().split("\n")[0]
    out.append({"passage_id": r.passage_id, "member_id": int(r.member_id), "mp": r.name, "party": r.party_abbr, "claim": c, "text": r.text[:400]})
    if i % 25 == 0: print(i, f"{time.time()-t0:.0f}s", r.party_abbr, "|", c, flush=True)
pd.DataFrame(out).to_json("claims.jsonl", orient="records", lines=True)
print("DONE", len(out), f"{time.time()-t0:.0f}s")
