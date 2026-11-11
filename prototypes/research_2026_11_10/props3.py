"""Generic: 3 one-idea candidate propositions per issue."""
import sys, re, json, pandas as pd
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
tag = sys.argv[1]
model, tok = load("mlx-community/Qwen3-4B-Instruct-2507-4bit")
c = pd.read_json(f"{tag}_issues.jsonl", lines=True)
SYS = ("Below are positions taken by different UK MPs on one policy issue. Write THREE different propositions, one per line, about the main "
       "policy choice these MPs argue over. Rules: each is ONE idea only, of the form '<policy> should <verb> ...', at most 10 words; "
       "no 'with', 'to', 'and', 'while' or 'in order to' add-ons; something UK MPs genuinely split on, not a goal everyone shares. "
       "No numbering, no extra text.")
out = {}
for iss, g in c.groupby("issue"):
    lst = "\n".join("- " + x for x in g.position.sample(min(30, len(g)), random_state=0))
    o = generate(model, tok, prompt=tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": lst}], add_generation_prompt=True, tokenize=False), max_tokens=90, sampler=make_sampler(temp=0.0))
    out[int(iss)] = [re.sub(r"^[\-\d\.\)\s]+", "", x).strip(' "') for x in o.split("\n") if x.strip()][:3]
    print(iss, out[int(iss)], flush=True)
json.dump(out, open(f"{tag}_cands.json", "w"), indent=1)
