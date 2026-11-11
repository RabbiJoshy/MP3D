"""Generic step C: one proposition per issue (written from the claims only), then each claim AGREE/DISAGREE/UNRELATED."""
import sys, re, pandas as pd
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
tag = sys.argv[1]
model, tok = load("mlx-community/Qwen3-4B-Instruct-2507-4bit")
S = make_sampler(temp=0.0)
ask = lambda sys_, u, n=60: generate(model, tok, prompt=tok.apply_chat_template([{"role": "system", "content": sys_}, {"role": "user", "content": u}], add_generation_prompt=True, tokenize=False), max_tokens=n, sampler=S).strip()
c = pd.read_json(f"{tag}_issues.jsonl", lines=True)
P_SYS = ("Below are positions taken by different UK MPs on one policy issue. Write ONE short proposition of the form "
         "'<policy> should <do something>' (max 15 words) that these MPs would split on: some agree, some disagree. "
         "Use the specific policy, not a vague goal everyone shares. Output only the proposition.")
L_SYS = ("Proposition: {p}\nDoes the following MP's position AGREE or DISAGREE with the proposition, or is it UNRELATED to it? "
         "Answer with one word: AGREE, DISAGREE or UNRELATED.")
props, labels = {}, []
for iss, g in c.groupby("issue"):
    lst = "\n".join("- " + x for x in g.position.sample(min(30, len(g)), random_state=0))
    props[iss] = ask(P_SYS, lst).split("\n")[0].strip(' "')
    print(f"\n[{iss}] n={len(g)} PROPOSITION: {props[iss]}", flush=True)
    for r in g.itertuples():
        a = ask(L_SYS.format(p=props[iss]), r.position, 4).upper()
        labels.append((r.Index, "AGREE" if a.startswith("AGREE") else "DISAGREE" if a.startswith("DISAGREE") else "UNRELATED"))
c["label"] = pd.Series(dict(labels)); c["proposition"] = c.issue.map(props)
c.to_json(f"{tag}_labelled.jsonl", orient="records", lines=True)
print(c.groupby("proposition").label.value_counts().unstack(fill_value=0).to_string())
