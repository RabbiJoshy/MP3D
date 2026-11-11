"""Generic step C v2: 3 simple candidate propositions per issue; keep the one that splits a probe set most evenly;
label every claim by reading the original speech (plus its summary)."""
import sys, re, time, pandas as pd
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
tag = sys.argv[1]
model, tok = load("mlx-community/Qwen3-4B-Instruct-2507-4bit")
S = make_sampler(temp=0.0)
ask = lambda s, u, n=60: generate(model, tok, prompt=tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": u}], add_generation_prompt=True, tokenize=False), max_tokens=n, sampler=S).strip()
c = pd.read_json(f"{tag}_issues.jsonl", lines=True)
P_SYS = ("Below are positions taken by different UK MPs on one policy issue. Write THREE different simple propositions, one per line, "
         "each of the form '<policy> should <verb> ...' in at most 10 words, about the main policy choice these MPs argue over. "
         "Each must be something UK MPs genuinely split on, not a goal everyone shares (avoid 'should be fair', 'should be improved'). "
         "No numbering, no extra text.")
L_SYS = ("Proposition: {p}\n\nYou will read an excerpt of a UK House of Commons speech. Would THIS SPEAKER agree or disagree with the proposition? "
         "Judge what the speaker argues, not what they describe: speakers often describe a policy or another party's plan in order to attack it. "
         "If the excerpt does not bear on the proposition, answer UNRELATED. Answer with one word: AGREE, DISAGREE or UNRELATED.")
def label(p, r):
    a = ask(L_SYS.format(p=p), f"Speech excerpt:\n{r.text[:2000]}\n\nSummary of the speaker's view: {r.position}", 4).upper()
    return "AGREE" if a.startswith("AGREE") else "DISAGREE" if a.startswith("DISAGREE") else "UNRELATED"
props, labels, t0 = {}, {}, time.time()
for iss, g in c.groupby("issue"):
    lst = "\n".join("- " + x for x in g.position.sample(min(30, len(g)), random_state=0))
    cands = [re.sub(r"^[\-\d\.\)\s]+", "", x).strip(' "') for x in ask(P_SYS, lst, 90).split("\n") if x.strip()][:3]
    probe = g.sample(min(15, len(g)), random_state=1)
    best, bscore = None, -1
    for p in cands:
        ls = [label(p, r) for r in probe.itertuples()]
        a, d = ls.count("AGREE"), ls.count("DISAGREE")
        score = (min(a, d) / (a + d)) * ((a + d) / len(ls)) if a + d else 0   # balanced AND relevant
        print(f"   [{iss}] {score:.2f} a{a} d{d} | {p}", flush=True)
        if score > bscore: best, bscore = p, score
    props[iss] = best
    for r in g.itertuples(): labels[r.Index] = label(best, r)
    print(f"[{iss}] n={len(g)} CHOSEN: {best}  ({time.time()-t0:.0f}s)", flush=True)
c["label"] = pd.Series(labels); c["proposition"] = c.issue.map(props)
c.to_json(f"{tag}_labelled.jsonl", orient="records", lines=True)
