"""sk24: plain names for the two ends of each map axis, written once by a model from the arguments at each end and cached.
The builder reuses a cached name only while the arguments at that end are still mostly the same (see build_hierarchy_app.py).
Run after a build:  .venv/bin/python prototypes/research_2026_11_10/sk24_axis_names.py   (one small DeepSeek call, well under 1 cent)"""
import json, re, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(HERE)); import sk4_common as C
D = json.loads(re.search(r'<script id="d" type="application/json">(.*?)</script>', (ROOT / "docs/index.html").read_text(encoding="utf-8"), re.S).group(1))
V = np.array(D["model"]["v"], float); F = D["fights"]; n = np.array([sum(v is not None for v in f["sc"]) for f in F])
X = np.array(D["xyz"], float); on = np.array(D["onmap"], bool); P = np.array([m["p"] for m in D["mps"]])
ends = []
for a in range(3):
    o = np.argsort(V[:, a] * np.sqrt(n))
    ends.append({"plus": [F[j]["label"] for j in o[::-1][:7]], "minus": [F[j]["label"] for j in o[:7]],
                 "parties": {k: round(float(X[on & (P == k), a].mean()), 2) for k in ["Lab", "Con", "LD", "RUK", "Green", "SNP"] if (on & (P == k)).sum() >= 3}})
SYS = ("You name the two ends of each axis of a map of UK MPs built from their Commons speeches. For each axis you get the proposals that MPs at the PLUS end "
       "support (and MPs at the MINUS end oppose), the proposals MPs at the MINUS end support, and the average position of each party (positive = plus end). "
       "Give each end a plain name of 2 to 4 words that a general reader would recognise as a political position or theme, e.g. 'Workers' rights and welfare', "
       "'Tighter borders', 'Government programme'. British English, neutral in tone, no party names, no bill names, no jargon. If an end is a mix with no clear theme, "
       "name the most common thread honestly rather than inventing one. If an end mixes proposals from opposite political wings that share only their opposition to the other end, say that plainly (e.g. 'Critics from left and right'), and if the other end is simply the measures the government of the day is passing, call it that ('Government programme'). Return JSON {\"axes\":[{\"plus\":...,\"minus\":...,\"why\":<one sentence>} x3]}.")
msg = "\n\n".join(f"AXIS {i+1}\nPLUS end supports: " + "; ".join(e["plus"]) + "\nMINUS end supports: " + "; ".join(e["minus"]) + f"\nParty averages: {e['parties']}" for i, e in enumerate(ends))
r = json.loads(C.chat(C.BIG, [{"role": "system", "content": SYS}, {"role": "user", "content": msg}], "big", max_tokens=500))["axes"]
out = [{"plus_name": x["plus"], "minus_name": x["minus"], "why": x.get("why", ""), "plus_args": e["plus"], "minus_args": e["minus"]} for x, e in zip(r, ends)]
json.dump({"axes": out, "cost_cents": round(C.cost_cents(), 3)}, open(HERE / "results/axis_names.json", "w"), indent=1, ensure_ascii=False)
for i, x in enumerate(out): print(f"Axis {i+1}: {x['plus_name']} ↔ {x['minus_name']}  | {x['why']}")
print("cents", round(C.cost_cents(), 3))
