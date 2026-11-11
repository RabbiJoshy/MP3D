"""Generic: contested debates (>=2 MPs each side) -> one axis. Each debate's direction is chosen to maximise agreement
among MPs who appear in more than one debate (brute force over signs); party is never used."""
import sys, json, itertools, numpy as np, pandas as pd
tag = sys.argv[1]
c = pd.read_json(f"{tag}_jev.jsonl", lines=True); c = c[c.jev != "UNRELATED"]
deb = [p for p, g in c.groupby("proposition") if min(g[g.jev == "AGREE"].member_id.nunique(), g[g.jev == "DISAGREE"].member_id.nunique()) >= 2]
c = c[c.proposition.isin(deb)].assign(s=lambda x: 2 * x.p_agree - 1)
M = c.groupby(["member_id", "proposition"]).s.mean().unstack()
multi = M[M.notna().sum(1) >= 2]
def fit(signs):
    X = multi[deb] * np.array(signs); return np.nansum([np.nanstd(r) for r in X.values])  # low spread within MP = consistent
best = min(itertools.product([1, -1], repeat=len(deb)), key=lambda s: (fit((1,) + s[1:]) if True else 0))
best = (1,) + best[1:] if best[0] == 1 else tuple(-x for x in best)
X = M[deb] * np.array(best)
mp = pd.DataFrame({"score": X.mean(1), "n": X.notna().sum(1)})
mem = c.drop_duplicates("member_id").set_index("member_id")[["mp", "party", "role"]]
mp = mp.join(mem).sort_values("score")
print("debates and direction:"); [print(f"  {'+' if s > 0 else '-'} {d}") for d, s in zip(deb, best)]
print(f"\nMPs in >=2 debates: {len(multi)}  (used to set directions)\n")
print(mp.to_string())
json.dump({"debates": [{"proposition": d, "sign": s} for d, s in zip(deb, best)],
           "mps": mp.reset_index().to_dict("records")}, open(f"{tag}_axis.json", "w"), indent=1, default=str)
