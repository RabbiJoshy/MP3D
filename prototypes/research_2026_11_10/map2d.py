"""Generic 2D stance map for one topic. Contested debates (>= MIN_SIDE MPs each side) -> MP x debate matrix
(value = 2*p_agree-1, averaged over an MP's claims; missing = 0 = no evidence) -> PCA on MPs with >= MIN_DEB debates.
Axes are named by the debates loading most on them. No party information is used. Output: <tag>_map2d.json"""
import sys, re, json, numpy as np, pandas as pd

tag = sys.argv[1]
MIN_SIDE, MIN_DEB = 3, 2
c = pd.read_json(f"{tag}_jev.jsonl", lines=True)
c = c[c.jev != "UNRELATED"].assign(s=lambda x: 2 * x.p_agree - 1)
sides = c.groupby("proposition").apply(lambda g: pd.Series({"a": g[g.jev == "AGREE"].member_id.nunique(), "d": g[g.jev == "DISAGREE"].member_id.nunique()}))
deb = sides[(sides.a >= MIN_SIDE) & (sides.d >= MIN_SIDE)].index.tolist()
print(f"{len(deb)} contested debates of {len(sides)}")
c = c[c.proposition.isin(deb)]
M = c.groupby(["member_id", "proposition"]).s.mean().unstack()[deb]
keep = M.notna().sum(axis=1) >= MIN_DEB
X = M[keep].fillna(0).values
X = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
var = S ** 2 / (S ** 2).sum()
coords = U[:, :2] * S[:2]
mem = c.drop_duplicates("member_id").set_index("member_id")[["mp", "party", "role"]]
mps = pd.DataFrame(coords, index=M.index[keep], columns=["x", "y"]).join(mem)
mps["n_debates"] = M[keep].notna().sum(axis=1).values

def short(s, n=230):
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"

axes = []
for a in range(2):
    load = pd.Series(Vt[a], index=deb).sort_values()
    axes.append({"var": float(var[a]), "neg": list(load.index[:3]), "pos": list(load.index[-3:][::-1]),
                 "load": {k: round(float(v), 3) for k, v in load.items()}})
    print(f"\nAXIS {a+1} ({var[a]:.0%}):")
    for k, v in load.items():
        if abs(v) > .15: print(f"   {v:+.2f} {k}")

debates = []
for p in deb:
    g = c[c.proposition == p]
    rec = {}
    for lab in ("AGREE", "DISAGREE"):
        h = g[g.jev == lab].assign(conf=lambda x: (x.p_agree - .5).abs(), bb=lambda x: x.role != "minister")
        h = h.sort_values(["bb", "conf"], ascending=False).drop_duplicates("member_id").head(2)
        rec[lab] = [{"mp": r.mp, "party": r.party, "position": r.position, "quote": short(r.text)} for r in h.itertuples()]
    debates.append({"proposition": p, "agree": int(sides.loc[p, "a"]), "disagree": int(sides.loc[p, "d"]),
                    "vec": [round(float(Vt[0][deb.index(p)]), 3), round(float(Vt[1][deb.index(p)]), 3)], "receipts": rec})

party_means = mps.groupby("party")[["x", "y"]].mean().round(2)
print("\nparty means:\n", party_means[mps.party.value_counts().reindex(party_means.index) >= 3].to_string())
print(f"\nMPs placed: {len(mps)}")
json.dump({"topic": tag, "n_mps": len(mps), "axes": axes, "debates": debates,
           "mps": mps.reset_index().round(3).to_dict("records")}, open(f"{tag}_map2d.json", "w"), indent=1, default=str)
