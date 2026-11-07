"""Bundle the stance proof of concept into one page: subjects, proposals, MP positions with evidence, concerns, checks."""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
D = ROOT / "data/quick/poc"
ns = {"__file__": str(HERE / "stance_poc.py"), "__name__": "stance_poc"}
exec((HERE / "stance_poc.py").read_text(), ns)
SUBJECTS, load = ns["SUBJECTS"], ns["load"]

members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
members[["party", "party_abbr", "constituency"]] = members[["party", "party_abbr", "constituency"]].fillna("")
members["party_colour"] = members.party_colour.fillna("888888")

mps, subjects = {}, []
for subj, S in SUBJECTS.items():
    if not (D / f"{subj}_scores.parquet").exists() or not (D / f"{subj}_checks.json").exists():
        print("skip", subj); continue
    d = load(subj)
    for (prop, mem), g in d[d.side != 0].groupby(["proposal", "member"]):
        ev = g.assign(strength=np.maximum(g.support, g.oppose)).sort_values("strength", ascending=False).drop_duplicates("debate").head(3)
        e = mps.setdefault(int(mem), {"positions": {}, "concerns": {}, "passages": {}})
        e["positions"][f"{subj}.{prop}"] = {"s": round(float(g.side.mean()), 2), "n": int(len(g)), "debates": int(g.debate.nunique()), "minister": bool((g.role == "minister").any()),
                                             "ev": [{"t": " ".join(r.snip.split()[:70]) + ("…" if len(r.snip.split()) > 70 else ""), "d": r.date, "side": int(r.side)} for r in ev.itertuples()]}
    for mem, g in d.groupby("member"):
        if int(mem) in mps and any(k.startswith(subj + ".") for k in mps[int(mem)]["positions"]):
            mps[int(mem)]["concerns"].update({f"{subj}.{k}": round(float((g[f"c_{k}"] >= .5).mean()), 2) for k in S["concerns"]})
            mps[int(mem)]["passages"][subj] = int(len(g))
    subjects.append({"k": subj, "title": S["title"], "proposals": [{"k": f"{subj}.{k}", "label": v["label"]} for k, v in S["proposals"].items() if not k.startswith("_")],
                     "concerns": [{"k": f"{subj}.{k}", "label": v} for k, v in S["concerns"].items()], "checks": json.loads((D / f"{subj}_checks.json").read_text())})

out = []
for mem, e in mps.items():
    m = members.loc[mem]
    out.append({"i": mem, "n": m["name"], "p": m["party_abbr"], "pn": m["party"], "c": "#" + m["party_colour"], "k": m["constituency"], **e})

# One shared map. Fill each MP's missing positions from their party, fit two axes (iterated low-rank fit), then place every MP,
# and in the page the user, with the same Gaussian update from the positions actually observed: few positions means staying near the prior.
P = [p["k"] for s in subjects for p in s["proposals"]]
X = np.full((len(out), len(P)), np.nan)
for a, m in enumerate(out):
    for b, k in enumerate(P):
        if k in m["positions"]: X[a, b] = m["positions"][k]["s"]
obs, mu, party = ~np.isnan(X), np.nanmean(X, 0), np.array([m["p"] for m in out])
F = X.copy()
for pa in set(party):
    r = party == pa
    pm = np.array([np.nanmean(c) if (~np.isnan(c)).sum() >= 3 else np.nan for c in X[r].T]) if r.sum() >= 3 else mu
    F[r] = np.where(obs[r], X[r], np.where(np.isnan(pm), mu, pm))
for _ in range(40):
    U, S, Vt = np.linalg.svd(F - mu, full_matrices=False)
    R = mu + (U[:, :2] * S[:2]) @ Vt[:2]
    F = np.where(obs, X, R)
W = Vt[:2].T.copy()
if (( F - mu) @ W[:, 0])[party == "Con"].mean() < 0: W[:, 0] *= -1          # Conservatives on the right
if W[P.index("europe.customs"), 1] < 0: W[:, 1] *= -1                         # closer to the EU at the top
Z = (F - mu) @ W
s2 = float(np.mean((X - R)[obs] ** 2))
prior = np.cov(Z.T)
def place(x, o, m0, S0):
    Wo = W[o]; prec = np.linalg.inv(S0) + Wo.T @ Wo / s2
    return np.linalg.solve(prec, np.linalg.inv(S0) @ m0 + Wo.T @ (x[o] - mu[o]) / s2)
pc = {pa: (Z[party == pa].mean(0), np.cov(Z[party == pa].T) + np.eye(2) * .3) for pa in set(party) if (party == pa).sum() >= 3}
for a, m in enumerate(out):
    m0, S0 = pc.get(m["p"], (np.zeros(2), prior))
    m["x"], m["y"] = [round(float(v), 2) for v in place(X[a], obs[a], m0, S0)]
ends = lambda d: {"pos": [P[k] for k in np.argsort(-W[:, d])[:3]], "neg": [P[k] for k in np.argsort(W[:, d])[:3]]}
space = {"keys": P, "W": W.round(4).tolist(), "mu": np.round(mu, 4).tolist(), "s2": round(s2, 4), "prior": prior.round(4).tolist(), "m0": Z.mean(0).round(4).tolist(),
         "share": round(float((S[:2] ** 2).sum() / (S ** 2).sum()), 3), "x": ends(0), "y": ends(1),
         "parties": [{"p": pa, "x": round(float(Z[party == pa, 0].mean()), 2), "y": round(float(Z[party == pa, 1].mean()), 2), "n": int((party == pa).sum())} for pa in pc]}
data = {"subjects": subjects, "mps": out, "space": space}
page = (HERE / "poc_template.html").read_text().replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
(HERE / "poc_stance.html").write_text(page)
print("axes explain", space["share"], "| noise", space["s2"], "| x+", space["x"]["pos"], "| y+", space["y"]["pos"])
print("page KB", len(page.encode()) // 1024, "| MPs", len(out), "| subjects", [s["k"] for s in subjects])
