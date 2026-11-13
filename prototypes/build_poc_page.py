"""Build the legacy stance prototype and its lab. Pass --publish to replace the public pages."""
import json
import pathlib
import sys

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

# The few questions everyone gets (the rest are optional), and a short name for each proposal, used to label the map's axes
CORE = {"asylum.rwanda", "asylum.work", "welfare.twochild", "welfare.winterfuel", "dying.assisted", "energy.northsea", "energy.netzero",
        "gaza.recognise", "gaza.arms", "europe.customs", "europe.rejoin", "justice.jury"}
SHORT = {"asylum.work": "asylum seekers working", "asylum.military": "military-site housing", "asylum.rwanda": "Rwanda scheme", "asylum.hubs": "return hubs",
         "asylum.routes": "safe routes", "asylum.echr": "leaving the ECHR", "asylum.digitalid": "digital ID", "asylum.idmandatory": "compulsory ID",
         "asylum.settlement": "slower settlement", "welfare.twochild": "scrapping two-child limit", "welfare.winterfuel": "winter fuel means test",
         "welfare.pip": "PIP cuts", "dying.assisted": "assisted dying", "energy.northsea": "North Sea drilling", "energy.netzero": "net zero",
         "energy.renewables": "onshore wind and solar", "gaza.arms": "Israel arms ban", "gaza.recognise": "recognising Palestine",
         "gaza.sanctions": "sanctions on Israel", "gaza.palaction": "Palestine Action ban", "europe.customs": "EU customs union",
         "europe.youth": "EU youth scheme", "europe.singlemarket": "single market", "europe.rejoin": "rejoining the EU",
         "justice.jury": "fewer jury trials", "justice.release": "early release", "justice.short": "community sentences",
         "tax.vat": "VAT on private schools", "tax.farms": "farm inheritance tax", "tax.borrow": "borrowing more",
         "work.dismissal": "day-one dismissal rights", "work.minwage": "higher minimum wage", "work.rentcap": "rent caps",
         "ownership.rail": "public rail", "ownership.water": "public water", "elections.pr": "proportional representation"}
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
    subjects.append({"k": subj, "title": S["title"], "proposals": [{"k": f"{subj}.{k}", "label": v["label"], "short": SHORT.get(f"{subj}.{k}", v["label"]), "core": f"{subj}.{k}" in CORE} for k, v in S["proposals"].items() if not k.startswith("_")],
                     "concerns": [{"k": f"{subj}.{k}", "label": v} for k, v in S["concerns"].items()], "checks": json.loads((D / f"{subj}_checks.json").read_text())})

out = []
for mem, e in mps.items():
    m = members.loc[mem]
    out.append({"i": mem, "n": m["name"], "p": m["party_abbr"], "pn": m["party"], "c": "#" + m["party_colour"], "k": m["constituency"], **e})

# One shared map. Fill each MP's missing positions from their party, fit three axes (iterated low-rank fit), then place every MP,
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
K = 3
for _ in range(40):
    U, S, Vt = np.linalg.svd(F - mu, full_matrices=False)
    R = mu + (U[:, :K] * S[:K]) @ Vt[:K]
    F = np.where(obs, X, R)
W = Vt[:K].T.copy()
if (( F - mu) @ W[:, 0])[party == "Con"].mean() < 0: W[:, 0] *= -1          # Conservatives on the right
if W[P.index("europe.customs"), 1] < 0: W[:, 1] *= -1                         # closer to the EU at the top
Z = (F - mu) @ W
s2 = float(np.mean((X - R)[obs] ** 2))
prior = np.cov(Z.T)
def place(x, o, m0, S0):
    Wo = W[o]; prec = np.linalg.inv(S0) + Wo.T @ Wo / s2
    return np.linalg.solve(prec, np.linalg.inv(S0) @ m0 + Wo.T @ (x[o] - mu[o]) / s2)
pc = {pa: (Z[party == pa].mean(0), np.cov(Z[party == pa].T) + np.eye(K) * .3) for pa in set(party) if (party == pa).sum() >= 3}
for a, m in enumerate(out):
    m0, S0 = pc.get(m["p"], (np.zeros(K), prior))
    m["x"], m["y"], m["z"] = [round(float(v), 2) for v in place(X[a], obs[a], m0, S0)]
ends = lambda d: {"pos": [P[k] for k in np.argsort(-W[:, d])[:3]], "neg": [P[k] for k in np.argsort(W[:, d])[:3]]}
space = {"keys": P, "W": W.round(4).tolist(), "mu": np.round(mu, 4).tolist(), "s2": round(s2, 4), "prior": prior.round(4).tolist(), "m0": Z.mean(0).round(4).tolist(),
         "share": round(float((S[:K] ** 2).sum() / (S ** 2).sum()), 3), "x": ends(0), "y": ends(1), "z": ends(2),
         "parties": [{"p": pa, "x": round(float(Z[party == pa, 0].mean()), 2), "y": round(float(Z[party == pa, 1].mean()), 2), "z": round(float(Z[party == pa, 2].mean()), 2), "n": int((party == pa).sum())} for pa in pc]}
# Name regions of the space rather than axis ends: cluster MPs in the 3D space, then describe each region
# by the proposals its MPs take a clear, different side on from everyone else.
Zm = np.array([[m["x"], m["y"], m["z"]] for m in out]); Zs = (Zm - Zm.mean(0)) / Zm.std(0)
rng, best = np.random.default_rng(0), None
for _ in range(10):
    C = Zs[rng.choice(len(Zs), 6, replace=False)]
    for _ in range(100):
        L = ((Zs[:, None] - C) ** 2).sum(2).argmin(1); C = np.array([Zs[L == c].mean(0) if (L == c).any() else C[c] for c in range(6)])
    inert = ((Zs - C[L]) ** 2).sum()
    if best is None or inert < best[0]: best = (inert, L.copy())
L, overall, regions = best[1], np.nanmean(X, 0), []
for c in range(6):
    r = L == c
    if r.sum() < 20: continue
    with np.errstate(all="ignore"): mean = np.nanmean(X[r], 0)
    n = (~np.isnan(X[r])).sum(0)
    d = sorted([q for q in range(len(P)) if n[q] >= 4 and abs(mean[q]) >= .5], key=lambda q: -abs(mean[q] - overall[q]))[:3]
    regions.append({"x": round(float(Zm[r, 0].mean()), 2), "y": round(float(Zm[r, 1].mean()), 2), "z": round(float(Zm[r, 2].mean()), 2), "n": int(r.sum()),
                    "says": [("for " if mean[q] > 0 else "against ") + SHORT[P[q]] for q in d]})
space["regions"] = regions
# How divisive each issue is vs how much it predicts other views (lab page only)
Sg = np.where(np.isnan(X), 0, np.sign(X) * (np.abs(X) > .2))
sal = {}
for subj in SUBJECTS:
    f = D / f"{subj}_snippets.parquet"
    if f.exists():
        sn = pd.read_parquet(f)
        for prop, g in sn.groupby("proposal"): sal[f"{subj}.{prop}"] = int(g.member.nunique())
issues = []
for a, k in enumerate(P):
    col = Sg[:, a]; nn = int((col != 0).sum()); ph = []
    for b in range(len(P)):
        r = (col != 0) & (Sg[:, b] != 0)
        if b != a and r.sum() >= 12 and col[r].std() > 0 and Sg[r, b].std() > 0: ph.append(abs(np.corrcoef(col[r], Sg[r, b])[0, 1]))
    if ph: issues.append({"k": k, "short": SHORT[k], "split": round(min((col > 0).sum(), (col < 0).sum()) / max(nn, 1), 2), "predicts": round(float(np.mean(ph)), 2), "spoke": sal.get(k, 0), "core": k in CORE})
# How much each MP talks about each subject: passages raising any of its proposals, as a share of everything they said
tot = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["member_id"]).member_id.value_counts()
att, attp = {}, {}
for subj in SUBJECTS:
    f = D / f"{subj}_snippets.parquet"
    if f.exists():
        sn = pd.read_parquet(f, columns=["member", "pid", "proposal"])
        for mem, n in sn.drop_duplicates("pid").member.value_counts().items(): att.setdefault(int(mem), {})[subj] = int(n)
        for (prop, mem), n in sn.drop_duplicates(["pid", "proposal"]).groupby(["proposal", "member"]).size().items(): attp.setdefault(int(mem), {})[f"{subj}.{prop}"] = int(n)
for m in out:
    m["ap"] = {k: n for k, n in attp.get(m["i"], {}).items() if k in P}
    t = int(tot.get(m["i"], 0)); m["att"] = {k: [n, round(n / t, 4)] for k, n in att.get(m["i"], {}).items()} if t else {}
data = {"subjects": subjects, "mps": out, "space": space}
dump = lambda d: json.dumps(d, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
# Keep the prototype outputs separate from the visitor app built by build_unified_space.py.
page = (HERE / "poc_template.html").read_text().replace("__DATA__", dump({**data, "subjects": [{k: v for k, v in S.items() if k != "checks"} for S in subjects]}))
(HERE / "poc.html").write_text(page)
# The lab: tabs for the app, research views and checks, all local
lab = {"subjects": [{"k": S["k"], "title": S["title"], "checks": S["checks"], "proposals": S["proposals"]} for S in subjects], "issues": issues,
       "space": {k: space[k] for k in ("share", "s2", "regions")}}
pr = ROOT / "data/quick/agenda_audit/pilot_report.json"
if pr.exists():
    lab["pilot"] = json.loads(pr.read_text())
    lab["pilot"]["proposals"] = json.loads((ROOT / "data/quick/agenda_audit/pilot_proposals.json").read_text())["proposals"]
lab_page = (HERE / "lab_template.html").read_text().replace("__DATA__", dump(lab))
(HERE / "poc_lab.html").write_text(lab_page.replace('src="app.html"', 'src="poc.html"'))
if "--publish" in sys.argv:
    raise SystemExit("Legacy --publish retired. Use prototypes/build_hierarchy_app.py for release pages.")
print("axes explain", space["share"], "| noise", space["s2"], "| x+", space["x"]["pos"], "| y+", space["y"]["pos"], "| z+", space["z"]["pos"], "| z-", space["z"]["neg"])
print("regions", [(r["n"], r["says"]) for r in regions])
print("page KB", len(page.encode()) // 1024, "| MPs", len(out), "| subjects", [s["k"] for s in subjects])
