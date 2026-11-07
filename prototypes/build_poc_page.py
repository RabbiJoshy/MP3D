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
    subjects.append({"k": subj, "title": S["title"], "proposals": [{"k": f"{subj}.{k}", "label": v["label"]} for k, v in S["proposals"].items()],
                     "concerns": [{"k": f"{subj}.{k}", "label": v} for k, v in S["concerns"].items()], "checks": json.loads((D / f"{subj}_checks.json").read_text())})

out = []
for mem, e in mps.items():
    m = members.loc[mem]
    out.append({"i": mem, "n": m["name"], "p": m["party_abbr"], "pn": m["party"], "c": "#" + m["party_colour"], "k": m["constituency"], **e})
data = {"subjects": subjects, "mps": out}
page = (HERE / "poc_template.html").read_text().replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
(HERE / "poc_stance.html").write_text(page)
print("page KB", len(page.encode()) // 1024, "| MPs", len(out), "| subjects", [s["k"] for s in subjects])
