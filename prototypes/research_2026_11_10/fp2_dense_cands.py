"""fp2 step 7a: dense candidates per unit (home topic in any member fight's topic, in the unit's own debates, words>=60, has member_id), cap per unit, seeded."""
import json, random, sys, pandas as pd
ROOT = "/Users/joshuathomasamar/PycharmProjects/MP3D"; D = ROOT + "/data/quick/research_2026_11_10"
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 150
U = json.load(open(D + "/fp2/units.json")); FP = {f["fight_id"]: f for f in json.load(open(D + "/fp2/fights_pre.json"))}
p = pd.read_parquet(ROOT + "/data/clean/passages.parquet", columns=["passage_id", "member_id", "date", "major", "words", "text"]); p["home"] = pd.read_parquet(D + "/homes.parquet")["home"].values
p["date"] = p["date"].astype(str); deb = pd.Series(list(zip(p.date, p.major)), index=p.index)
rows = []; rep = []
for u in sorted(U, key=lambda u: (-u["n_mps"], u["unit"])):
    ds = {tuple(d) for d in u["debates"]}
    q = p[p.home.isin(u["topics"]) & (p.words >= 60) & p.member_id.notna() & deb.isin(ds)]
    acc = {c["passage_id"] for j in u["members"] for c in FP[j]["claims"]}
    ids = sorted(set(q.passage_id)); keep = sorted(acc & set(ids)); rest = [i for i in ids if i not in acc]
    r = random.Random(20261110); r.shuffle(keep); r.shuffle(rest)
    sel = (keep + rest)[:CAP]
    t = q.set_index("passage_id").text.reindex(sel)
    est = (t.str.len().clip(upper=3000).sum() / 4 + len(t) * 120) * 0.042 / 1e6  # one call answers both questions
    rep.append(dict(unit=u["unit"], n_mps=u["n_mps"], avail=len(ids), n=len(sel), est=est))
    for i in sel: rows.append({"passage_id": i, "unit": u["unit"], "label": u["label"], "n_mps": u["n_mps"]})
c = pd.DataFrame(rows); c.to_json(D + "/fp2/dense_cands.jsonl", orient="records", lines=True)
r = pd.DataFrame(rep); r.to_json(D + "/fp2/dense_cands_report.json", orient="records")
print("units", len(r), "cands", len(c), "uncapped", int(r.avail.sum()), "est $ (1 call/passage)", round(r.est.sum(), 3), "est $ (2x)", round(r.est.sum() * 2, 3))
