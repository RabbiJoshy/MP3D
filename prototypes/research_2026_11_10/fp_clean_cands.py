"""Step 3a: build final3 fight units (merged / relabelled) and dense candidates for the ones to rejudge."""
import json, random, pathlib, pandas as pd
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D"); R = ROOT / "prototypes/research_2026_11_10/results"; D = ROOT / "data/quick/research_2026_11_10"
F = {f["fight_id"]: f for f in json.load(open(R / "fp_fights_final2.json"))["fights"]}
L = {o["fight_id"]: o for o in json.load(open(R / "fp_clean_labels.json"))["labels"]}
M = json.load(open(R / "fp_clean_merges.json"))
p = pd.read_parquet(ROOT / "data/clean/passages.parquet"); p["home"] = pd.read_parquet(D / "homes.parquet")["home"].values
p["date"] = p["date"].astype(str); pid2 = p.set_index("passage_id"); deb = pd.Series(list(zip(p.date, p.major)), index=p.index)
n_mps = lambda f: len({m["member_id"] for k in ("mps_for", "mps_against") for m in f[k]})
units = []; inc = {i for c in M["components"] for i in c}
for c in M["components"]:
    lead = max(c, key=lambda i: (n_mps(F[i]), -i)); units.append({"members": c, "lead": lead, "merged": True, "reason": "merged"})
for i in F:
    if i not in inc:
        l = L[i]; units.append({"members": [i], "lead": i, "merged": False, "reason": "unclear" if not l["old_label_clear"] else "flipped" if not l["same_direction"] else None})
rows = []; info = []
for k, u in enumerate(units):
    u["unit"] = k; u["label"] = L[u["lead"]]["new"]; u["description"] = L[u["lead"]]["description"]; u["topic"] = F[u["lead"]]["topic"]
    if not u["reason"]: continue
    acc = {c["passage_id"] for i in u["members"] for kk in ("mps_for", "mps_against", "mps_both_sides") for m in F[i][kk] for c in m["claims"]}
    ad = pid2.loc[[x for x in acc if x in pid2.index]]; ds = set(zip(ad.date, ad.major)); tops = {F[i]["topic"] for i in u["members"]}
    q = p[p.home.isin(tops) & (p.words >= 60) & p.member_id.notna() & deb.isin(ds)]
    ids = sorted(set(q.passage_id) | (acc & set(pid2.index))); keep = sorted(acc & set(ids)); rest = [x for x in ids if x not in acc]
    r = random.Random(20261110); r.shuffle(rest); sel = keep + rest[:max(0, 200 - len(keep))]
    t = pid2.text[sel]; est = (t.str.len().clip(upper=3000).sum() / 4 + len(t) * 120) * 0.042 / 1e6 * 100  # cents (one call = both questions)
    est2 = (t.str.len().clip(upper=3000).sum() / 4 * 2 + len(t) * 240) * 0.042 / 1e6 * 100
    u["n_cands"] = len(sel); u["est_cents"] = round(est2, 3); info.append((k, u["label"][:50], len(sel), round(est2, 2), u["reason"]))
    for x in sel: rows.append({"passage_id": x, "unit": k, "label": u["label"]})
pd.DataFrame(rows).to_json(D / "fp/dense3_cands.jsonl", orient="records", lines=True)
json.dump(units, open(R / "fp_clean_units.json", "w"), indent=1)
for x in sorted(info, key=lambda x: -x[2]): print(x)
print("units", len(units), "to rejudge", len(info), "cands", len(rows), "est cents", round(sum(x[3] for x in info), 2))
