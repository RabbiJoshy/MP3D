"""Step 4: write fp_fights_final3.json."""
import json, collections, difflib, pathlib, pandas as pd
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D"); D = ROOT / "data/quick/research_2026_11_10"; R = ROOT / "prototypes/research_2026_11_10/results"
inp = json.load(open(R / "fp_fights_final2.json")); F = {f["fight_id"]: f for f in inp["fights"]}
L = {o["fight_id"]: o for o in json.load(open(R / "fp_clean_labels.json"))["labels"]}
M = json.load(open(R / "fp_clean_merges.json")); U = json.load(open(R / "fp_clean_units.json")); J = json.load(open(R / "fp_clean_judge.json"))
s = pd.read_json(D / "fp/dense_sides3.jsonl", lines=True)
mem = pd.read_parquet(ROOT / "data/clean/members.parquet").drop_duplicates("member_id").set_index("member_id")
pm = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "member_id"]); s = s.merge(pm, on="passage_id"); s = s[s.verdict.isin(["FOR", "AGAINST"])]
dirm = {}
for r in M["merges"]: dirm[(r["a"], r["b"])] = r["direction"]; dirm[(r["b"], r["a"])] = r["direction"]
def status(F_, A): return "fight" if F_ >= 2 and A >= 2 else "one_short" if (F_ >= 2 or A >= 2) and F_ >= 1 and A >= 1 else "one_sided"
out = []; log = []
for u in U:
    lead = F[u["lead"]]; f = json.loads(json.dumps(lead)); f["label"] = u["label"]; f["description"] = u["description"]; f["topic"] = u["topic"]
    f["old_label"] = lead["label"]; f["merged_from"] = u["members"] if u["merged"] else None
    if u["merged"]:
        g = []
        for i in u["members"]:
            d_ = 1 if i == u["lead"] else dirm.get((u["lead"], i), 1)
            g += [{**x, "direction": x["direction"] * d_, "from_fight": i} for x in F[i]["groups"]]
        f["groups"] = g; f["dropped"] = [x for i in u["members"] for x in F[i]["dropped"]]; f["n_dropped"] = sum(F[i]["n_dropped"] for i in u["members"])
    if u["unit"] in J["done"]:
        meta = {m["member_id"]: (m["name"], m["party"]) for i in u["members"] for k in ("mps_for", "mps_against", "mps_both_sides") for m in F[i][k]}
        before = {m["member_id"] for i in u["members"] for k in ("mps_for", "mps_against") for m in F[i][k]}
        Fo, A, Mx = [], [], []
        for mid, gg in s[s.unit == u["unit"]].groupby("member_id"):
            mid = int(mid); v = [1 if x == "FOR" else -1 for x in gg.verdict]; st = sum(v) / len(v)
            name, party = meta.get(mid, (mem.name.get(mid, "?"), mem.party_abbr.get(mid, "?")))
            e = {"member_id": mid, "name": name, "party": party, "claims": [{"passage_id": a, "measure": u["label"], "side": b} for a, b in zip(gg.passage_id, gg.verdict)], "stance": round(st, 3), "focus": len(v)}
            (Fo if st > 0 else A if st < 0 else Mx).append(e)
        f["mps_for"], f["mps_against"], f["mps_both_sides"] = Fo, A, Mx; f["n_dense_candidates"] = int((s.unit == u["unit"]).sum()); f["rejudged"] = u["reason"]
        after = {e["member_id"] for e in Fo + A}; log.append({"unit": u["unit"], "label": u["label"], "reason": u["reason"], "members": u["members"], "mps_before": len(before), "mps_after": len(after)})
    else:
        f["rejudged"] = None
        if u["merged"]:   # merged but not rejudged (budget): union old positions of members, flag it
            f["rejudged"] = "SKIPPED(budget)"
    f["status"] = status(len(f["mps_for"]), len(f["mps_against"])); f["dense"] = True
    if len(f["mps_for"]) + len(f["mps_against"]) >= 3: out.append(f)
    else: log.append({"unit": u["unit"], "label": u["label"], "dropped_lt3": True})
C = collections.Counter(f["status"] for f in out)
res = json.loads(json.dumps(inp)); res["fights"] = out
res["totals"]["by_status_before"] = inp["totals"]["by_status"]; res["totals"]["by_status"] = dict(C); res["totals"]["dense_by_status"] = dict(C); res["totals"]["fights_reported"] = len(out)
pt = {}
for f in out: pt.setdefault(f["topic"], collections.Counter())[f["status"]] += 1
for t, v in res["per_topic"].items(): v["fights_by_status"] = dict(pt.get(t, {}))
res["cost"]["clean"] = {"relabel_usd": json.load(open(R / "fp_clean_labels.json"))["usd"], "merge_usd": M["usd"], "judge_usd": round(J["usd"], 4)}
json.dump(res, open(R / "fp_fights_final3.json", "w"), indent=1)
ch = sorted(((difflib.SequenceMatcher(None, L[i]["old"].lower(), L[i]["new"].lower()).ratio(), i) for i in L))[:8]
before = collections.Counter(f["status"] for f in F.values())
# Yvette Cooper
yc = [(f["label"], "FOR" if any(m["name"].endswith("Yvette Cooper") for m in f["mps_for"]) else "AGAINST", f["topic"]) for f in out if any("Yvette Cooper" in m["name"] for m in f["mps_for"] + f["mps_against"])]
yc_old = [(f["label"], "FOR" if any("Yvette Cooper" in m["name"] for m in f["mps_for"]) else "AGAINST") for f in F.values() if any("Yvette Cooper" in m["name"] for m in f["mps_for"] + f["mps_against"]) and f["topic"] == "Immigration"]
summ = {"before": dict(before), "after": dict(C), "n_before": len(F), "n_after": len(out), "most_changed": [(L[i]["old"], L[i]["new"]) for _, i in ch], "yvette_after": yc, "yvette_before": yc_old, "log": log}
json.dump(summ, open(R / "fp_clean_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != "log"}, indent=1)); print([x for x in log if x.get("dropped_lt3")])
for x in log:
    if not x.get("dropped_lt3"): print(x["unit"], x["label"][:55], x["mps_before"], "->", x["mps_after"])
