"""Generic step D: keep issues with real disagreement; per debate: sides, MPs by party, receipts."""
import sys, re, json, pandas as pd
tag = sys.argv[1]
c = pd.read_json(f"{tag}_labelled.jsonl", lines=True)
c = c[c.label != "UNRELATED"]
short = lambda s, n=260: (s := re.sub(r"\s+", " ", s).strip()) if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"
out = []
for prop, g in c.groupby("proposition"):
    mp = g.groupby(["member_id", "mp", "party"]).label.agg(lambda s: s.mode()[0]).reset_index()
    a, d = (mp.label == "AGREE").sum(), (mp.label == "DISAGREE").sum()
    if min(a, d) < 2: continue
    def rec(lab):
        h = g[g.label == lab].assign(bb=lambda x: x.role != "minister").sort_values(["bb"], ascending=False).drop_duplicates("member_id")
        return [{"mp": r.mp, "party": r.party, "position": r.position, "quote": short(r.text)} for r in h.head(3).itertuples()]
    out.append({"proposition": prop, "agree": int(a), "disagree": int(d), "split": round(min(a, d) / (a + d), 2),
                "agree_parties": mp[mp.label == "AGREE"].party.value_counts().to_dict(),
                "disagree_parties": mp[mp.label == "DISAGREE"].party.value_counts().to_dict(),
                "agree_receipts": rec("AGREE"), "disagree_receipts": rec("DISAGREE")})
out.sort(key=lambda x: -(x["agree"] + x["disagree"]) * x["split"])
json.dump(out, open(f"{tag}_debates.json", "w"), indent=1)
for o in out:
    print(f"\n{o['proposition']}  agree {o['agree']} {o['agree_parties']} | disagree {o['disagree']} {o['disagree_parties']}")
    for s in ("agree", "disagree"):
        for r in o[s + "_receipts"][:2]: print(f"   {s[:3].upper()} {r['mp']} ({r['party']}): {r['position']}")
