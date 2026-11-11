"""Build prototypes/edu_measures.html (static, self-contained) from the edu_jev2 results."""
import json, pathlib, re
import pandas as pd
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
D = ROOT / "data/quick/research_2026_11_10"
R = ROOT / "prototypes/research_2026_11_10/results"
OUT = ROOT / "prototypes/edu_measures.html"
MAXLEN = 4000

full = json.load(open(R / "edu_jev2_full.json"))
hc = json.load(open(R / "edu_handcheck.json"))
sides = pd.read_json(D / "edu_jev2_sides.jsonl", lines=True, dtype={"passage_id": str})
pas = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "date", "source", "major", "minor", "text"]).set_index("passage_id")
hcmap = {}
for g in hc["groups"]:
    for c in g["claims"]:
        hcmap[c["passage_id"]] = c

def url(pid, src):
    m = re.match(r"uk\.org\.publicwhip/(debate|westminhall)/([^#]+)", pid)
    if not m: return None
    base = "debates" if m.group(1) == "debate" else "whall"
    return f"https://www.theyworkforyou.com/{base}/?id={m.group(2)}"

def excerpt(text, quote):
    if quote:
        i = text.find(quote[:40])
        if i >= 0:
            return quote
        return quote
    t = text.strip()
    return t[:300].rsplit(" ", 1)[0] + "…" if len(t) > 300 else t

def claim_obj(r):
    p = pas.loc[r["passage_id"]]
    h = hcmap.get(r["passage_id"])
    badge = None
    if h:
        badge = "different" if not h["same_measure"] else ("correct" if h["side_correct"] == "yes" else "wrong")
    t = p["text"]
    return {"mp": r["mp"], "mid": int(r["member_id"]), "party": r["party"], "date": str(p["date"])[:10],
            "debate": " — ".join(x for x in [p["major"], p["minor"]] if isinstance(x, str) and x),
            "ex": excerpt(t, h["quote"] if h else None),
            "full": t[:MAXLEN] + ("…" if len(t) > MAXLEN else ""),
            "url": url(r["passage_id"], p["source"]), "badge": badge,
            "stance": r["stance"], "verdict": r["new_verdict"]}

measures, check = [], []
for g in full["groups"]:
    sub = sides[sides.g == g["g"]]
    acc = {"FOR": {}, "AGAINST": {}}
    dropped = []
    for _, r in sub.iterrows():
        o = claim_obj(r)
        if r["new_verdict"] == r["stance"]:
            acc[r["stance"]].setdefault(r["member_id"], []).append(o)
        else:
            o["reason"] = "no clear position" if r["new_verdict"] == "NEITHER" else "judge read the opposite side"
            dropped.append(o)
    both = set(acc["FOR"]) & set(acc["AGAINST"])
    c = g["c"]
    check.append((g["label"], len(acc["FOR"]), c["mps_for"], len(acc["AGAINST"]), c["mps_against"], len(both)))
    def lst(s): return [{"mp": v[0]["mp"], "party": v[0]["party"], "claims": v} for k, v in acc[s].items() if k not in both]
    bl = [{"mp": acc["FOR"][k][0]["mp"], "party": acc["FOR"][k][0]["party"],
           "claims": acc["FOR"][k] + acc["AGAINST"][k]} for k in both]
    f, a = lst("FOR"), lst("AGAINST")
    section = 0 if len(f) >= 2 and len(a) >= 2 else (1 if (len(f) >= 2 and len(a) == 1) or (len(a) >= 2 and len(f) == 1) or (len(f) >= 1 and len(a) >= 1) else 2)
    measures.append({"label": g["label"], "n": g["n_claims"], "for": f, "against": a, "both": bl, "dropped": dropped, "sec": section})
measures.sort(key=lambda m: (m["sec"], -(len(m["for"]) + len(m["against"]))))
bad = [x for x in check if x[1] != x[2] or x[3] != x[4]]
print("count mismatches vs edu_jev2_full c:", bad if bad else "none"); print("MPs moved to both-sides:", [(x[0], x[5]) for x in check if x[5]])
print("sections:", [sum(m["sec"] == s for m in measures) for s in range(3)])
T = full["totals"]
stats = {"n_measures": T["n_groups"], "real": sum(m["sec"] == 0 for m in measures), "n_claims": T["n_claims"],
         "dropped_pct": round(100 * T["share"]["NEITHER"])}
data = json.dumps({"measures": measures, "stats": stats}, ensure_ascii=False).replace("</", "<\\/")
html = (ROOT / "prototypes/research_2026_11_10/edu_page_template.html").read_text().replace("__DATA__", data)
OUT.write_text(html); print(OUT, round(OUT.stat().st_size / 1e6, 2), "MB")
