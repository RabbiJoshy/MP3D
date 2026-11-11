"""Build prototypes/space.html from fp_fights(_v2).json.  Usage: build_space_page.py [input.json]"""
import json, pathlib, re, sys, collections
import numpy as np, pandas as pd
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
R = ROOT / "prototypes/research_2026_11_10/results"
OUT = ROOT / "prototypes/space.html"
if len(sys.argv) > 1: INP = pathlib.Path(sys.argv[1])
else: INP = R / "fp_fights_final3.json"
SEED, LAM, RANK = 20261110, 1.0, 3
d = json.load(open(INP)); F = d["fights"]
print("input:", INP, len(F), "fights")

# ---------- passages ----------
need = set()
for f in F:
    for s in ("mps_for", "mps_against", "mps_both_sides"):
        for m in f[s]:
            for c in m["claims"]: need.add(c["passage_id"])
    for x in f["dropped"]: need.add(x["passage_id"])
pas = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "member_id", "date", "source", "major", "minor", "text"])
pas = pas[pas.passage_id.isin(need)].set_index("passage_id")
mem = pd.read_parquet(ROOT / "data/clean/members.parquet", columns=["member_id", "name", "party_abbr"]).drop_duplicates("member_id").set_index("member_id")
accepted = set()
for f in F:
    for s in ("mps_for", "mps_against", "mps_both_sides"):
        for m in f[s]:
            for c in m["claims"]: accepted.add(c["passage_id"])

def url(pid):
    m = re.match(r"uk\.org\.publicwhip/(debate|westminhall)/([^#]+)", pid)
    if not m: return None
    return f"https://www.theyworkforyou.com/{'debates' if m.group(1)=='debate' else 'whall'}/?id={m.group(2)}"
W = lambda s: set(re.findall(r"[a-z]{4,}", s.lower()))
def excerpt(text, measure, n=300):
    t = " ".join(text.split())
    if len(t) <= n: return t
    sents = re.split(r"(?<=[.!?])\s+", t); mw = W(measure)
    best, bi = -1, 0
    for i, s in enumerate(sents):
        sc = len(W(s) & mw)
        if sc > best: best, bi = sc, i
    out = ""
    for s in sents[bi:]:
        if len(out) + len(s) > n and out: break
        out += (" " if out else "") + s
    out = out[:n].rsplit(" ", 1)[0] if len(out) > n else out
    return ("… " if bi > 0 else "") + out + ("…" if len(out) < len(t) else "")
P = {}
def pas_obj(pid, measure):
    if pid in P: return
    r = pas.loc[pid]; t = r["text"] or ""
    cap = 4000 if pid in accepted else 1500
    P[pid] = {"date": str(r["date"])[:10],
              "debate": " — ".join(x for x in [r["major"], r["minor"]] if isinstance(x, str) and x),
              "ex": excerpt(t, measure), "full": t[:cap] + ("…" if len(t) > cap else ""), "url": url(pid)}

fights = []
for i, f in enumerate(F):
    def lst(key, side):
        out = []
        for m in f[key]:
            cl = []
            for c in m["claims"]:
                pas_obj(c["passage_id"], c["measure"]); cl.append({"p": c["passage_id"], "m": c["measure"], "s": c["side"]})
            o = {"id": m["member_id"], "mp": m["name"], "party": m["party"], "claims": cl}
            if f.get("dense"): o["stance"] = m["stance"]; o["focus"] = m["focus"]
            out.append(o)
        return out
    dr = []
    for x in f["dropped"]:
        pid = x["passage_id"]
        if pid not in pas.index: continue
        pas_obj(pid, f["label"]); mid = pas.loc[pid, "member_id"]
        nm, pt = (mem.loc[mid, "name"], mem.loc[mid, "party_abbr"]) if mid in mem.index else ("Unknown MP", "?")
        dr.append({"p": pid, "mp": nm, "party": pt, "reason": x["reason"]})
    fl_ = lst("mps_for", "FOR"); ag_ = lst("mps_against", "AGAINST")
    sp = []
    for pt in sorted({x["party"] for x in fl_ + ag_}):
        a = [x for x in fl_ if x["party"] == pt]; b = [x for x in ag_ if x["party"] == pt]
        small = a if len(a) < len(b) else b
        if len(small) >= 2: sp.append({"party": pt, "nf": len(a), "na": len(b), "names": [x["mp"] for x in small], "small": "for" if small is a else "against", "m": min(len(a), len(b))})
    fights.append({"i": i, "dense": bool(f.get("dense")), "splits": sp, "topic": f["topic"], "label": f["label"], "desc": f.get("description", ""), "status": f["status"],
                   "groups": f["groups"], "for": lst("mps_for", "FOR"), "against": lst("mps_against", "AGAINST"),
                   "both": lst("mps_both_sides", ""), "dropped": dr, "n_dropped": f["n_dropped"]})

# ---------- topic space ----------
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import topic_space as TS
TSR = TS.build(F)
HO = TSR.pop("holdout")
json.dump(HO, open(R / "fp_topic_holdout.json", "w"), indent=1)
json.dump(TSR, open(R / "fp_topic_space.json", "w"), indent=1)
for i, f in enumerate(fights):
    fv = TSR["free_vote"][i]; f["conscience"] = fv["flag"]; f["fv"] = fv["parties"]
print("conscience:", [c["label"] for c in TSR["conscience"]]); print("holdout", json.dumps(HO))
M_ = TSR["map"]
mapd = {"mps": M_["mps"], "axes": M_["axes"], "party_mean": M_["party_mean"], "beyond": TSR["beyond"], "holdout": HO, "distinct": TSR["distinct"], "tnames": TSR["tnames"]}
MPI = {y['member_id']: (y['name'], y['party']) for f in F for k in ('mps_for', 'mps_against', 'mps_both_sides') for y in f[k]}
topic_scales = {t: {k: v for k, v in s.items() if k in ("scale", "loadings", "explained", "explained_multi", "n_multi", "ref_party", "fights_all", "fights_used")} for t, s in TSR["scales"].items()}
for t, s in TSR["scales"].items():
    if s["scale"]:
        topic_scales[t]["mps"] = [{"id": m, "mp": MPI[m][0], "party": MPI[m][1], "score": sc, "focus": TSR["focus"][t].get(m, 0)} for m, sc in s["mps"].items()]

# ---------- stats ----------
T = d["totals"]; fight_n = [f for f in fights if f["status"] == "fight"]
dn_f = [f for f in F if f.get("dense")]
chk = {c["passage_id"] for f in dn_f for k in ("mps_for", "mps_against", "mps_both_sides") for m in f[k] for c in m["claims"]} | {x["passage_id"] for f in dn_f for x in f["dropped"] if x["reason"] != "NOT_JUDGED(budget)"}
try:
    _fid = {f["fight_id"] for f in dn_f}
    _v = pd.concat([pd.read_json(ROOT / "data/quick/research_2026_11_10/fp" / n, lines=True)[["passage_id", "fight"]] for n in ("dense_sides.jsonl", "dense_sides2.jsonl")])
    _rj = set(pd.read_json(ROOT / "data/quick/research_2026_11_10/fp/dense_sides3.jsonl", lines=True).passage_id)
    _keep = {f["fight_id"] for f in dn_f if not f.get("rejudged")}
    chk = set(_v[_v.fight.isin(_keep)].passage_id) | _rj | chk
except Exception as e: print("checked fallback", e)
stats = {"speeches": T.get("passages_extracted", sum(v["passages"] for v in d["per_topic"].values())), "checked": len(chk), "dense_fights": len(dn_f),
         "topics": len(d["per_topic"]), "fights_real": len(fight_n),
         "mps_placed": M_["n_mps"], "claims": T.get("claims"), "fights_total": len(fights)}
payload = {"fights": fights, "P": P, "topics": d["per_topic"], "stats": stats, "scales": topic_scales, "conscience_topic": TS.CONSC,
           "no_scale": TSR["no_scale"], "excluded_small": TSR["excluded_small"], "coverage": TSR["coverage"], "map": mapd}
data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o)).replace("</", "<\\/")
html = (pathlib.Path(__file__).parent / "space_page_template.html").read_text().replace("__DATA__", data)
OUT.write_text(html); print(OUT, round(OUT.stat().st_size / 1e6, 2), "MB", "stats", stats)
