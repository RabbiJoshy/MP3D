"""Build prototypes/app2.html from a fights JSON (same schema as fp_fights_final3.json).
Usage: build_app2.py [fights.json]   Free and local. Optional fields per fight: for_means / against_means.
Axes + map come from research_2026_11_10/topic_space.py (imported, not edited)."""
import json, pathlib, re, sys, collections
import pandas as pd

ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
HERE = ROOT / "prototypes"
R = HERE / "research_2026_11_10" / "results"
INP = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else R / "fp_fights_final3.json"
OUT = HERE / "app2.html"
d = json.load(open(INP)); F = d["fights"]
print("input:", INP, len(F), "fights")

SK = ("mps_for", "mps_against", "mps_both_sides")
need = {c["passage_id"] for f in F for k in SK for m in f.get(k, []) for c in m["claims"]}
pas = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "date", "major", "minor", "text"])
pas = pas[pas.passage_id.isin(need)].set_index("passage_id")


def url(pid):
    m = re.match(r"uk\.org\.publicwhip/(debate|westminhall)/([^#]+)", pid)
    return f"https://www.theyworkforyou.com/{'debates' if m.group(1) == 'debate' else 'whall'}/?id={m.group(2)}" if m else None


W = lambda s: set(re.findall(r"[a-z]{4,}", s.lower()))


def excerpt(text, measure, n=300):
    t = " ".join(text.split())
    if len(t) <= n: return t
    sents = re.split(r"(?<=[.!?])\s+", t); mw = W(measure); best, bi = -1, 0
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
    if pid in P or pid not in pas.index: return
    r = pas.loc[pid]; t = r["text"] or ""
    P[pid] = {"date": str(r["date"])[:10], "debate": " — ".join(x for x in [r["major"], r["minor"]] if isinstance(x, str) and x),
              "ex": excerpt(t, measure), "full": t[:4000] + ("…" if len(t) > 4000 else ""), "url": url(pid)}


fights = []
for i, f in enumerate(F):
    def lst(key):
        out = []
        for m in f.get(key, []):
            cl = []
            for c in m["claims"]:
                pas_obj(c["passage_id"], c["measure"])
                if c["passage_id"] in P: cl.append(c["passage_id"])
            out.append({"id": m["member_id"], "mp": m["name"], "party": m["party"], "st": m.get("stance", 0), "n": int(m.get("focus") or len(cl)), "p": cl})
        return out
    fights.append({"i": i, "topic": f["topic"], "label": f["label"], "desc": f.get("description", ""), "status": f["status"],
                   "fm": f.get("for_means", ""), "am": f.get("against_means", ""),
                   "for": lst("mps_for"), "against": lst("mps_against")})

# ---- party colours
mem = pd.read_parquet(ROOT / "data/clean/members.parquet", columns=["member_id", "party_abbr", "party_colour"]).drop_duplicates("member_id")
pc = {}
for r in mem.itertuples():
    if r.party_abbr and r.party_abbr not in pc and isinstance(r.party_colour, str): pc[r.party_abbr] = "#" + r.party_colour
used = {m["party"] for f in fights for k in ("for", "against") for m in f[k]}
pcol = {p: pc.get(p, "#8a94a3") for p in used}

# ---- axes + map from the same computation as space.html
sys.path.insert(0, str(HERE / "research_2026_11_10"))
import topic_space as TS
TSR = TS.build(F)
M_ = TSR["map"]
axes = []
for k, a in enumerate(M_["axes"]):
    ld = a["loadings"]; mx = max(abs(l["v"]) for l in ld) or 1
    pos = [l["topic"] for l in ld if l["v"] > 0.25 * mx][:3]; neg = [l["topic"] for l in ld if l["v"] < -0.25 * mx][:3]
    top = [l["topic"] for l in ld[:3]]
    if pos and neg:
        txt = f"Mostly separates {', '.join(pos)} (the + end) from {', '.join(neg)} (the - end): MPs at the + end take the Labour side on the first group and the opposite side on the second."
    else:
        txt = f"Mostly one shared direction across {', '.join(top)}: MPs at the + end sit on the Labour side of these topics, MPs at the - end on the other side."
    axes.append({"spread": a["spread"], "party": a["party_eta2"], "text": txt, "loadings": ld})
mp_xyz = {m["id"]: m["xyz"] for m in M_["mps"]}
space = {"axes": axes, "tnames": TSR["tnames"], "n_mps": M_["n_mps"], "xyz": mp_xyz,
         "party_mean": {p: None for p in used}}
import numpy as np
for p in used:
    v = [mp_xyz[m["id"]] for f in fights for k in ("for", "against") for m in f[k] if m["party"] == p and m["id"] in mp_xyz]
    # unique by id
    seen = {}
    for f in fights:
        for k in ("for", "against"):
            for m in f[k]:
                if m["party"] == p and m["id"] in mp_xyz: seen[m["id"]] = mp_xyz[m["id"]]
    space["party_mean"][p] = np.mean(list(seen.values()), 0).round(3).tolist() if len(seen) >= 3 else None

stat = collections.Counter(f["status"] for f in fights)
mps = {}
for f in fights:
    for k in ("for", "against"): 
        for m in f[k]: mps.setdefault(m["id"], (m["mp"], m["party"]))
T = d["totals"]
stats = {"fights": len(fights), "by_status": dict(stat), "mps": len(mps), "speeches": T.get("passages_extracted"), "claims": T.get("claims"),
         "topics": len({f["topic"] for f in fights}), "all_topics": sorted(d["per_topic"]), "topic_stats": d["per_topic"]}
payload = {"fights": fights, "P": P, "pcol": pcol, "space": space, "stats": stats}
data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o)).replace("</", "<\\/")
html = (HERE / "app2_template.html").read_text().replace("__DATA__", data)
OUT.write_text(html)
print(OUT, round(OUT.stat().st_size / 1e6, 2), "MB", "status", dict(stat), "mps positioned", len(mps), "on map", M_["n_mps"], "passages", len(P))
