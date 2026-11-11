"""Build the fuller topic space and export contested MP speech evidence."""
import json, pathlib, re, sys, collections, xml.etree.ElementTree as ET
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "prototypes/research_2026_11_10/results"
OUT = ROOT / "prototypes/space_full.html"


def export_evidence(fights, passages, source, output):
    """Keep the original fight indices and resolve full contributions from local Hansard."""
    contested = [fight for fight in fights if fight["status"] == "fight"]
    ids = {
        claim["p"]
        for fight in contested
        for side in ("for", "against", "both")
        for member in fight[side]
        for claim in member["claims"]
    } | {drop["p"] for fight in contested for drop in fight["dropped"]}
    wanted = collections.defaultdict(set)
    for pid in ids:
        row = source.loc[pid]
        wanted[row["source"], str(row["date"])[:10]].add(pid.rsplit("#", 1)[0])
    originals = {}
    for (kind, date), speech_ids in sorted(wanted.items()):
        for path in sorted((ROOT / "data/raw/hansard" / kind).glob(f"*{date}*.xml")):
            for speech in ET.parse(path).getroot().iter("speech"):
                sid = speech.get("id")
                if sid in speech_ids:
                    paragraphs = [" ".join("".join(p.itertext()).split()) for p in speech.findall("p")]
                    originals[sid] = "\n\n".join(p for p in paragraphs if p)
            if speech_ids.issubset(originals):
                break

    evidence = {}
    fallback = 0
    for pid in sorted(ids):
        full = originals.get(pid.rsplit("#", 1)[0])
        if not full:
            full = source.at[pid, "text"]
            fallback += 1
        evidence[pid] = dict(passages[pid], full=full)
    output.write_text(json.dumps({"fights": contested, "P": evidence}, ensure_ascii=False, separators=(",", ":")))
    print(output, len(contested), "contested fights;", len(evidence), "passages;", fallback, "used cleaned text", flush=True)


def export_pilot_evidence():
    pilot_html = (ROOT / "prototypes/space.html").read_text()
    start = '<script id="d" type="application/json">'
    payload = json.loads(pilot_html.split(start, 1)[1].split("</script>", 1)[0])
    source = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "source", "date", "text"])
    source = source[source.passage_id.isin(payload["P"])].drop_duplicates("passage_id").set_index("passage_id")
    export_evidence(payload["fights"], payload["P"], source, R / "fp_mp_evidence.json")


if len(sys.argv) > 1: INP = pathlib.Path(sys.argv[1])
else: INP = R / "fp2_fights.json"
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
pas = pas[pas.passage_id.isin(need)].drop_duplicates("passage_id").set_index("passage_id")
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
        if pd.isna(nm): nm = "Unknown MP"
        if pd.isna(pt): pt = "?"
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
json.dump(HO, open(R / "fp2_topic_holdout.json", "w"), indent=1)
json.dump(TSR, open(R / "fp2_topic_space.json", "w"), indent=1)
for i, f in enumerate(fights):
    fv = TSR["free_vote"][i]; f["conscience"] = fv["flag"]; f["fv"] = fv["parties"]
export_evidence(fights, P, pas, R / "fp2_mp_evidence.json")
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
chk = {c["passage_id"] for f in dn_f for k in ("mps_for", "mps_against", "mps_both_sides") for m in f[k] for c in m["claims"]} | {x["passage_id"] for f in dn_f for x in f["dropped"]}
stats = {"speeches": T.get("passages_extracted", sum(v["passages"] for v in d["per_topic"].values())), "checked": T.get("passages_judged", len(chk)), "dense_fights": len(dn_f),
         "topics": len(d["per_topic"]), "fights_real": len(fight_n),
         "mps_placed": M_["n_mps"], "claims": T.get("claims"), "fights_total": len(fights)}
payload = {"fights": fights, "P": P, "topics": d["per_topic"], "stats": stats, "scales": topic_scales, "conscience_topic": TS.CONSC,
           "no_scale": TSR["no_scale"], "excluded_small": TSR["excluded_small"], "coverage": TSR["coverage"], "map": mapd}
data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o)).replace("</", "<\\/")
html = (pathlib.Path(__file__).parent / "space_page_template.html").read_text().replace("__DATA__", data)
html = html.replace("First pass — about 10,000 speeches from the biggest contested debates;", "Fuller run — about 22,500 speeches from the biggest debates of every topic;")
OUT.write_text(html); print(OUT, round(OUT.stat().st_size / 1e6, 2), "MB", "stats", stats)
