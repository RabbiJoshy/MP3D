"""Retrieve top-200 BGE-matched passages for 12 high-contrast sk19 causes and score with strict jev-1.14.0 judge."""
import json, pathlib, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
sys.path.insert(0, str(ROOT / "prototypes/research_2026_11_10"))
import fp_dense2_run as J

OUT = pathlib.Path(__file__).resolve().parent
CAUSES = [
    {"cid": 10, "label": "Increase defence spending", "subj": "Defence", "fm": "Supports higher defence spending", "am": "Opposes or questions higher defence spending"},
    {"cid": 15, "label": "Cut the welfare budget", "subj": "Social welfare", "fm": "Supports reducing welfare spending", "am": "Opposes welfare budget cuts"},
    {"cid": 16, "label": "Implement rent controls", "subj": "Housing", "fm": "Supports rent controls", "am": "Opposes rent controls"},
    {"cid": 19, "label": "Introduce a wealth tax and tax excess corporate profits", "subj": "Macroeconomics", "fm": "Supports a wealth tax or higher taxes on excess profits", "am": "Opposes a wealth tax or higher corporate taxes"},
    {"cid": 24, "label": "Approve new North Sea oil and gas drilling projects", "subj": "Energy", "fm": "Supports new North Sea oil and gas drilling", "am": "Opposes new North Sea oil and gas drilling"},
    {"cid": 25, "label": "Allow asylum seekers to work", "subj": "Immigration", "fm": "Supports allowing asylum seekers to work", "am": "Opposes allowing asylum seekers to work"},
    {"cid": 26, "label": "Create safe and legal routes for asylum seekers", "subj": "Immigration", "fm": "Supports safe and legal asylum routes", "am": "Opposes expanding safe and legal asylum routes"},
    {"cid": 28, "label": "Implement a digital ID system", "subj": "Civil rights", "fm": "Supports a national digital ID system", "am": "Opposes a national digital ID system"},
    {"cid": 29, "label": "Disapply the Human Rights Act in immigration cases", "subj": "Civil rights", "fm": "Supports disapplying the Human Rights Act in immigration cases", "am": "Opposes disapplying the Human Rights Act"},
    {"cid": 31, "label": "Create a defence pact with the European Union", "subj": "Defence", "fm": "Supports a UK-EU defence pact", "am": "Opposes a UK-EU defence pact"},
    {"cid": 34, "label": "Ban harmful social media for under-16s", "subj": "Culture", "fm": "Supports banning or restricting social media for under-16s", "am": "Opposes a social media ban for under-16s"},
    {"cid": 40, "label": "Reduce green-belt protections to build more homes", "subj": "Housing", "fm": "Supports building on green-belt land", "am": "Opposes reducing green-belt protections"},
]

D_html = json.loads(re.search(r'<script id="d" type="application/json">(.*?)</script>', (ROOT / "docs/index.html").read_text(), re.S)[1])
app_mids = {m["id"] for m in D_html["mps"]}

print("Loading passages...", flush=True)
pas = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "member_id", "date", "major", "minor", "words", "text"])
pas = pas[pas.member_id.isin(app_mids) & (pas.words >= 45)].reset_index(drop=True)
print(f"Eligible passages: {len(pas)}", flush=True)

bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu", local_files_only=True)
# Pre-filter by keyword regex per cause to make BGE encoding super fast (~10s instead of encoding 200k passages)
KW = {
    10: r"\bdefence spend|2\.5%|3% of GDP|military spend",
    15: r"\bwelfare|benefit|universal credit|disability|PIP\b",
    16: r"\brent control|cap on rent|rental cap|private rent",
    19: r"\bwealth tax|capital gains|excess profit|windfall|super-rich",
    24: r"\bRosebank|North Sea|oil and gas|drilling|new licences",
    25: r"\basylum seeker.{0,30}work|right to work.{0,30}asylum",
    26: r"\bsafe and legal route|safe route.{0,30}asylum|small boat",
    28: r"\bdigital ID|digital identity|ID card",
    29: r"\bHuman Rights Act|ECHR|European Convention on Human Rights",
    31: r"\bEU defence|European.{0,20}defence pact|security pact.{0,20}EU",
    34: r"\bsocial media.{0,30}under.?16|smartphone.{0,20}ban|online safety.{0,20}children",
    40: r"\bgreen belt|greenbelt|grey belt",
}

cands = []
for c in CAUSES:
    sub = pas[pas.text.str.contains(KW[c["cid"]], case=False, regex=True, na=False)]
    if len(sub) > 1500:
        sub = sub.sample(1500, random_state=42)
    q_emb = bge.encode([c["label"]], normalize_embeddings=True, show_progress_bar=False)[0]
    p_emb = bge.encode(sub.text.str[:800].tolist(), normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    sims = p_emb @ q_emb
    top_ix = np.argsort(-sims)[:200]
    for ix in top_ix:
        row = sub.iloc[ix]
        cands.append({"cid": c["cid"], "label": c["label"], "passage_id": row["passage_id"], "member_id": int(row["member_id"]), "text": row["text"], "sim": float(sims[ix])})
    print(f"Cause {c['cid']} ({c['label'][:40]}): {len(sub)} kw matches -> top {len(top_ix)} (max sim {sims[top_ix[0]]:.3f})", flush=True)

print(f"Total candidates to judge: {len(cands)} (est ~${len(cands)*500*0.042/1e6:.3f})", flush=True)
lock = threading.Lock()
results = []
done_n = [0]

def run_one(item):
    f, a = J.ask({"label": item["label"], "text": item["text"]})
    with lock:
        done_n[0] += 1
        if done_n[0] % 300 == 0:
            print(f"Judged {done_n[0]}/{len(cands)} | tokens {J.used[0]} (${J.used[0]*0.042/1e6:.4f})", flush=True)
        if f is not None and ((f >= 0.8 and a <= 0.2) or (a >= 0.8 and f <= 0.2)):
            side = "FOR" if f > a else "AGAINST"
            results.append({"cid": item["cid"], "passage_id": item["passage_id"], "member_id": item["member_id"], "side": side, "p_for": f, "p_against": a})

with ThreadPoolExecutor(12) as ex:
    list(ex.map(run_one, cands))

cost = J.used[0] * 0.042 / 1e6
print(f"Done! Tokens: {J.used[0]}, Cost: ${cost:.4f}, Strict directional verdicts: {len(results)}", flush=True)

by_c = {c["cid"]: [] for c in CAUSES}
for r in results:
    by_c[r["cid"]].append(r)

kept_causes = []
for c in CAUSES:
    rs = by_c[c["cid"]]
    by_mp = {}
    for r in rs:
        by_mp.setdefault(r["member_id"], []).append(r)
    mps_for, mps_ag, mps_both = [], [], []
    for mid, mrs in by_mp.items():
        net = sum(1 if x["side"] == "FOR" else -1 for x in mrs)
        rec = {"id": mid, "stance": round(net / len(mrs), 3), "claims": [{"p": x["passage_id"], "s": x["side"]} for x in mrs]}
        if net > 0: mps_for.append(rec)
        elif net < 0: mps_ag.append(rec)
        else: mps_both.append(rec)
    status = "two_sided" if len(mps_for) >= 2 and len(mps_ag) >= 2 else "other"
    print(f"  {c['cid']:2d} | {status:9s} | FOR={len(mps_for):2d} AG={len(mps_ag):2d} | {c['label']}")
    if status == "two_sided":
        kept_causes.append({**c, "for": mps_for, "against": mps_ag, "both": mps_both})

(OUT / "causes_scored.json").write_text(json.dumps({"cost_usd": cost, "tokens": J.used[0], "causes": kept_causes}, indent=1))
print(f"Saved {len(kept_causes)} two-sided causes to {OUT / 'causes_scored.json'}")
