"""Step 2: cross-topic duplicate detection on the new labels (bge neighbours + Jev Q1/Q2 in both orders)."""
import re, json, pathlib, requests, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D"); R = ROOT / "prototypes/research_2026_11_10/results"
F = {f["fight_id"]: f for f in json.load(open(R / "fp_fights_final2.json"))["fights"]}
L = {o["fight_id"]: o for o in json.load(open(R / "fp_clean_labels.json"))["labels"]}
ids = sorted(F); lab = {i: L[i]["new"] for i in ids}
p = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "date", "major"]).set_index("passage_id")
def claims(f): return [c for k in ("mps_for", "mps_against", "mps_both_sides") for m in f[k] for c in m["claims"]]
deb = {i: set(zip(p.date.astype(str)[[c["passage_id"] for c in claims(F[i]) if c["passage_id"] in p.index]], p.major[[c["passage_id"] for c in claims(F[i]) if c["passage_id"] in p.index]])) for i in ids}
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True, device="mps")
E = m.encode([lab[i] for i in ids], normalize_embeddings=True); S = E @ E.T; np.fill_diagonal(S, -1)
pairs = {}
for a in range(len(ids)):
    for b in np.argsort(-S[a])[:5]:
        k = tuple(sorted((ids[a], ids[int(b)]))); pairs[k] = float(S[a, b])
Q1 = "Measure 1: {a}. Measure 2: {b}. Are these two measures about the same specific policy decision, so that MPs arguing about one are taking part in the same political fight as MPs arguing about the other? Answer yes even if one is phrased as the opposite of the other or uses different words for the same policy."
Q2 = "Measure 1: {a}. Measure 2: {b}. Would an MP who wants Measure 1 to happen also want Measure 2 to happen?"
key = next(mm[1].strip().strip("\"'") for l in (ROOT / ".env").read_text().splitlines() if (mm := re.match(r"\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)", l)))
Sx = requests.Session(); Sx.headers.update({"Authorization": f"Bearer {key}"}); used = [0]
def ask(pr):
    res = []
    for x, y in (pr, pr[::-1]):
        q = {"q1": {"type": "noul", "instructions": Q1.format(a=lab[x], b=lab[y])}, "q2": {"type": "noul", "instructions": Q2.format(a=lab[x], b=lab[y])}}
        r = None
        for _ in range(6):
            try:
                z = Sx.post("https://api.typesafe.ai/v1/systemone", json={"model": "jev-1.14.0", "state": "Two policy measures are compared.", "questions": q}, timeout=60)
                if z.status_code == 200:
                    d = z.json(); used[0] += d["usage"]["input_tokens"]; r = (d["answers"]["q1"]["noul"], d["answers"]["q2"]["noul"]); break
            except requests.RequestException: pass
        res.append(r)
    return res
PL = sorted(pairs); print("pairs", len(PL))
with ThreadPoolExecutor(8) as ex: res = list(ex.map(ask, PL))
def stances(f): return {m["member_id"]: (1 if m["stance"] > 0 else -1) for k in ("mps_for", "mps_against") for m in f[k]}
rows = []
for (a, b), r in zip(PL, res):
    if None in r: continue
    sa, sb = stances(F[a]), stances(F[b]); com = set(sa) & set(sb)
    ag = sum(sa[x] == sb[x] for x in com)
    rows.append(dict(a=a, b=b, label_a=lab[a], label_b=lab[b], cos=pairs[(a, b)], p1=(r[0][0] + r[1][0]) / 2, p2=(r[0][1] + r[1][1]) / 2,
                     shared_debate=bool(deb[a] & deb[b]), common_mps=len(com), agree=ag, disagree=len(com) - ag))
rows.sort(key=lambda x: -x["p1"])
comp = {i: {i} for i in ids}; merges = []
for r in rows:
    if not (r["p1"] >= .5 or (r["p1"] >= .3 and r["shared_debate"])): continue
    ca, cb = comp[r["a"]], comp[r["b"]]
    if ca is cb or len(ca) + len(cb) > 4: r["merged"] = False; continue
    u = ca | cb
    for i in u: comp[i] = u
    r["merged"] = True
    r["direction"] = (1 if r["agree"] >= r["disagree"] else -1) if r["common_mps"] >= 3 else (1 if r["p2"] >= .5 else -1)
    r["direction_source"] = "overlap" if r["common_mps"] >= 3 else "jev_q2"
    merges.append(r)
seen = []; comps = []
for i in ids:
    if comp[i] not in seen and len(comp[i]) > 1: seen.append(comp[i]); comps.append(sorted(comp[i]))
json.dump({"tokens": used[0], "usd": round(used[0] * 0.042 / 1e6, 5), "pairs": rows, "merges": merges, "components": comps}, open(R / "fp_clean_merges.json", "w"), indent=1)
print("tokens", used[0], "usd", round(used[0] * 0.042 / 1e6, 4), "merges", len(merges), "components", comps)
for r in merges: print(r["a"], r["b"], f"p1={r['p1']:.2f} dir={r['direction']}({r['direction_source']},{r['agree']}/{r['disagree']}) ", lab[r["a"]][:50], "||", lab[r["b"]][:50])
