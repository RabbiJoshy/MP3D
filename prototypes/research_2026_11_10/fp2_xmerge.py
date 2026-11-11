"""fp2 step 6: cross-topic duplicate merge on new labels (5 nearest by bge; Jev Q1/Q2 both orders; >=.5 or >=.3 & shared debate; max 4 per component).
Direction is not needed: positions come only from dense scoring against the lead label. Writes data/.../fp2/units.json, results/fp2_xmerge.json"""
import re, json, pathlib, requests, numpy as np
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D"); R = ROOT / "prototypes/research_2026_11_10/results"; D = ROOT / "data/quick/research_2026_11_10"
F = {f["fight_id"]: f for f in json.load(open(D / "fp2/fights_pre.json"))}
L = {o["fight_id"]: o for o in json.load(open(R / "fp2_labels.json"))["labels"]}
ids = sorted(F); lab = {i: L[i]["new"] for i in ids}
deb = {i: {tuple(d) for d in F[i]["debates"]} for i in ids}
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True, device="mps")
E = m.encode([lab[i] for i in ids], normalize_embeddings=True); S = E @ E.T; np.fill_diagonal(S, -1)
pairs = {}
for a in range(len(ids)):
    for b in np.argsort(-S[a])[:5]:
        pairs[tuple(sorted((ids[a], ids[int(b)])))] = float(S[a, b])
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
rows = []
for (a, b), r in zip(PL, res):
    if None in r: continue
    rows.append(dict(a=a, b=b, label_a=lab[a], label_b=lab[b], cos=pairs[(a, b)], p1=(r[0][0] + r[1][0]) / 2, p2=(r[0][1] + r[1][1]) / 2, shared_debate=bool(deb[a] & deb[b])))
rows.sort(key=lambda x: -x["p1"])
comp = {i: {i} for i in ids}; merges = []
for r in rows:
    if not (r["p1"] >= .5 or (r["p1"] >= .3 and r["shared_debate"])): continue
    ca, cb = comp[r["a"]], comp[r["b"]]
    if ca is cb or len(ca) + len(cb) > 4: r["merged"] = False; continue
    u = ca | cb
    for i in u: comp[i] = u
    r["merged"] = True; merges.append(r)
seen = []; units = []
for i in ids:
    c = comp[i]
    if any(c is s for s in seen): continue
    seen.append(c); mem = sorted(c); lead = max(mem, key=lambda j: (F[j]["n_mps"], -j))
    units.append(dict(unit=len(units), members=mem, lead=lead, merged=len(mem) > 1, label=lab[lead], description=L[lead]["description"], for_means=L[lead]["for_means"], against_means=L[lead]["against_means"],
                      topic=F[lead]["topic"], topics=sorted({F[j]["topic"] for j in mem}), n_mps=len({c_["member_id"] for j in mem for c_ in F[j]["claims"]}),
                      debates=sorted({d for j in mem for d in deb[j]}, key=str), old_label=F[lead]["label"]))
json.dump(units, open(D / "fp2/units.json", "w"))
json.dump({"tokens": used[0], "usd": round(used[0] * 0.042 / 1e6, 5), "pairs": rows, "merges": merges, "n_units": len(units), "n_merged_units": sum(u["merged"] for u in units)}, open(R / "fp2_xmerge.json", "w"), indent=1)
print("tokens", used[0], "usd", round(used[0] * 0.042 / 1e6, 4), "merges", len(merges), "units", len(units))
for r in merges: print(r["a"], r["b"], f"p1={r['p1']:.2f}", lab[r["a"]][:55], "||", lab[r["b"]][:55])
