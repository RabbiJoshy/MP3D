"""Freeze 30 same-debate pairs. python pair_select.py"""
import json, random, itertools, collections, pathlib, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D"); R = ROOT / "prototypes/research_2026_11_10/results"
c = pd.read_json(ROOT / "data/quick/research_2026_11_10/edu2_claims.jsonl", lines=True)
c = c[c.measure.notna() & ~c.measure.str.upper().str.startswith("NONE")].drop_duplicates("passage_id")
p = pd.read_parquet(ROOT / "data/clean/passages.parquet")[["passage_id", "date", "major", "minor"]]
c = c.merge(p, on="passage_id", how="left")
c = c[c.text.str.split().str.len() >= 60].copy()
c["dkey"] = c.date.astype(str) + "|" + c.major.fillna("").astype(str) + "|" + c.minor.fillna("").astype(str)
m = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True, device="mps")
E = m.encode(c.measure.tolist(), normalize_embeddings=True, batch_size=64)
c = c.reset_index(drop=True); pairs = []
for k, idx in c.groupby("dkey").indices.items():
    for i, j in itertools.combinations(idx, 2):
        if c.member_id[i] != c.member_id[j]: pairs.append((i, j, float(E[i] @ E[j])))
sims = np.array([x[2] for x in pairs]); thr = float(np.quantile(sims, 0.9))
print("universe", len(pairs), "near threshold", round(thr, 4))
near = [x for x in pairs if x[2] >= thr]; rest = [x for x in pairs if x[2] < thr]
nd = [x for x in near if c.party[x[0]] != c.party[x[1]]]; ns = [x for x in near if c.party[x[0]] == c.party[x[1]]]
rng = random.Random(20261110); dc = collections.Counter(); mc = collections.Counter(); used = set(); chosen = []
def take(pool, n, stratum):
    pool = list(pool); rng.shuffle(pool); got = 0
    for i, j, s in pool:
        if got == n: break
        if (i, j) in used: continue
        d = c.dkey[i]; a, b = c.member_id[i], c.member_id[j]
        if dc[d] >= 2 or mc[a] >= 3 or mc[b] >= 3: continue
        dc[d] += 1; mc[a] += 1; mc[b] += 1; used.add((i, j)); chosen.append((i, j, s, stratum)); got += 1
    print(stratum, got)
take(nd, 10, "near-diff"); take(ns, 10, "near-same"); take(rest, 10, "random")
rng.shuffle(chosen); mem = pd.read_parquet(ROOT / "data/clean/members.parquet")
side = lambda i: {"passage_id": c.passage_id[i], "mp": c.mp[i], "party": c.party[i], "date": str(c.date[i])[:10],
                  "debate": f"{c.major[i]} / {c.minor[i]}", "text": c.text[i][:3000]}
fr, key = [], []
for n, (i, j, s, st) in enumerate(chosen, 1):
    fr.append({"pair_id": n, "a": side(i), "b": side(j)})
    key.append({"pair_id": n, "stratum": st, "similarity": s, "a_measure": c.measure[i], "a_side": c.stance[i],
                "b_measure": c.measure[j], "b_side": c.stance[j]})
json.dump(fr, open(R / "pairs_frozen.json", "w"), indent=1, ensure_ascii=False)
json.dump(key, open(R / "pairs_key.json", "w"), indent=1, ensure_ascii=False)
print("debates", len({c.dkey[i] for i, _, _, _ in chosen}), "MPs", len({c.member_id[k] for i, j, _, _ in chosen for k in (i, j)}))
