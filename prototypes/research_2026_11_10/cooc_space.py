"""Joint position+MP space from co-occurrence (no Jev, no propositions, no party).
1. Positions = clusters of claim text, clustered separately for 'Supports' and 'Opposes' claims (so opposites never merge).
2. MP x position counts -> PPMI (how surprising each link is, given the MP's volume and the position's popularity).
3. SVD -> joint coordinates for positions and MPs. python cooc_space.py HousingV4 [threshold] [min_pos_size] [min_mp_claims]"""
import sys, re, json, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
tag = sys.argv[1]; TH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.22
MINP = int(sys.argv[3]) if len(sys.argv) > 3 else 6; MINM = int(sys.argv[4]) if len(sys.argv) > 4 else 3
c = pd.read_json(f"{tag}_claims.jsonl", lines=True)
c = c[c.position.fillna("").str.match(r"^\s*(Supports|Opposes)\s", case=False)].copy()
VAGUE = re.compile(r"^(supports|opposes)\s+(the|this|that|these|his|her|their|our|my)?\s*(bill|motion|amendment|amendments|new clause|clause|government|budget|proposal|proposals|policy|measures?|statement|legislation|scheme|plan)s?\b\.?$", re.I)
c = c[~c.position.str.strip(" .").str.match(VAGUE) & (c.position.str.split().str.len() >= 4)]
c["sign"] = np.where(c.position.str.lower().str.startswith("opposes"), "-", "+")
m = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True, device="mps")
E = m.encode(c.position.tolist(), normalize_embeddings=True, batch_size=64)
c["pos"] = -1; labels = {}; nid = 0
for s in "+-":
    idx = np.where(c.sign.values == s)[0]
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=TH, metric="cosine", linkage="average").fit_predict(E[idx])
    for L in np.unique(lab):
        members = idx[lab == L]
        if len(members) < MINP: continue
        cen = E[members].mean(0); best = members[np.argmax(E[members] @ cen)]
        c.iloc[members, c.columns.get_loc("pos")] = nid; labels[nid] = c.position.iloc[best]; nid += 1
c = c[c.pos >= 0]
cnt = c.member_id.value_counts(); c = c[c.member_id.isin(cnt[cnt >= MINM].index)]
print(f"{len(labels)} positions ({sum(v.lower().startswith('opposes') for v in labels.values())} 'opposes'), {len(c)} claims, {c.member_id.nunique()} MPs")
C = pd.crosstab(c.member_id, c.pos).astype(float)
P = C / C.values.sum(); pm = P.sum(1).values[:, None]; pp = P.sum(0).values[None, :]
with np.errstate(divide="ignore"): PMI = np.log(P.values / (pm * pp))
X = np.maximum(PMI, 0); X[~np.isfinite(X)] = 0
X = X - X.mean(0)
U, S, Vt = np.linalg.svd(X, full_matrices=False); var = S**2 / (S**2).sum()
mpc = pd.DataFrame(U[:, :3] * np.sqrt(S[:3]), index=C.index, columns=["x", "y", "z"])
psc = pd.DataFrame(Vt[:3].T * np.sqrt(S[:3]), index=C.columns, columns=["x", "y", "z"]); psc["label"] = psc.index.map(labels); psc["topic"] = c.groupby("pos").topic.agg(lambda s: s.mode()[0]).reindex(psc.index).values if "topic" in c else ""
psc["n"] = c.pos.value_counts().reindex(psc.index).values
info = c.drop_duplicates("member_id").set_index("member_id")[["mp", "party", "role"]]
mpc = mpc.join(info)
for a, col in enumerate("xyz"):
    o = psc.sort_values(col)
    print(f"\nAXIS {a+1} ({var[a]:.0%})\n  [-] " + "\n      ".join(f"{l} ({n}, {t})" for l, n, t in zip(o.label.head(6), o.n.head(6), o.topic.head(6))))
    print("  [+] " + "\n      ".join(f"{l} ({n}, {t})" for l, n, t in zip(o.label.tail(6)[::-1], o.n.tail(6)[::-1], o.topic.tail(6)[::-1])))
    mo = mpc.sort_values(col)
    print("  MPs -:", ", ".join(f"{r.mp} ({r.party})" for r in mo.head(5).itertuples()))
    print("  MPs +:", ", ".join(f"{r.mp} ({r.party})" for r in mo.tail(5).itertuples()))
g = mpc.groupby("party")[["x", "y", "z"]].mean().round(2); n = mpc.party.value_counts()
print("\nparty means (>=3):\n", g[n.reindex(g.index) >= 3].to_string())
for a, col in enumerate("xyz"):
    gg = mpc.groupby("party")[col]; r2 = 1 - ((mpc[col] - gg.transform("mean"))**2).sum() / ((mpc[col] - mpc[col].mean())**2).sum()
    print(f"party R2 axis {a+1}: {r2:.2f}")
try:
    j = pd.DataFrame(json.load(open(f"{tag}_map2d.json"))["mps"]).set_index("member_id")
    k = mpc.join(j[["x", "y"]], rsuffix="_jev", how="inner")
    print(f"corr with Jev axis 1 on {len(k)} MPs:", [round(float(np.corrcoef(k[col], k.x_jev)[0, 1]), 2) for col in "xyz"])
except Exception as e: print(e)
json.dump({"var": var[:3].tolist(), "positions": psc.reset_index().round(3).to_dict("records"), "mps": mpc.reset_index().round(3).to_dict("records")},
          open(f"{tag}_cooc.json", "w"), indent=1, default=str)
