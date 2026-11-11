"""Targeted sampling for the joint space. Generic, no hand-picked positions.
1. Anchor positions: cluster claims already extracted (all_claims = 300/topic random, permp = 5/MP random; HousingV4 excluded
   to avoid over-weighting the tuned topic). Keep clusters backed by >=3 distinct MPs, drop vague ones ('Supports the Bill'),
   cap at 12 per topic (largest first).
2. For every MP: score each of their substantive passages by similarity to the anchors; take the top K passages, each from a
   different anchor where possible, skipping passages already extracted.
Output: targeted_sample.jsonl (passages to extract) and anchors.json."""
import re, json, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
K, CAP, TH = 5, 12, 0.22
R = "/Users/joshuathomasamar/PycharmProjects/MP3D/"
VAGUE = re.compile(r"^(supports|opposes)\s+(the|this|that|these|his|her|their|our|my)?\s*(bill|motion|amendment|amendments|new clause|clause|government|budget|proposal|proposals|policy|measures?|statement|legislation|scheme|plan)s?\b\.?$", re.I)
c = pd.concat([pd.read_json("all_claims.jsonl", lines=True), pd.read_json("permp_claims.jsonl", lines=True)])
c = c[c.position.fillna("").str.match(r"^\s*(Supports|Opposes)\s", case=False) & ~c.position.fillna("").str.strip(" .").str.match(VAGUE)]
c = c[c.position.str.split().str.len() >= 4]
m = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True, device="mps")
E = m.encode(c.position.tolist(), normalize_embeddings=True, batch_size=64)
anchors = []
for topic in c.topic.unique():
    idx = np.where(c.topic.values == topic)[0]
    if len(idx) < 3: continue
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=TH, metric="cosine", linkage="average").fit_predict(E[idx])
    cands = []
    for L in np.unique(lab):
        mem = idx[lab == L]
        nmp = c.member_id.iloc[mem].nunique()
        if nmp < 3: continue
        cen = E[mem].mean(0); cen /= np.linalg.norm(cen)
        cands.append((nmp, c.position.iloc[mem[np.argmax(E[mem] @ cen)]], cen))
    for nmp, label, cen in sorted(cands, key=lambda x: -x[0])[:CAP]:
        anchors.append({"topic": topic, "label": label, "n_mps": int(nmp), "vec": cen})
print(len(anchors), "anchors across", len({a['topic'] for a in anchors}), "topics")
A = np.vstack([a["vec"] for a in anchors])
p = pd.read_parquet(R + "data/clean/passages.parquet", columns=["passage_id", "member_id", "major", "minor", "speech_type", "role", "text", "words"])
p["home"] = pd.read_parquet("homes.parquet").home.values
ok = (p.home != "No policy") & ~p.speech_type.str.contains("Chair", na=False) & (p.words >= 60)
done = set(c.passage_id) | set(pd.read_json("permp_claims.jsonl", lines=True).passage_id)
EP = np.load(R + "data/embed/passages.npy", mmap_mode="r")
mem = pd.read_parquet(R + "data/clean/members.parquet").set_index("member_id")
rows = []
for mid, g in p[ok & ~p.passage_id.isin(done)].groupby("member_id"):
    S = np.asarray(EP[g.index.values]) @ A.T          # passage x anchor similarity
    best_anchor, best_sim = S.argmax(1), S.max(1)
    order = np.argsort(-best_sim); used, picked = set(), []
    for i in order:                                    # one passage per anchor first, then fill
        if best_anchor[i] not in used: picked.append(i); used.add(best_anchor[i])
        if len(picked) == K: break
    for i in order:
        if len(picked) >= K: break
        if i not in picked: picked.append(i)
    for i in picked:
        r = g.iloc[i]
        rows.append({"passage_id": r.passage_id, "member_id": int(mid), "mp": mem.loc[mid, "name"] if mid in mem.index else "?",
                     "party": mem.loc[mid, "party_abbr"] if mid in mem.index else "?", "role": r.role, "debate": r.major or r.minor,
                     "topic": r.home, "text": r.text, "anchor": anchors[best_anchor[i]]["label"], "sim": round(float(best_sim[i]), 3)})
t = pd.DataFrame(rows)
t.to_json("targeted_sample.jsonl", orient="records", lines=True)
json.dump([{k: v for k, v in a.items() if k != "vec"} for a in anchors], open("anchors.json", "w"), indent=1)
print(len(t), "passages for", t.member_id.nunique(), "MPs; median similarity", t.sim.median())
print(t.anchor.value_counts().head(12).to_string())
