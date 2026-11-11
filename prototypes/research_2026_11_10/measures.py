"""Generic: group canonical MEASURES (bge, cosine threshold), label each group by its most central measure,
count distinct MPs FOR and AGAINST, keep groups with >= K on each side. python measures.py edu 0.25 2"""
import sys, json, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
tag, TH, K = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
c = pd.read_json(f"{tag}_claims.jsonl", lines=True)
c = c[c.measure.notna() & ~c.measure.str.upper().str.startswith("NONE") & c.stance.isin(["FOR", "AGAINST"])].copy()
m = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True, device="mps")
E = m.encode(c.measure.tolist(), normalize_embeddings=True, batch_size=64)
c["g"] = AgglomerativeClustering(n_clusters=None, distance_threshold=TH, metric="cosine", linkage="average").fit_predict(E)
lab = {}
for g, idx in c.groupby("g").indices.items():
    cen = E[idx].mean(0); lab[g] = c.measure.iloc[idx[np.argmax(E[idx] @ cen)]]
c["label"] = c.g.map(lab)
s = c.groupby("g").apply(lambda x: pd.Series({"label": lab[x.name], "for": x[x.stance == "FOR"].member_id.nunique(),
                                               "against": x[x.stance == "AGAINST"].member_id.nunique(), "claims": len(x)}))
print(f"{len(c)} claims with a measure; {len(s)} measure groups; {(s['for']+s['against']>=3).sum()} with >=3 MPs")
con = s[(s["for"] >= K) & (s["against"] >= K)].sort_values("claims", ascending=False)
print(f"\nCONTESTED (>= {K} MPs each side): {len(con)}")
print(con.to_string())
print("\nBIGGEST ONE-SIDED:")
print(s[~s.index.isin(con.index)].sort_values("claims", ascending=False).head(8).to_string())
c.to_json(f"{tag}_measures.jsonl", orient="records", lines=True); con.to_json(f"{tag}_contested.json")
