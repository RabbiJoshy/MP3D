"""Generic step B: group claims into issues by the POLICY they discuss (so both sides land together)."""
import sys, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
tag = sys.argv[1]
c = pd.read_json(f"{tag}_claims.jsonl", lines=True)
c = c[c.position.notna() & ~c.position.fillna("").str.upper().str.startswith("NONE") & c.policy.notna()].copy()
m = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True, device="mps")
X = m.encode(c.policy.tolist(), normalize_embeddings=True)
c["issue"] = AgglomerativeClustering(n_clusters=None, distance_threshold=0.3, metric="cosine", linkage="average").fit_predict(X)
sz = c.issue.value_counts(); keep = sz[sz >= int(sys.argv[2]) if len(sys.argv) > 2 else 8].index
c = c[c.issue.isin(keep)]
print(len(c), "claims in", len(keep), "issues with >=8 claims")
c.to_json(f"{tag}_issues.jsonl", orient="records", lines=True)
