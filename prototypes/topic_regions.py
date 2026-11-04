"""Group topics into regions using two signals, then lay out regions and topics in 3D.

Two topics are related if they use similar words, or if the same MPs talk about both.
Regions: Ward clustering on both signals. Layout: MDS between regions, then MDS inside each region,
so a neighbourhood always stays together. Out: data/quick/topic_layout.npz
"""
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import MDS
from sklearn.preprocessing import normalize

ROOT = pathlib.Path(__file__).resolve().parent.parent
K_REGIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 11
SEED = 7
q = np.load(ROOT / "data/quick/topic_space.npz", allow_pickle=True)
centres, is_chatter, counts, top_terms = q["centres"], q["is_chatter"], q["counts"], q["top_terms"]
real = np.where(~is_chatter)[0]

# Who talks about each topic, compared with what you'd expect from how much they speak overall
c = counts[:, real] + 0.5
c = c[counts[:, real].sum(1) > 3000]
pmi = np.maximum(np.log(c * c.sum() / (c.sum(1, keepdims=True) * c.sum(0, keepdims=True))), 0)
people = normalize(TruncatedSVD(25, random_state=SEED).fit_transform(pmi.T))
words = normalize(TruncatedSVD(25, random_state=SEED).fit_transform(centres[real]))
features = np.hstack([words, people])

labels = AgglomerativeClustering(K_REGIONS, linkage="ward").fit_predict(features)
sim = features @ features.T / 2
dist = np.clip(1 - sim, 0, None)
np.fill_diagonal(dist, 0)

reg_ids = sorted(set(labels))
reg_dist = np.array([[dist[np.ix_(labels == a, labels == b)].mean() for b in reg_ids] for a in reg_ids])
np.fill_diagonal(reg_dist, 0)
reg_xyz = MDS(3, dissimilarity="precomputed", random_state=SEED, n_init=8, normalized_stress="auto").fit_transform(reg_dist)
reg_xyz = reg_xyz / np.percentile(np.linalg.norm(reg_xyz - reg_xyz.mean(0), axis=1), 90) * 13

topic_pos = np.zeros((len(centres), 3))
for r in reg_ids:
    idx = np.where(labels == r)[0]
    if len(idx) == 1:
        local = np.zeros((1, 3))
    else:
        d = dist[np.ix_(idx, idx)]
        local = MDS(3, dissimilarity="precomputed", random_state=SEED, n_init=8, normalized_stress="auto").fit_transform(d)
        local = local - local.mean(0)
        local = local / max(np.linalg.norm(local, axis=1).max(), 1e-6)
    radius = 1.3 + 0.55 * np.sqrt(len(idx))
    topic_pos[real[idx]] = reg_xyz[r] + local * radius

region = np.full(len(centres), -1)
region[real] = labels
np.savez(ROOT / "data/quick/topic_layout.npz", topic_pos=topic_pos, region=region, reg_xyz=reg_xyz)
for r in reg_ids:
    print(f"\nREGION {r}")
    for k in real[labels == r]:
        print(f"  {k:2d} {', '.join(top_terms[k][:5])}")
