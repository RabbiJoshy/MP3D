"""Group topics into regions using two signals, then lay out regions and topics in 3D.

Two topics are related if they use similar words, or if the same MPs talk about both.
Regions: Ward clustering on both signals. Layout: one MDS across all topics, so distances are comparable
everywhere on the map. Regions are colours on that map, so they can overlap. Out: data/quick/topic_layout.npz
"""
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import numpy as np
from scipy.stats import spearmanr
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
# One shared space: every topic is placed relative to every other topic, so nearer generally means more related.
layout = MDS(3, dissimilarity="precomputed", random_state=SEED, n_init=8, normalized_stress="auto").fit_transform(dist)
layout = layout - np.median(layout, axis=0)
layout = layout / np.percentile(np.linalg.norm(layout, axis=1), 95) * 15
topic_pos = np.zeros((len(centres), 3))
topic_pos[real] = layout
reg_xyz = np.array([layout[labels == r].mean(0) for r in reg_ids])

iu = np.triu_indices(len(real), 1)
same = labels[iu[0]] == labels[iu[1]]
mapped = np.linalg.norm(layout[:, None] - layout[None], axis=2)
print(f"map distance vs real similarity: all {spearmanr(mapped[iu], dist[iu])[0]:.2f}, "
      f"same region {spearmanr(mapped[iu][same], dist[iu][same])[0]:.2f}, across regions {spearmanr(mapped[iu][~same], dist[iu][~same])[0]:.2f}")
nearest = (mapped + np.eye(len(real)) * 1e9).argmin(1)
print(f"nearest topic is in the same region for {(labels[nearest] == labels).mean():.0%} of topics; "
      f"closest pair distance {np.min(mapped[iu]):.2f}, median nearest-topic distance {np.median((mapped + np.eye(len(real)) * 1e9).min(1)):.2f}")

region = np.full(len(centres), -1)
region[real] = labels
np.savez(ROOT / "data/quick/topic_layout.npz", topic_pos=topic_pos, region=region, reg_xyz=reg_xyz)
for r in reg_ids:
    print(f"\nREGION {r}")
    for k in real[labels == r]:
        print(f"  {k:2d} {', '.join(top_terms[k][:5])}")
