"""Remove noise directions from the passage vectors, build candidate MP vectors and score them.

The vector step is the "subtract what overlaps" idea, done so it can't delete the subject matter:
  * Differences WITHIN a debate can't be about the debate's subject. So parliamentary register
    (minister vs backbencher, question vs answer, chamber vs committee) is learned from those
    differences and its directions are projected out of every vector.

Candidate MP vectors:
  raw       plain average of the original vectors, as a baseline
  cleaned   average after removing register directions
  residual  average of (passage minus its debate's mean): how an MP talks given the subject
  topics    which topic clusters the MP over-uses (PPMI over k-means clusters)

Scores:
  self      split each MP's passages in two; how often half A's nearest half B is the same MP.
            This directly tests whether there's enough text to place an MP specifically.
  party     share of an MP's 10 nearest neighbours in the same party
  role      accuracy at telling ministers from others (lower = less driven by job title)
"""
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLEAN, EMB = ROOT / "data/clean", ROOT / "data/embed"
MIN_PASSAGES, K_TOPICS, SEED = 30, 150, 7
rng = np.random.default_rng(SEED)


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


df = pd.read_parquet(CLEAN / "passages.parquet")
V = np.load(EMB / "passages.npy", mmap_mode="r")
assert len(df) == len(V)
V = np.asarray(V, dtype="float32")
global_mean = V.mean(0)

df["debate"] = df.date + "|" + df.source + "|" + df.major.fillna("") + "|" + df.minor.fillna("")
debate_idx, _ = pd.factorize(df.debate)
debate_means = np.zeros((debate_idx.max() + 1, V.shape[1]), "float32")
np.add.at(debate_means, debate_idx, V)
debate_counts = np.bincount(debate_idx)
debate_means /= debate_counts[:, None]
speakers = df.groupby(debate_idx).member_id.nunique().to_numpy()
residual = V - debate_means[debate_idx]
usable = speakers[debate_idx] >= 3  # a debate with one or two voices has no meaningful "mean"
print(f"passages in debates with 3+ speakers: {usable.mean():.1%}", flush=True)

# Register labels, learned on within-debate differences so that subject can't leak in
labels = {
    "minister": (df.role == "minister").to_numpy(),
    "question": df.speech_type.fillna("").str.contains("Question").to_numpy(),
    "committee": (df.source == "standing").to_numpy(),
    "westminster_hall": (df.source == "westminhall").to_numpy(),
}
P = np.eye(V.shape[1], dtype="float32")
sample = rng.choice(np.where(usable)[0], size=min(60000, usable.sum()), replace=False)
report = {"register_accuracy": {}}
removed = 0
for name, y in labels.items():
    y_s = y[sample]
    if y_s.mean() < 0.01:
        continue
    for it in range(4):
        X = residual[sample] @ P
        clf = LogisticRegression(max_iter=300, class_weight="balanced", C=1.0)
        acc = cross_val_score(clf, X[:20000], y_s[:20000], cv=3, scoring="balanced_accuracy").mean()
        report["register_accuracy"].setdefault(name, []).append(round(float(acc), 3))
        if acc < 0.58:
            break
        w = clf.fit(X, y_s).coef_[0]
        w = w / np.linalg.norm(w)
        P = P @ (np.eye(len(w), dtype="float32") - np.outer(w, w))
        removed += 1
    print(name, report["register_accuracy"][name], flush=True)
report["directions_removed"] = removed
np.save(EMB / "noise_projection.npy", P)

C = (V - global_mean) @ P          # cleaned passage vectors
R = residual @ P                   # cleaned within-debate residuals
np.save(EMB / "passages_clean.npy", C)

km = MiniBatchKMeans(K_TOPICS, batch_size=8192, n_init=5, random_state=SEED, max_iter=200)
km.fit(norm(C[rng.choice(len(C), 150000, replace=False)]))
topic = km.predict(norm(C))
df["topic"] = topic
np.save(EMB / "topic_centres.npy", km.cluster_centers_)
df[["passage_id", "member_id", "topic"]].to_parquet(EMB / "passage_topics.parquet", index=False)

members = pd.read_parquet(CLEAN / "members.parquet").set_index("member_id")
counts = df.member_id.value_counts()
ids = counts.index[counts >= MIN_PASSAGES].to_numpy()
pos = {m: i for i, m in enumerate(ids)}
owner = df.member_id.map(pos).to_numpy()
keep = ~np.isnan(owner.astype(float))
report["mps_scored"], report["mps_excluded_too_little_text"] = len(ids), int(df.member_id.nunique() - len(ids))


def ppmi(matrix):
    total = matrix.sum()
    expected = matrix.sum(1, keepdims=True) * matrix.sum(0, keepdims=True) / total
    return np.maximum(np.log(np.maximum(matrix, 1e-9) / np.maximum(expected, 1e-9)), 0)


def mp_vectors(kind, mask):
    idx = owner[mask].astype(int)
    if kind == "topics":
        m = np.zeros((len(ids), K_TOPICS), "float32")
        np.add.at(m, (idx, topic[mask]), 1)
        return norm(ppmi(m + 0.01))
    source = {"raw": V - global_mean, "cleaned": C, "residual": R}[kind]
    m = np.zeros((len(ids), V.shape[1]), "float32")
    np.add.at(m, idx, source[mask])
    return norm(m - m.mean(0))


split = rng.random(len(df)) < 0.5
party = members.party_abbr.reindex(ids).fillna("?").to_numpy()
report["candidates"] = {}
out = {}
words = df.groupby("member_id").words.sum().reindex(ids).to_numpy()
for kind in ("raw", "cleaned", "residual", "topics"):
    full = mp_vectors(kind, keep)
    a, b = mp_vectors(kind, keep & split), mp_vectors(kind, keep & ~split)
    sim = a @ b.T
    ranks = (-sim).argsort(1).argsort(1)[np.arange(len(ids)), np.arange(len(ids))]
    nn = NearestNeighbors(n_neighbors=11, metric="cosine").fit(full).kneighbors(full, return_distance=False)[:, 1:]
    same_party = (party[nn] == party[:, None]).mean(1)
    big = np.isin(party, ["Lab", "Con", "LD", "SNP", "RUK"])
    report["candidates"][kind] = {
        "self_top1": round(float((ranks == 0).mean()), 3), "self_top10": round(float((ranks < 10).mean()), 3),
        "self_top1_quiet_quartile": round(float((ranks[words < np.quantile(words, .25)] == 0).mean()), 3),
        "party_purity": round(float(same_party[big].mean()), 3),
    }
    out[kind] = full
    print(kind, report["candidates"][kind], flush=True)

np.savez(EMB / "mp_vectors.npz", ids=ids, **out)
(EMB / "noise_report.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
print("baseline party purity if neighbours were random:",
      round(float(np.mean([(party == p).mean() for p in party])), 3))
