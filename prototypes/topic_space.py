"""Fast topic-first space from word use: passages become stars, topics form clouds, related topics sit together.

Passages -> TF-IDF -> SVD (LSA) -> k-means topics -> UMAP on a sample of passages to 3D.
CPU only, two threads. Out: data/quick/topic_space.npz, topic_space_terms.txt
"""
import os
import pathlib
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "quick"
K_TOPICS, K_REGIONS, STAR_SAMPLE, SEED = 90, 12, 24000, 7
rng = np.random.default_rng(SEED)

PARLIAMENT_WORDS = """hon right friend friends member members gentleman lady minister ministers secretary state
government house bill clause clauses amendment amendments committee debate speaker sir madam people country
uk british britain think want wants know need needs make makes made said say says going just like does did
ask asked agree important new way ensure work working year years time today constituency constituents local
support issue issues point points absolutely thank thanks labour conservative conservatives liberal democrats
snp reform party parties tory tories opposition lord lords chancellor prime shadow previous last""".split()

t0 = time.time()
df = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["member_id", "text", "words"])
print(len(df), "passages", flush=True)
fit_idx = rng.choice(len(df), 90000, replace=False)
vectoriser = TfidfVectorizer(
    lowercase=True, stop_words=sorted(ENGLISH_STOP_WORDS | set(PARLIAMENT_WORDS)), ngram_range=(1, 2),
    token_pattern=r"[a-zA-Z][a-zA-Z\-]{2,}", min_df=8, max_df=0.2, max_features=40000, sublinear_tf=True)
vectoriser.fit(df.text.iloc[fit_idx])
X = vectoriser.transform(df.text)
terms = np.array(vectoriser.get_feature_names_out())
print("tfidf", X.shape, f"{time.time() - t0:.0f}s", flush=True)

svd = TruncatedSVD(100, random_state=SEED, n_iter=4)
L = normalize(svd.fit_transform(X)).astype("float32")
print("svd", f"{time.time() - t0:.0f}s", flush=True)

km = MiniBatchKMeans(K_TOPICS, batch_size=4096, n_init=5, random_state=SEED, max_iter=120)
topic = km.fit_predict(L)
centres = normalize(km.cluster_centers_)
sim_to_centre = (L * centres[topic]).sum(1)
print("kmeans", f"{time.time() - t0:.0f}s", flush=True)

# Topic words: which terms are most over-represented in a topic relative to all passages
overall = np.asarray(X.mean(0)).ravel()
top_terms, sizes = [], []
for k in range(K_TOPICS):
    idx = np.where(topic == k)[0]
    mean = np.asarray(X[idx].mean(0)).ravel()
    top_terms.append(terms[np.argsort(-(mean * np.log1p(mean / (overall + 1e-6))))[:12]].tolist())
    sizes.append(len(idx))

# Some clusters are parliamentary talk, not subjects. Found by reading the first run's top words;
# they're left out of the map and out of each MP's topic mix.
CHATTER = {"sure", "things", "place", "look forward", "good", "action", "clear", "question", "happy", "public",
           "act", "sector", "all-party", "office", "united", "legislation", "tribute", "review"}
is_chatter = np.array([t[0] in CHATTER for t in top_terms])
print("chatter topics:", [top_terms[k][0] for k in np.where(is_chatter)[0]], f"{is_chatter[topic].mean():.1%} of passages", flush=True)

# Stars: passages sitting closest to their topic's centre, so the filler between topics doesn't blur the map
keep = np.zeros(len(df), bool)
for k in np.where(~is_chatter)[0]:
    members_k = np.where(topic == k)[0]
    n_k = max(40, int(min(len(members_k), STAR_SAMPLE * len(members_k) / len(df))))
    keep[members_k[np.argsort(-sim_to_centre[members_k])[:len(members_k) // 2]][:n_k * 2]] = True
stars = np.where(keep)[0]
if len(stars) > STAR_SAMPLE:
    stars = rng.choice(stars, STAR_SAMPLE, replace=False)
print("stars", len(stars), flush=True)

# Include the topic centres, so topic positions come from the same layout as their stars
layout_in = L[stars]
star_xyz = umap.UMAP(n_components=3, n_neighbors=25, min_dist=0.08, metric="cosine", random_state=SEED, n_jobs=1).fit_transform(layout_in)
topic_xyz = np.zeros((K_TOPICS, 3))
for k in np.where(~is_chatter)[0]:
    topic_xyz[k] = np.median(star_xyz[topic[stars] == k], axis=0)
print("umap", f"{time.time() - t0:.0f}s", flush=True)

# Regions group topics by their position in the map, so what's nearby on screen shares a region
region = np.full(K_TOPICS, -1)
region[~is_chatter] = AgglomerativeClustering(K_REGIONS, linkage="ward").fit_predict(topic_xyz[~is_chatter])
# How well the 3D layout keeps related topics together: nearest topic in 3D vs in the full LSA space
real = np.where(~is_chatter)[0]
dist3d = np.linalg.norm(topic_xyz[real][:, None] - topic_xyz[real][None], axis=2)
np.fill_diagonal(dist3d, 1e9)
simhi = centres[real] @ centres[real].T
np.fill_diagonal(simhi, -1)
hi5 = np.argsort(-simhi, 1)[:, :5]
lo5 = np.argsort(dist3d, 1)[:, :5]
print("share of a topic's 5 nearest topics preserved in 3D:", round(float(np.mean([len(set(a) & set(b)) / 5 for a, b in zip(hi5, lo5)])), 2), "(random would be", round(5 / (len(real) - 1), 2), ")", flush=True)

ids = np.array(sorted(df.member_id.unique()))
pos = {m: i for i, m in enumerate(ids)}
owner = df.member_id.map(pos).to_numpy()
counts = np.zeros((len(ids), K_TOPICS), "float32")
np.add.at(counts, (owner, topic), df.words.to_numpy())

np.savez(OUT / "topic_space.npz", centres=centres, is_chatter=is_chatter, star_xyz=star_xyz, star_topic=topic[stars], star_idx=stars, topic_xyz=topic_xyz, region=region,
         sizes=np.array(sizes), ids=ids, counts=counts, topic=topic, sim=sim_to_centre, top_terms=np.array(top_terms, dtype=object))
lines = [f"{k:2d} r{region[k]:2d} n={sizes[k]:6d} | {', '.join(top_terms[k][:6])}" for k in np.argsort(region)]
(OUT / "topic_space_terms.txt").write_text("\n".join(lines))
print("\n".join(lines))
print("total", f"{time.time() - t0:.0f}s")
