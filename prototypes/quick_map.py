"""Fast, rough 3D map of MPs from word use only: no GPU, no embeddings. A demo, not the final method.

MP documents -> TF-IDF -> NMF topics -> UMAP to 3D. Cap on passages per MP keeps it to a few minutes.
Out: data/quick/quick.npz, topic_terms.txt
"""
import os
import pathlib
import re
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "quick"
OUT.mkdir(parents=True, exist_ok=True)
CAP, MIN_PASSAGES, N_TOPICS, SEED = 150, 30, 30, 7

PARLIAMENT_WORDS = """hon right friend friends member members gentleman lady minister ministers secretary state
government house bill clause clauses amendment amendments committee debate speaker sir madam people country
uk british britain think want wants know need needs make makes made said say says going just like does did
ask asked agree important new way ensure work working year years time today constituency constituents local
support issue issues point points absolutely thank thanks minister's labour conservative conservatives liberal
democrats snp reform party parties tory tories opposition lord lords chancellor prime shadow previous last""".split()

t0 = time.time()
df = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["member_id", "text", "role"])
counts = df.member_id.value_counts()
ids = np.array(sorted(counts.index[counts >= MIN_PASSAGES]))
df = df[df.member_id.isin(ids)]
df = df.groupby("member_id", group_keys=False).sample(frac=1, random_state=SEED).groupby("member_id").head(CAP)
docs = df.groupby("member_id").text.apply(" ".join).reindex(ids)
print(f"{len(ids)} MPs, {len(df)} passages, {time.time() - t0:.0f}s", flush=True)

vectoriser = TfidfVectorizer(
    lowercase=True, stop_words=sorted(ENGLISH_STOP_WORDS | set(PARLIAMENT_WORDS)), ngram_range=(1, 2),
    token_pattern=r"[a-zA-Z][a-zA-Z\-]{2,}", min_df=5, max_df=0.4, max_features=40000, sublinear_tf=True)
X = vectoriser.fit_transform(docs)
terms = np.array(vectoriser.get_feature_names_out())
print("tfidf", X.shape, f"{time.time() - t0:.0f}s", flush=True)

nmf = NMF(N_TOPICS, init="nndsvda", random_state=SEED, max_iter=250)
W = nmf.fit_transform(X)
H = nmf.components_
print("nmf done", f"{time.time() - t0:.0f}s", flush=True)
shares = W / np.maximum(W.sum(1, keepdims=True), 1e-9)

xyz = umap.UMAP(n_components=3, n_neighbors=12, min_dist=0.25, metric="cosine", random_state=SEED, n_jobs=1).fit_transform(shares)
print("umap done", f"{time.time() - t0:.0f}s", flush=True)

top_terms = [terms[np.argsort(-H[k])[:14]].tolist() for k in range(N_TOPICS)]
# Words that most set each MP apart from the Commons as a whole
distinct = []
for i in range(len(ids)):
    row = X[i].toarray().ravel()
    distinct.append(terms[np.argsort(-row)[:12]].tolist())

np.savez(OUT / "quick.npz", ids=ids, xyz=xyz, shares=shares, X_terms=np.array(distinct, dtype=object), top_terms=np.array(top_terms, dtype=object))

members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
lines = []
for k, t in enumerate(top_terms):
    who = members.name.reindex(ids[np.argsort(-shares[:, k])[:5]]).tolist()
    lines.append(f"{k:2d} n={(shares.argmax(1) == k).sum():3d} | {', '.join(t)} || {', '.join(who)}")
(OUT / "topic_terms.txt").write_text("\n".join(lines))
print("\n".join(lines))
print("total", f"{time.time() - t0:.0f}s")
