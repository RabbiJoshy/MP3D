"""A map of what the Commons talks about, with no MPs. Nothing about who spoke goes into it, and no paid calls.

Text:     original Hansard, chunked into passages. Stage directions and motion text are dropped using Hansard's own markup.
Words:    a term appearing across too many separate debates can't identify a subject, so it's dropped. Names, seats and
          party words come from Parliament's member list, so places and people don't define topics.
Topics:   TF-IDF, SVD, k-means.
General talk: a real subject concentrates in few debates; general talk spreads across many. The cut-off comes from the
          data, not from a list of words.
Relations: two topics are related if they use similar words, or come up in the same debates.
Layout:   one 3D space across all topics, so distances are comparable everywhere.
Tests:    relations from half the debates vs the other half; regions redrawn from resampled debates.

Choices that are still mine: number of topics, the debate-frequency limit, chunk length. Names are the data's own words.
Out: data/quick/topic_only.npz, topic_only.json, topic_only_report.txt
"""
import collections
import json
import os
import pathlib
import re
import time
import xml.etree.ElementTree as ET

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/hansard"
K_TOPICS, SVD_DIMS, CHUNK_WORDS, MIN_WORDS, MAX_DEBATE_SHARE, STARS, SEED = 110, 120, 120, 25, 0.10, 30000, 7
rng = np.random.default_rng(SEED)
lines = []
t0 = time.time()
def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)

BRACKET = re.compile(r"\[[^\]]*\]")
SPLIT = re.compile(r"(?<=[.?!])\s+(?=[\"“‘A-Z(])")
tidy = lambda t: BRACKET.sub("", " ".join("".join(t.itertext()).split())).strip(" —-")
chunks = []
for kind in ("debates", "westminhall", "standing"):
    for f in sorted((RAW / kind).glob("*.xml")):
        date = re.search(r"\d{4}-\d{2}-\d{2}", f.name).group()
        root = ET.parse(f).getroot()
        major, minor = "", ""
        bill = root.find("bill")
        if kind == "standing":
            major = bill.get("title") if bill is not None else ""
        for el in root.iter():
            if el.tag in ("major-heading", "oral-heading") and kind != "standing":
                major, minor = tidy(el), ""
            elif el.tag == "minor-heading" and kind != "standing":
                minor = tidy(el)
            elif el.tag == "speech" and el.get("person_id"):
                text = " ".join(" ".join("".join(p.itertext()).split()) for p in el.findall("p") if p.get("class") != "italic" and p.get("pwmotiontext") != "yes")
                heading = minor or major
                buf, n = [], 0
                for sent in SPLIT.split(re.sub(r"\bhon\.\s", "hon ", text)):
                    w = len(sent.split())
                    if n + w > CHUNK_WORDS * 1.4 and n >= MIN_WORDS:
                        chunks.append((date, kind, heading, " ".join(buf)))
                        buf, n = [], 0
                    buf.append(sent)
                    n += w
                if n >= MIN_WORDS:
                    chunks.append((date, kind, heading, " ".join(buf)))
df = pd.DataFrame(chunks, columns=["date", "kind", "heading", "text"])
df["debate"] = df.date + "|" + df.kind + "|" + df.heading
debate_idx, debates = pd.factorize(df.debate)
say(f"{len(df)} passages from {len(debates)} debates, {time.time() - t0:.0f}s")

vec = CountVectorizer(ngram_range=(1, 2), min_df=30, max_features=120000, token_pattern=r"[a-zA-Z][a-zA-Z\-']+", lowercase=True, dtype=np.int32)
vec.fit(df.text.iloc[rng.choice(len(df), min(120000, len(df)), replace=False)])
C = vec.transform(df.text)
terms = np.array(vec.get_feature_names_out())
say("counted", C.shape, f"{time.time() - t0:.0f}s")

# A term in too many separate debates can't mark out a subject
D = sp.csr_matrix((np.ones(len(df)), (np.arange(len(df)), debate_idx)), shape=(len(df), len(debates)))
in_debate = (C > 0).astype(np.int32).T @ D
debate_share = np.asarray((in_debate > 0).sum(1)).ravel() / len(debates)
members = pd.read_parquet(ROOT / "data/clean/members.parquet")
identity = {w for col in ["name", "constituency", "party"] for v in members[col].dropna() for w in re.findall(r"[a-zA-Z][a-zA-Z\-']+", str(v).lower())}
generic = debate_share > MAX_DEBATE_SHARE
named = np.array([any(w in identity for w in t.split()) for t in terms])
# A bigram made of generic words is generic too
generic_words = set(terms[generic & np.array([" " not in t for t in terms])])
pair = np.array([" " in t and all(w in generic_words for w in t.split()) for t in terms])
keep = ~(generic | named | pair)
say(f"terms dropped: {generic.sum()} too widespread, {named.sum()} names and seats, {pair.sum()} generic pairs. Kept {keep.sum()}.")
say("examples of widespread terms:", terms[generic][np.argsort(-debate_share[generic])][:25].tolist())
C, terms = C[:, keep], terms[keep]
enough = np.asarray((C > 0).sum(1)).ravel() >= 3
df, C = df[enough].reset_index(drop=True), C[enough]
debate_idx, debates = pd.factorize(df.debate)
say(f"passages with enough signal: {len(df)}")

X = TfidfTransformer(sublinear_tf=True).fit_transform(C)
L = normalize(TruncatedSVD(SVD_DIMS, random_state=SEED, n_iter=4).fit_transform(X)).astype("float32")
km = MiniBatchKMeans(K_TOPICS, n_init=5, batch_size=8192, max_iter=150, random_state=SEED)
topic = km.fit_predict(L)
centres = normalize(km.cluster_centers_)
fit = (L * centres[topic]).sum(1)
say(f"topics built, {time.time() - t0:.0f}s")

# General talk: spread evenly over many debates, compared with what a subject of that size would do
sizes = np.bincount(topic, minlength=K_TOPICS)
T = sp.csr_matrix((np.ones(len(df)), (topic, debate_idx)), shape=(K_TOPICS, len(debates)))
share_t = T.multiply(1 / sizes[:, None]).tocsr()
perplexity = np.array([np.exp(-(np.log(r.data) * r.data).sum()) for r in share_t])
spread = perplexity / sizes
logs = np.log(spread)
# Only clear outliers count: more than three robust standard deviations above the typical topic
cut = np.median(logs) + 3 * 1.4826 * np.median(np.abs(logs - np.median(logs)))
is_general = logs >= cut
overall = np.asarray(X.mean(0)).ravel()
top_terms = []
for k in range(K_TOPICS):
    mean = np.asarray(X[topic == k].mean(0)).ravel()
    top_terms.append(terms[np.argsort(-(mean * np.log1p(mean / (overall + 1e-6))))[:14]].tolist())
say(f"\ngeneral talk: {is_general.sum()} of {K_TOPICS} clusters, {is_general[topic].mean():.0%} of passages, cut from the data")
for k in np.argsort(-spread)[:20]:
    say(f"  {'GENERAL' if is_general[k] else 'subject'} spread {spread[k]:.2f} n={sizes[k]:5d} | {', '.join(top_terms[k][:7])}")

real = np.where(~is_general)[0]
pos_of = {k: i for i, k in enumerate(real)}
mask = ~is_general[topic]
say(f"\nsubject topics: {len(real)}; passages {mask.sum()}")


def relations(which):
    """Two topics are related by words, or by coming up in the same debates."""
    keep_chunk = mask & which
    M = sp.csr_matrix((np.ones(keep_chunk.sum()), ([pos_of[t] for t in topic[keep_chunk]], debate_idx[keep_chunk])), shape=(len(real), len(debates))).toarray()
    M = M[:, M.sum(0) > 0]
    M = M / M.sum(1, keepdims=True).clip(1)
    e = M.sum(0, keepdims=True) / M.sum()
    pmi = np.maximum(np.log((M / M.sum(1, keepdims=True).clip(1e-9)).clip(1e-12) / e.clip(1e-12)), 0)
    return normalize(pmi)


all_debates = np.ones(len(df), bool)
Tw = centres[real]
Td = relations(all_debates)
Sw, Sd = Tw @ Tw.T, Td @ Td.T
iu = np.triu_indices(len(real), 1)
say(f"word similarity vs debate similarity between topics: rank correlation {spearmanr(Sw[iu], Sd[iu])[0]:.2f}")

ranks = lambda S: pd.Series(S[iu]).rank(pct=True).to_numpy()
combined = np.zeros_like(Sw)
combined[iu] = (ranks(Sw) + ranks(Sd)) / 2
combined = combined + combined.T
np.fill_diagonal(combined, 1)
dist = 1 - combined
np.fill_diagonal(dist, 0)

layout = MDS(3, dissimilarity="precomputed", random_state=SEED, n_init=8, normalized_stress="auto").fit_transform(dist)
layout = layout - layout.mean(0)
layout = layout / np.percentile(np.linalg.norm(layout, axis=1), 95) * 16
mapped = np.linalg.norm(layout[:, None] - layout[None], axis=2)
say(f"map distance vs topic similarity: rank correlation {spearmanr(mapped[iu], dist[iu])[0]:.2f}")
nn_real = np.argsort(dist + np.eye(len(real)) * 9, axis=1)[:, :5]
nn_map = np.argsort(mapped + np.eye(len(real)) * 1e9, axis=1)[:, :5]
say(f"of each topic's 5 nearest, share also among its 5 nearest on the map: {np.mean([len(set(a) & set(b)) / 5 for a, b in zip(nn_real, nn_map)]):.2f} (chance {5 / (len(real) - 1):.2f})")

def region_features(Td_):
    return np.hstack([normalize(TruncatedSVD(25, random_state=SEED).fit_transform(Tw)), normalize(TruncatedSVD(25, random_state=SEED).fit_transform(Td_))])


features = region_features(Td)
sil = {r: silhouette_score(features, AgglomerativeClustering(r, linkage="ward").fit_predict(features)) for r in range(6, 25)}
n_regions = max(sil, key=sil.get)
region = AgglomerativeClustering(n_regions, linkage="ward").fit_predict(features)
say(f"neighbourhoods: {n_regions} chosen by silhouette", {k: round(v, 3) for k, v in sil.items()})
say("sizes:", np.bincount(region).tolist())

# Stability: are relations and neighbourhoods reproducible from separate debates?
r_vals, ari = [], []
for rep in range(8):
    pick = np.random.default_rng(100 + rep).random(len(debates)) < 0.5
    a, b = pick[debate_idx], ~pick[debate_idx]
    Ta, Tb = relations(a), relations(b)
    r_vals.append(spearmanr((Ta @ Ta.T)[iu], (Tb @ Tb.T)[iu])[0])
    ari.append(adjusted_rand_score(region, AgglomerativeClustering(n_regions, linkage="ward").fit_predict(region_features(Ta))))
say(f"debate-based relations, half the debates vs the other half: {np.mean(r_vals):.2f}")
say(f"neighbourhoods redrawn from half the debates vs full map, adjusted Rand: {np.mean(ari):.2f} (1 same, 0 chance)")


def headings(k, n=4):
    h = df.heading[topic == k]
    h = h[h.str.len() > 3]
    return h.value_counts().head(n).index.tolist()


say("\nNEIGHBOURHOODS AND THEIR TOPICS")
out_topics = []
for r in range(n_regions):
    idx = np.where(region == r)[0]
    mean = np.asarray(X[np.isin(topic, real[idx])].mean(0)).ravel()
    words = terms[np.argsort(-(mean * np.log1p(mean / (overall + 1e-6))))[:8]].tolist()
    say(f"\nREGION {r} ({len(idx)} topics): {', '.join(words)}")
    for i in idx:
        k = real[i]
        near = [top_terms[real[j]][0] for j in nn_real[i][:3]]
        say(f"  n={sizes[k]:5d} {', '.join(top_terms[k][:4]):52} | near: {', '.join(near)}")

# Stars keep the local shape of each topic, sized so clouds don't swallow their neighbours
local_scale = np.partition(mapped + np.eye(len(real)) * 1e9, 0, axis=1)[:, 0] * 0.42
star_xyz, star_topic, examples = [], [], {}
quota = (sizes[real] / sizes[real].sum() * STARS).astype(int).clip(40)
for i, k in enumerate(real):
    idx = np.where(topic == k)[0]
    pick = idx[np.argsort(-fit[idx])[:min(len(idx), quota[i] * 2)]]
    pick = rng.choice(pick, min(len(pick), quota[i]), replace=False)
    Lk = L[pick] - L[pick].mean(0)
    U, s, _ = np.linalg.svd(Lk, full_matrices=False)
    coords = U[:, :3] * s[:3]
    coords = coords / max(np.percentile(np.linalg.norm(coords, axis=1), 85), 1e-6) * local_scale[i]
    star_xyz.append(layout[i] + np.clip(coords, -2 * local_scale[i], 2 * local_scale[i]))
    star_topic.append(np.full(len(pick), i))
    best = idx[np.argsort(-fit[idx])[:30]]
    seen, ex = set(), []
    for b in best:
        if df.heading[b] not in seen and len(df.text[b]) > 80:
            seen.add(df.heading[b])
            first = " ".join(df.text[b].split()[:42])
            ex.append({"t": first + "…", "h": df.heading[b], "d": df.date[b]})
        if len(ex) == 3:
            break
    examples[i] = ex
star_xyz, star_topic = np.vstack(star_xyz), np.concatenate(star_topic)

region_info = []
for r in range(n_regions):
    idx = np.where(region == r)[0]
    mean = np.asarray(X[np.isin(topic, real[idx])].mean(0)).ravel()
    region_info.append({"words": terms[np.argsort(-(mean * np.log1p(mean / (overall + 1e-6))))[:6]].tolist(), "x": [round(float(v), 2) for v in layout[idx].mean(0)], "size": int(len(idx))})
out = {"regions": region_info, "topics": [{"words": top_terms[k][:10], "r": int(region[i]), "x": [round(float(v), 2) for v in layout[i]], "n": int(sizes[k]),
                                             "heads": headings(k), "near": [int(j) for j in nn_real[i][:6]], "ex": examples[i], "cohesion": round(float(fit[topic == k].mean()), 4)} for i, k in enumerate(real)],
       "stats": {"passages": int(len(df)), "debates": int(len(debates)), "general_share": round(float(is_general[topic].mean()), 3), "map_corr": round(float(spearmanr(mapped[iu], dist[iu])[0]), 2),
                 "half_corr": round(float(np.mean(r_vals)), 2), "ari": round(float(np.mean(ari)), 2)}}
(ROOT / "data/quick/topic_only.json").write_text(json.dumps(out, ensure_ascii=False))
np.savez(ROOT / "data/quick/topic_only.npz", star_xyz=star_xyz, star_topic=star_topic, layout=layout, region=region)
(ROOT / "data/quick/topic_only_report.txt").write_text("\n".join(lines))
say(f"\ndone {time.time() - t0:.0f}s, stars {len(star_xyz)}")
