"""A space of stances, built by who holds them together. Labelled only afterwards.

1. Each passage of a speech becomes a vector (TF-IDF, then SVD). Common words are dropped by how many debates they
   appear in, and names by Parliament's member list.
2. Two parameters per passage: the thing it's about (top-level cluster) and the stance on that thing (a split of that
   cluster's passages). Together they make one unit.
3. Units move closer when the same people say them, more than chance. Party is never used for this.
4. Tests: shuffled speakers, relations from one half of MPs vs the other half, and the same space with each party's
   average removed, to see which links are more than party.
5. Labels come last, from each unit's own top words. The probes at the end only read the finished space.

Ministers' passages are left out, because they speak for their job. Choices still mine: numbers of things and stances,
evidence thresholds. Out: data/quick/belief.npz, belief.json, belief_report.txt
"""
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
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import MDS
from sklearn.preprocessing import normalize

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
K_THINGS, STANCES_PER_THING, SVD_DIMS, CHUNK_WORDS, MIN_WORDS, MAX_DEBATE_SHARE, MIN_MP_SPEECHES, SPACE_DIMS, SEED = 90, 3, 100, 120, 25, 0.10, 12, 25, 7
rng = np.random.default_rng(SEED)
lines = []
t0 = time.time()
def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)

people = json.loads((RAW / "people.json").read_text())
person_to_member = {p["id"]: int(i["identifier"]) for p in people["persons"] for i in p.get("identifiers", []) if i.get("scheme") == "datadotparl_id"}
members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
gov_posts = {m: json.loads(v) for m, v in members.gov_posts.items()}
chair_posts = {m: [p for p in json.loads(v) if re.search("speaker|ways and means", p.get("name") or "", re.I)] for m, v in members.other_posts.items()}
def in_post(posts, date):
    return any((p.get("startDate") or "")[:10] <= date <= (p.get("endDate") or "9999")[:10] for p in posts)

BRACKET = re.compile(r"\[[^\]]*\]")
SPLIT = re.compile(r"(?<=[.?!])\s+(?=[\"“‘A-Z(])")
tidy = lambda t: BRACKET.sub("", " ".join("".join(t.itertext()).split())).strip(" —-")
rows = []
for kind in ("debates", "westminhall", "standing"):
    for f in sorted((RAW / "hansard" / kind).glob("*.xml")):
        date = re.search(r"\d{4}-\d{2}-\d{2}", f.name).group()
        root = ET.parse(f).getroot()
        major, minor = "", ""
        chairs = {m.get("person_id") for m in root.findall("./committee/chairmen/mpname")}
        bill = root.find("bill")
        if kind == "standing":
            major = bill.get("title") if bill is not None else ""
        for el in root.iter():
            if el.tag in ("major-heading", "oral-heading") and kind != "standing":
                major, minor = tidy(el), ""
            elif el.tag == "minor-heading" and kind != "standing":
                minor = tidy(el)
            elif el.tag == "speech" and el.get("person_id"):
                pid = el.get("person_id")
                member = person_to_member.get(pid)
                if member not in members.index or pid in chairs or in_post(gov_posts.get(member, []), date) or in_post(chair_posts.get(member, []), date):
                    continue
                text = " ".join(" ".join("".join(p.itertext()).split()) for p in el.findall("p") if p.get("class") != "italic" and p.get("pwmotiontext") != "yes")
                buf, n = [], 0
                for sent in SPLIT.split(re.sub(r"\bhon\.\s", "hon ", text)):
                    w = len(sent.split())
                    if n + w > CHUNK_WORDS * 1.4 and n >= MIN_WORDS:
                        rows.append((el.get("id"), member, date, minor or major, " ".join(buf)))
                        buf, n = [], 0
                    buf.append(sent)
                    n += w
                if n >= MIN_WORDS:
                    rows.append((el.get("id"), member, date, minor or major, " ".join(buf)))
df = pd.DataFrame(rows, columns=["speech", "member", "date", "heading", "text"])
df["debate"] = df.date + "|" + df.heading
debate_idx, debates = pd.factorize(df.debate)
say(f"{len(df)} non-ministerial passages by {df.member.nunique()} MPs, {time.time() - t0:.0f}s")

vec = CountVectorizer(ngram_range=(1, 2), min_df=25, max_features=100000, token_pattern=r"[a-zA-Z][a-zA-Z\-']+", lowercase=True, dtype=np.int32)
vec.fit(df.text.iloc[rng.choice(len(df), min(100000, len(df)), replace=False)])
C = vec.transform(df.text)
terms = np.array(vec.get_feature_names_out())
D = sp.csr_matrix((np.ones(len(df)), (np.arange(len(df)), debate_idx)), shape=(len(df), len(debates)))
share = np.asarray((((C > 0).astype(np.int32).T @ D) > 0).sum(1)).ravel() / len(debates)
identity = {w for col in ["name", "constituency", "party"] for v in members[col].dropna() for w in re.findall(r"[a-zA-Z][a-zA-Z\-']+", str(v).lower())}
generic = share > MAX_DEBATE_SHARE
named = np.array([any(w in identity for w in t.split()) for t in terms])
generic_words = set(terms[generic & np.array([" " not in t for t in terms])])
pair = np.array([" " in t and all(w in generic_words for w in t.split()) for t in terms])
keep = ~(generic | named | pair)
C, terms = C[:, keep], terms[keep]
enough = np.asarray((C > 0).sum(1)).ravel() >= 3
df, C = df[enough].reset_index(drop=True), C[enough]
X = TfidfTransformer(sublinear_tf=True).fit_transform(C)
L = normalize(TruncatedSVD(SVD_DIMS, random_state=SEED, n_iter=4).fit_transform(X)).astype("float32")
say(f"{len(df)} passages as {SVD_DIMS}-number vectors, vocabulary {len(terms)}, {time.time() - t0:.0f}s")

# Parameter 1: the thing
thing = MiniBatchKMeans(K_THINGS, n_init=5, batch_size=8192, max_iter=150, random_state=SEED).fit_predict(L)
# Parameter 2: the stance, a split of that thing's passages. The thing's own average is removed first, so what's left
# is how passages differ from each other about the same subject.
unit = np.zeros(len(df), int)
thing_of_unit, side_of_unit = [], []
next_id = 0
for k in range(K_THINGS):
    idx = np.where(thing == k)[0]
    residual = L[idx] - L[idx].mean(0)
    parts = KMeans(STANCES_PER_THING, n_init=5, random_state=SEED).fit_predict(residual)
    for part in range(STANCES_PER_THING):
        unit[idx[parts == part]] = next_id
        thing_of_unit.append(k)
        side_of_unit.append(part)
        next_id += 1
thing_of_unit = np.array(thing_of_unit)
n_units = next_id
df["unit"] = unit
say(f"{K_THINGS} things x {STANCES_PER_THING} stances = {n_units} units, {time.time() - t0:.0f}s")

overall = np.asarray(X.mean(0)).ravel()
unit_words = []
for u in range(n_units):
    mean = np.asarray(X[unit == u].mean(0)).ravel()
    unit_words.append(terms[np.argsort(-(mean * np.log1p(mean / (overall + 1e-6))))[:10]].tolist())
label = lambda u: ", ".join(unit_words[u][:5])

speech_counts = df.groupby("member").speech.nunique()
mps = speech_counts.index[speech_counts >= MIN_MP_SPEECHES].to_numpy()
pos = {m: i for i, m in enumerate(mps)}
party = members.party_abbr.reindex(mps).fillna("?").to_numpy()
say(f"MPs supplying evidence: {len(mps)}", pd.Series(party).value_counts().head(6).to_dict())
d = df[df.member.isin(pos)]


def pmi_matrix(frame):
    """One piece of evidence per MP, per speech, per unit. Positive means an MP says this more than expected."""
    u = frame.drop_duplicates(["member", "speech", "unit"])
    M = sp.csr_matrix((np.ones(len(u)), (u.member.map(pos), u.unit)), shape=(len(mps), n_units)).toarray()
    row = M.sum(1, keepdims=True).clip(1)
    col = M.sum(0, keepdims=True).clip(1)
    return np.log(((M + 0.5) / (row + 0.5 * n_units)) / (col / col.sum()))


def similarity(P, dims=SPACE_DIMS):
    """Units are similar when the same MPs over-use both."""
    centred = P - P.mean(0)
    U, s, Vt = np.linalg.svd(centred, full_matrices=False)
    vectors = normalize(Vt[:dims].T * s[:dims])
    return vectors @ vectors.T, vectors


P = pmi_matrix(d)
S, vectors = similarity(P)
iu = np.triu_indices(n_units, 1)
residual = P.copy()
for p in set(party):
    residual[party == p] -= residual[party == p].mean(0)
S_within, _ = similarity(residual)
parties_only = np.array([P[party == p].mean(0) for p in sorted(set(party)) if (party == p).sum() >= 3])
say(f"how much of the space is party: correlation of relations with party removed vs not {spearmanr(S[iu], S_within[iu])[0]:.2f}")

# Tests
half = rng.permutation(len(mps)) < len(mps) / 2
Sa, _ = similarity(P[half])
Sb, _ = similarity(P[~half])
say(f"relations from one half of MPs vs the other half: {spearmanr(Sa[iu], Sb[iu])[0]:.2f}")
Sa2, _ = similarity(residual[half])
Sb2, _ = similarity(residual[~half])
say(f"same with party removed: {spearmanr(Sa2[iu], Sb2[iu])[0]:.2f}")
shuffled = d.copy()
speech_owner = d.drop_duplicates("speech").set_index("speech").member
shuffled["member"] = shuffled.speech.map(pd.Series(rng.permutation(speech_owner.to_numpy()), index=speech_owner.index))
Ps = pmi_matrix(shuffled)
Ssa, _ = similarity(Ps[half])
Ssb, _ = similarity(Ps[~half])
say(f"chance: speeches shuffled between MPs, half vs half: {spearmanr(Ssa[iu], Ssb[iu])[0]:.2f}")

# Does the space cross topics, and do sides of one thing separate?
same_thing = thing_of_unit[iu[0]] == thing_of_unit[iu[1]]
say(f"\nmean similarity: same thing {S[iu][same_thing]:.3f}" if False else f"\nmean similarity: same thing {S[iu][same_thing].mean():.3f}, different things {S[iu][~same_thing].mean():.3f}")
say(f"similarity needed to be in the top 1% of all pairs: {np.quantile(S[iu], .99):.2f}")
say(f"top 1% most similar pairs that cross things: {(~same_thing[np.argsort(-S[iu])[:len(S[iu]) // 100]]).mean():.0%}")

say("\nSTRONGEST LINKS BETWEEN DIFFERENT THINGS, party removed (these can't just be party)")
order = np.argsort(-S_within[iu])
shown = 0
for o in order:
    a, b = iu[0][o], iu[1][o]
    if thing_of_unit[a] != thing_of_unit[b]:
        say(f"  {S_within[a, b]:.2f} [{label(a)}] <-> [{label(b)}]")
        shown += 1
    if shown == 18:
        break

say("\nTHINGS WHOSE SIDES ARE HELD BY DIFFERENT PEOPLE (lowest similarity inside a thing)")
gaps = []
for k in range(K_THINGS):
    us = np.where(thing_of_unit == k)[0]
    pairs = [(S[a, b], a, b) for i, a in enumerate(us) for b in us[i + 1:]]
    gaps.append(min(pairs) + (k,))
for value, a, b, k in sorted(gaps)[:14]:
    pa = [(p, round(float(P[party == p, a].mean() - P[party == p, b].mean()), 2)) for p in ["Lab", "Con", "LD"]]
    say(f"  {value:+.2f} [{label(a)}] vs [{label(b)}]  | party lean {pa}")

features = vectors
region = AgglomerativeClustering(16, linkage="ward").fit_predict(features)
layout = MDS(3, dissimilarity="precomputed", random_state=SEED, n_init=4, max_iter=200, normalized_stress="auto").fit_transform(np.clip(1 - S, 0, None) * (1 - np.eye(n_units)))
mapped = np.linalg.norm(layout[:, None] - layout[None], axis=2)
say(f"\n3D map distance vs similarity: rank correlation {spearmanr(mapped[iu], -S[iu])[0]:.2f}")
say("\nWHAT FORMED: groups of units, labelled afterwards")
for r in range(16):
    us = np.where(region == r)[0]
    lean = {p: round(float(P[party == p][:, us].mean()), 2) for p in ["Lab", "Con", "LD", "RUK", "SNP", "Green"] if (party == p).any()}
    say(f"\nGROUP {r} ({len(us)} units, {len(set(thing_of_unit[us]))} things) party lean {lean}")
    sizes = np.bincount(unit, minlength=n_units)[us]
    for u in us[np.argsort(-sizes)[:8]]:
        say(f"    {label(u)}")

# Reading the finished space: the example that started this
def find(*needles):
    return [u for u in range(n_units) if any(n in " ".join(unit_words[u][:6]) for n in needles)]
say("\nPROBES (after the fact; a pair's percentile among all pairs, 100 is closest; first number includes party, second has it removed)")
pct = pd.Series(S[iu]).rank(pct=True).to_numpy()
pct_w = pd.Series(S_within[iu]).rank(pct=True).to_numpy()
index = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(*iu))}
for name_a, a_terms, name_b, b_terms in [("tax", ("tax rises", "taxes", "tax cut", "income tax", "national insurance"), "boats/deterrent", ("deterrent", "small boat", "boats", "illegal")),
                                         ("brexit", ("brexit", "european union", "customs"), "boats/deterrent", ("deterrent", "small boat", "boats", "illegal")),
                                         ("tax", ("tax rises", "taxes", "tax cut", "income tax", "national insurance"), "brexit", ("brexit", "european union", "customs")),
                                         ("refugees", ("refugee", "safe routes", "persecution"), "gaza", ("gaza", "palestin")),
                                         ("refugees", ("refugee", "safe routes", "persecution"), "boats/deterrent", ("deterrent", "small boat", "boats", "illegal"))]:
    ua, ub = find(*a_terms), find(*b_terms)
    pairs = [(pct[index[(min(a, b), max(a, b))]], pct_w[index[(min(a, b), max(a, b))]], a, b) for a in ua for b in ub if a != b]
    if pairs:
        best = max(pairs)
        say(f"  {name_a} vs {name_b}: {len(pairs)} pairs, median {np.median([x[0] for x in pairs]) * 100:.0f} / {np.median([x[1] for x in pairs]) * 100:.0f}; closest {best[0] * 100:.0f} / {best[1] * 100:.0f} [{label(best[2])}] <-> [{label(best[3])}]")

np.savez(ROOT / "data/quick/belief.npz", layout=layout, region=region, thing=thing_of_unit, S=S.astype("float32"), S_within=S_within.astype("float32"), P=P.astype("float32"), mps=mps)
(ROOT / "data/quick/belief.json").write_text(json.dumps({"words": unit_words}, ensure_ascii=False))
(ROOT / "data/quick/belief_report.txt").write_text("\n".join(lines))
say(f"\ndone {time.time() - t0:.0f}s")
