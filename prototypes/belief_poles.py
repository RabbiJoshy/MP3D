"""Stances as opposite poles, linked only by whether MPs' sides line up. Labelled afterwards.

The first attempt (belief_space.py) rebuilt subjects, because MPs specialise. Here, whether an MP talks about a subject
is ignored. Inside each subject, MPs who speak on it are placed along its main axes of difference. Two axes from different
subjects are compared only across MPs who speak on both. Each axis gives two opposite poles, so the two sides of one
subject can't sit together. Party and votes aren't used to build it, only to test it afterwards.

Choices still mine: number of subjects, axes per subject, evidence thresholds. Out: data/quick/poles.npz, poles_report.txt
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

AXES, MIN_PASSAGES_MP, MIN_SPEECHES_MP, MIN_OVERLAP = 3, 3, 2, 25
overall = np.asarray(X.mean(0)).ravel()
thing_words = []
for k in range(K_THINGS):
    mean = np.asarray(X[thing == k].mean(0)).ravel()
    thing_words.append(terms[np.argsort(-(mean * np.log1p(mean / (overall + 1e-6))))[:6]].tolist())
speech_counts = df.groupby("member").speech.nunique()
mps = speech_counts.index[speech_counts >= 12].to_numpy()
pos = {m: i for i, m in enumerate(mps)}
party = members.party_abbr.reindex(mps).fillna("?").to_numpy()
member_arr = df.member.to_numpy()
speech_arr = df.speech.to_numpy()
all_rows, axis_info = [], []


def scores_for(mask_rows):
    """MPs x axes. Blank where an MP hasn't said enough on that subject in this set of passages."""
    out = np.full((len(mps), len(axis_info)), np.nan)
    frame = df[mask_rows]
    for j, info in enumerate(axis_info):
        idx = frame.index[(frame.thing == info["thing"])]
        if not len(idx):
            continue
        s = pd.Series(L[idx] @ info["axis"], index=df.member.iloc[idx].to_numpy())
        g = s.groupby(level=0)
        speeches = df.iloc[idx].groupby("member").speech.nunique()
        ok = (g.size() >= MIN_PASSAGES_MP) & (speeches.reindex(g.size().index) >= MIN_SPEECHES_MP)
        m = g.mean()[ok]
        keep_ = [i for i in m.index if i in pos]
        out[[pos[i] for i in keep_], j] = m.loc[keep_].to_numpy()
    return out


df["thing"] = thing
for k in range(K_THINGS):
    idx = np.where(thing == k)[0]
    frame = df.iloc[idx]
    n = frame.groupby("member").speech.nunique()
    c = frame.member.value_counts()
    eligible = [m for m in n.index if n[m] >= MIN_SPEECHES_MP and c[m] >= MIN_PASSAGES_MP and m in pos]
    if len(eligible) < 30:
        continue
    means = np.vstack([L[idx[frame.member.to_numpy() == m]].mean(0) for m in eligible])
    centre = means.mean(0)
    U, s, Vt = np.linalg.svd(means - centre, full_matrices=False)
    for a in range(AXES):
        axis = Vt[a]
        pscore = L[idx] @ axis
        lo, hi = idx[pscore <= np.quantile(pscore, .1)], idx[pscore >= np.quantile(pscore, .9)]
        ml, mh = np.asarray(X[lo].mean(0)).ravel(), np.asarray(X[hi].mean(0)).ravel()
        axis_info.append({"thing": k, "a": a, "axis": axis, "neg": terms[np.argsort(mh - ml)[:7]].tolist(), "pos": terms[np.argsort(ml - mh)[:7]].tolist()})
say(f"{len(axis_info)} axes from {len(set(i['thing'] for i in axis_info))} subjects, {time.time() - t0:.0f}s")
A = scores_for(np.ones(len(df), bool))
A = (A - np.nanmean(A, 0)) / (np.nanstd(A, 0) + 1e-9)
say(f"share of MP x axis cells known: {(~np.isnan(A)).mean():.0%}")
axis_thing = np.array([i["thing"] for i in axis_info])
name = lambda j, sign: f"{thing_words[axis_thing[j]][0]}: " + ", ".join(axis_info[j]["pos" if sign > 0 else "neg"][:5])


def corr_matrix(M):
    mask = ~np.isnan(M)
    Y = np.where(mask, M, 0.0)
    n = mask.T.astype(float) @ mask
    sxy = Y.T @ Y
    sx = Y.T @ mask.astype(float)
    mx = sx / n.clip(1)
    my = mx.T
    sxx = (Y ** 2).T @ mask.astype(float)
    syy = sxx.T
    cov = sxy - n * mx * my
    r = cov / np.sqrt(((sxx - n * mx ** 2) * (syy - n * my ** 2)).clip(1e-9))
    r[n < MIN_OVERLAP] = np.nan
    return r, n


def eig_dims(M, k=6):
    r, _ = corr_matrix(M)
    R = np.nan_to_num(r)
    np.fill_diagonal(R, 1)
    vals, vecs = np.linalg.eigh(R)
    return vals[::-1][:k], vecs[:, ::-1][:, :k], r


def within_party(M):
    Mw = M.copy()
    for p in set(party):
        Mw[party == p] -= np.nanmean(Mw[party == p], axis=0)
    return Mw


def shuffle(M, seed):
    r = np.random.default_rng(seed)
    S = M.copy()
    for j in range(S.shape[1]):
        o = np.where(~np.isnan(S[:, j]))[0]
        S[o, j] = S[r.permutation(o), j]
    return S


vals, vecs, R = eig_dims(A)
null = np.mean([eig_dims(shuffle(A, s))[0] for s in range(5)], axis=0)
Aw = within_party(A)
vals_w, vecs_w, Rw = eig_dims(Aw)
null_w = np.mean([eig_dims(shuffle(Aw, s))[0] for s in range(5)], axis=0)
say("strength of shared dimensions:", np.round(vals, 1).tolist(), "| shuffled:", np.round(null, 1).tolist())
say("same with each party's average removed:", np.round(vals_w, 1).tolist(), "| shuffled:", np.round(null_w, 1).tolist())

# Reproducibility: links between subjects from one half of MPs vs the other
half = np.random.default_rng(SEED).permutation(len(mps)) < len(mps) / 2
cross = axis_thing[None] != axis_thing[:, None]
tri = np.triu(cross, 1)
def half_corr(M):
    ra, _ = corr_matrix(M[half])
    rb, _ = corr_matrix(M[~half])
    a, b = ra[tri], rb[tri]
    ok = ~np.isnan(a) & ~np.isnan(b)
    return np.corrcoef(a[ok], b[ok])[0, 1], ok.sum()
say(f"links between subjects, half of MPs vs other half: {half_corr(A)[0]:.2f} ({half_corr(A)[1]} pairs) | shuffled {half_corr(shuffle(A, 1))[0]:.2f}")
say(f"same with party removed: {half_corr(Aw)[0]:.2f} | shuffled {half_corr(shuffle(Aw, 1))[0]:.2f}")
say(f"correlation of the first dimension with being Conservative rather than Labour: {np.corrcoef(np.nan_to_num(A) @ vecs[:, 0], party == 'Con')[0, 1]:+.2f}")


def show_links(Rm, title):
    say(f"\n{title}")
    v = np.where(tri, Rm, np.nan)
    order = np.argsort(-np.abs(np.nan_to_num(v)), axis=None)[:16]
    for o in order:
        a, b = np.unravel_index(o, v.shape)
        sa, sb = 1, int(np.sign(v[a, b]))
        say(f"  r={v[a, b]:+.2f} [{name(a, sa)}] goes with [{name(b, sb)}]")


show_links(R, "STRONGEST LINKS BETWEEN SUBJECTS")
show_links(Rw, "STRONGEST LINKS WITH PARTY REMOVED (can't just be party)")
for title, V, Rx in (("SHARED DIMENSIONS", vecs, A), ("SHARED DIMENSIONS WITH PARTY REMOVED", vecs_w, Aw)):
    say(f"\n{title}")
    for d in range(3):
        o = np.argsort(V[:, d])
        z = np.nan_to_num(Rx) @ V[:, d]
        mo = np.argsort(z)
        say(f" dim {d}: end A MPs {[members.name[mps[i]] + ' (' + party[i] + ')' for i in mo[:7]]}")
        say(f"        end B MPs {[members.name[mps[i]] + ' (' + party[i] + ')' for i in mo[-7:]]}")
        for j in o[:6]:
            say(f"   A {V[j, d]:+.2f} {name(j, -1 if V[j, d] > 0 else 1)}")
        for j in o[-6:][::-1]:
            say(f"   B {V[j, d]:+.2f} {name(j, 1 if V[j, d] > 0 else -1)}")

votes = pd.read_parquet(ROOT / "data/clean/votes_long.parquet")
from sklearn.metrics import roc_auc_score
say("\nVOTES IT NEVER SAW, LABOUR MPs (labour-only axes direction learned on other MPs; 0.5 is chance; includes speech after the vote)")
lab = party == "Lab"
ZL = np.nan_to_num(Aw[lab]) @ vecs_w[:, :4]
for div, nm in [(2406, "Asylum bill"), (2074, "Welfare bill"), (2078, "Palestine Action ban"), (2071, "Assisted dying"), (2058, "Abortion")]:
    v = votes[votes.division_id == div].drop_duplicates("member_id", keep=False).set_index("member_id").vote
    ids = mps[lab]
    m = np.array([i in v.index for i in ids])
    y = (v.reindex(ids[m]).to_numpy() == 1).astype(int)
    out = np.zeros(m.sum())
    order_ = np.random.default_rng(1).permutation(m.sum())
    for fold in np.array_split(order_, 5):
        train = np.setdiff1d(order_, fold)
        w = np.array([np.corrcoef(ZL[m][train, d], y[train])[0, 1] for d in range(4)])
        out[fold] = ZL[m][fold] @ np.nan_to_num(w)
    say(f"  {nm:24} n={m.sum()} minority={min(y.sum(), (1 - y).sum())}: {roc_auc_score(y, out):.2f}")

np.savez(ROOT / "data/quick/poles.npz", mps=mps, A=A, vecs=vecs, vecs_w=vecs_w, axis_thing=axis_thing, vals=vals, vals_w=vals_w)
(ROOT / "data/quick/poles_report.txt").write_text("\n".join(lines))
say(f"\ndone {time.time() - t0:.0f}s")
