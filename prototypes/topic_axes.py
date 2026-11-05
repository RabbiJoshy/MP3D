"""Every topic gets its two widest axes: what MPs differ on most when they talk about it.

Inside a topic, passages group into arguments, and MPs are placed by which arguments they use more than expected, counting
one piece of evidence per speech. The two main directions of variation are the axes. Each end is described by its own words.
Votes, party and human labels never enter the fit; party is only added afterwards for display.

Trust rating, from automatic tests: variation compared with shuffled speakers, and whether MPs land in similar places when
axes are built from one half of the debates and MPs are placed from the other half.
The rating cut-offs, number of topics and evidence thresholds are still my choices. Ministers' speech is left out, so most
frontbench Government MPs aren't placed. An axis describes differences in what MPs say, which isn't always opposition.

Out: data/quick/topic_axes.json, topic_axes_report.txt
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
K_THINGS, STANCES_PER_THING, SVD_DIMS, CHUNK_WORDS, MIN_WORDS, MAX_DEBATE_SHARE, MIN_MP_SPEECHES, SPACE_DIMS, SEED = 110, 3, 100, 120, 25, 0.10, 12, 25, 7
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

from sklearn.feature_extraction.text import TfidfTransformer as _T
MIN_SPEECHES, MIN_DATES, SHRINK, SPLITS, SHUFFLES = 3, 2, 6, 6, 10
df["thing"] = thing
overall = np.asarray(X.mean(0)).ravel()
df_idx = df.reset_index(drop=True)


def evidence_matrix(frame, ids, args, k):
    u = frame.drop_duplicates(["member", "speech", "arg"])
    u = u[u.member.isin(ids)]
    pos = {m: i for i, m in enumerate(ids)}
    return sp.csr_matrix((np.ones(len(u)), (u.member.map(pos), u.arg)), shape=(len(ids), k)).toarray()


def mix(M, base):
    return np.sqrt((M + SHRINK * base) / (M.sum(1, keepdims=True) + SHRINK)) - np.sqrt(base)


out = []
for t in range(K_THINGS):
    rows = np.where(thing == t)[0]
    fr = df.iloc[rows].copy()
    e = fr.groupby("member").agg(sp=("speech", "nunique"), dt=("date", "nunique"))
    ids = e.index[(e.sp >= MIN_SPEECHES) & (e.dt >= MIN_DATES)].to_numpy()
    mean = np.asarray(X[rows].mean(0)).ravel()
    words = terms[np.argsort(-(mean * np.log1p(mean / (overall + 1e-6))))[:10]].tolist()
    if len(ids) < 20 or len(rows) < 150:
        out.append({"t": t, "words": words, "n": int(len(rows)), "mps": int(len(ids)), "skip": True})
        continue
    Xt = X[rows]
    df_in = np.asarray((Xt > 0).mean(0)).ravel()
    use = np.where((df_in > 0.004) & (df_in < 0.3))[0]
    Xu = normalize(Xt[:, use])
    k_args = int(np.clip(len(rows) / 60, 8, 24))
    Lt = normalize(TruncatedSVD(min(30, Xu.shape[1] - 1), random_state=SEED).fit_transform(Xu))
    km = KMeans(k_args, n_init=3, random_state=SEED).fit(Lt)
    fr["arg"] = km.labels_
    fit_ = (Lt * normalize(km.cluster_centers_)[km.labels_]).sum(1)
    arg_words = []
    mean_u = np.asarray(Xu.mean(0)).ravel()
    arg_vec = np.zeros((k_args, Xu.shape[1]))
    for a in range(k_args):
        m_a = np.asarray(Xu[km.labels_ == a].mean(0)).ravel()
        arg_vec[a] = m_a - mean_u
    M = evidence_matrix(fr, ids, None, k_args)
    base = M.sum(0) / M.sum()
    V = mix(M, base)
    Vc = V - V.mean(0)
    U, s, Vt = np.linalg.svd(Vc, full_matrices=False)
    Z = Vc @ Vt.T[:, :2]
    var = s ** 2 / (s ** 2).sum()
    # Chance: speeches shuffled between these MPs
    null = []
    speech_owner = fr[fr.member.isin(ids)].drop_duplicates("speech").set_index("speech").member
    frs = fr[fr.member.isin(ids)]
    for sh in range(SHUFFLES):
        owner = pd.Series(np.random.default_rng(sh).permutation(speech_owner.to_numpy()), index=speech_owner.index)
        fs = frs.assign(member=frs.speech.map(owner))
        Ms = evidence_matrix(fs, ids, None, k_args)
        _, ss, _ = np.linalg.svd(mix(Ms, base) - mix(Ms, base).mean(0), full_matrices=False)
        null.append(ss ** 2 / (ss ** 2).sum())
    null = np.mean(null, axis=0)
    # Held out: axes from half the debates, MPs placed from the other half
    deb = frs.debate.unique()
    held = [[], []]
    n_used = []
    if len(deb) >= 6:
        for sp_i in range(SPLITS):
            r = np.random.default_rng(50 + sp_i)
            a_set = set(r.permutation(deb)[:len(deb) // 2])
            fa, fb = frs[frs.debate.isin(a_set)], frs[~frs.debate.isin(a_set)]
            Ma, Mb = evidence_matrix(fa, ids, None, k_args), evidence_matrix(fb, ids, None, k_args)
            ok = (Ma.sum(1) >= 3) & (Mb.sum(1) >= 3)
            if ok.sum() < 12:
                continue
            Va, Vb = mix(Ma, base), mix(Mb, base)
            _, _, Ta = np.linalg.svd(Va - Va.mean(0), full_matrices=False)
            for d in range(2):
                c = np.corrcoef(Va[ok] @ Ta[d], Vb[ok] @ Ta[d])[0, 1]
                held[d].append(abs(c) if not np.isnan(c) else 0)
            n_used.append(int(ok.sum()))
    heldout = [float(np.mean(h)) if h else None for h in held]
    axes = []
    for d in range(2):
        ratio = var[d] / null[d]
        h = heldout[d]
        grade = "weak"
        if h is not None and h >= 0.5 and ratio >= 1.25:
            grade = "strong"
        elif h is not None and h >= 0.3 and ratio >= 1.1:
            grade = "moderate"
        w = (arg_vec.T @ Vt[d])
        lo, hi = use[np.argsort(w)[:9]], use[np.argsort(-w)[:9]]
        side = (km.labels_[:, None] == np.arange(k_args)[None]).astype(float) @ Vt[d]
        score = side * fit_
        pick = lambda order_: [{"t": " ".join(df_idx.text.iloc[rows[i]].split()[:40]) + "…", "m": int(df_idx.member.iloc[rows[i]]), "h": df_idx.heading.iloc[rows[i]], "d": df_idx.date.iloc[rows[i]]} for i in order_]
        eligible = fr.member.isin(ids).to_numpy()
        ord_lo = [i for i in np.argsort(score) if eligible[i]]
        ord_hi = [i for i in np.argsort(-score) if eligible[i]]
        def diverse(order_, n=2):
            seen, got = set(), []
            for i in order_[:80]:
                if df_idx.member.iloc[rows[i]] not in seen:
                    seen.add(df_idx.member.iloc[rows[i]])
                    got.append(i)
                if len(got) == n:
                    break
            return got
        z = Z[:, d]
        g = pd.Series(z).groupby(members.party_abbr.reindex(ids).fillna("?").to_numpy())
        eta = float(((g.transform("mean") - z.mean()) ** 2).sum() / ((z - z.mean()) ** 2).sum())
        axes.append({"lo": terms[lo].tolist(), "hi": terms[hi].tolist(), "var": round(float(var[d]), 3), "null": round(float(null[d]), 3), "held": None if h is None else round(h, 2),
                     "grade": grade, "party": round(eta, 2), "ex_lo": pick(diverse(ord_lo)), "ex_hi": pick(diverse(ord_hi))})
    scale = np.percentile(np.abs(Z), 97, axis=0).clip(1e-6)
    # An MP's own words: the passage nearest each MP's dominant direction on the first axis
    quotes = {}
    arg_score = (Vt[:2].T @ np.ones(2))
    for m_i, m in enumerate(ids):
        idx = np.where(fr.member.to_numpy() == m)[0]
        idx = idx[np.argsort(-(Vt[0][km.labels_[idx]] * np.sign(Z[m_i, 0] + 1e-9)))]
        best = idx[0]
        quotes[int(m)] = {"t": " ".join(df_idx.text.iloc[rows[best]].split()[:32]) + "…", "h": df_idx.heading.iloc[rows[best]], "d": df_idx.date.iloc[rows[best]]}
    out.append({"t": t, "words": words, "n": int(len(rows)), "mps": int(len(ids)), "axes": axes,
                "pts": [[int(m), round(float(Z[i, 0] / scale[0]), 3), round(float(Z[i, 1] / scale[1]), 3), int(e.sp[m])] for i, m in enumerate(ids)], "quotes": quotes})
    if t % 20 == 0:
        print(f"topic {t}/{K_THINGS}, {time.time() - t0:.0f}s", flush=True)

(ROOT / "data/quick/topic_axes.json").write_text(json.dumps(out, ensure_ascii=False))
done = [o for o in out if not o.get("skip")]
say(f"\ntopics with axes: {len(done)} of {K_THINGS}; skipped {K_THINGS - len(done)} (too few speakers)")
grades = pd.Series([o["axes"][0]["grade"] for o in done]).value_counts().to_dict()
say("first axis grades:", grades, "| second axis grades:", pd.Series([o["axes"][1]["grade"] for o in done]).value_counts().to_dict())
say("MPs per topic: median", int(np.median([o["mps"] for o in done])), "range", min(o["mps"] for o in done), max(o["mps"] for o in done))
say("party share of first axis: median", round(float(np.median([o["axes"][0]["party"] for o in done])), 2))
for grade in ["strong", "moderate", "weak"]:
    subset = sorted([o for o in done if o["axes"][0]["grade"] == grade], key=lambda o: -(o["axes"][0]["held"] or 0))
    say(f"\n{grade.upper()} FIRST AXES ({len(subset)})")
    for o in subset[:24 if grade != "weak" else 10]:
        a = o["axes"][0]
        say(f"  {', '.join(o['words'][:3]):38} MPs {o['mps']:3d} held {a['held']} var {a['var']:.2f}/{a['null']:.2f} party {a['party']:.0%} | {', '.join(a['lo'][:5])}  <->  {', '.join(a['hi'][:5])}")
(ROOT / "data/quick/topic_axes_report.txt").write_text("\n".join(lines))
say(f"\nDONE {time.time() - t0:.0f}s")
