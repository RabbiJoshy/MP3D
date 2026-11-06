"""Every topic decides its own number of axes, from 0 to 3: what MPs differ on most when they talk about it.

Inside a topic, passages group into arguments, and MPs are placed by which arguments they use more than expected, counting
one piece of evidence per speech. Axes are added in order while each one both beats shuffled speakers and holds up when
rebuilt from one half of the debates and tested on the other. The first that fails stops the count; three is the cap.
Every axis that passes, across all topics, is then combined into one space of MPs.

Votes, party and human labels never enter the fit; party is only added afterwards for display and testing.
Pass marks, number of topics and evidence thresholds are still my choices. Ministers' speech is left out.
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

MIN_SPEECHES, MIN_DATES, SHRINK, SPLITS, SHUFFLES, MAX_AXES = 3, 2, 6, 6, 10, 3
PASS_HELD, PASS_RATIO, STRONG_HELD, STRONG_RATIO = 0.3, 1.1, 0.5, 1.25
df["thing"] = thing
overall = np.asarray(X.mean(0)).ravel()
df_idx = df.reset_index(drop=True)
debate_half = {d: (hash(d) % 2) for d in df.debate.unique()}
debate_half = {d: int(np.random.default_rng(abs(hash(d)) % (2 ** 32)).random() < 0.5) for d in sorted(df.debate.unique())}


def evidence_matrix(frame, ids, k):
    u = frame.drop_duplicates(["member", "speech", "arg"])
    u = u[u.member.isin(ids)]
    pos = {m: i for i, m in enumerate(ids)}
    return sp.csr_matrix((np.ones(len(u)), (u.member.map(pos), u.arg)), shape=(len(ids), k)).toarray()


def mix(M, base):
    return np.sqrt((M + SHRINK * base) / (M.sum(1, keepdims=True) + SHRINK)) - np.sqrt(base)


out, columns, half_cols = [], [], []
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
    mean_u = np.asarray(Xu.mean(0)).ravel()
    arg_vec = np.vstack([np.asarray(Xu[km.labels_ == a].mean(0)).ravel() - mean_u for a in range(k_args)])
    M = evidence_matrix(fr, ids, k_args)
    base = M.sum(0) / M.sum()
    V = mix(M, base)
    Vmean = V.mean(0)
    U, s, Vt = np.linalg.svd(V - Vmean, full_matrices=False)
    Z = (V - Vmean) @ Vt.T[:, :MAX_AXES]
    var = s ** 2 / (s ** 2).sum()
    frs = fr[fr.member.isin(ids)]
    speech_owner = frs.drop_duplicates("speech").set_index("speech").member
    null = []
    for sh in range(SHUFFLES):
        owner = pd.Series(np.random.default_rng(sh).permutation(speech_owner.to_numpy()), index=speech_owner.index)
        Ms = evidence_matrix(frs.assign(member=frs.speech.map(owner)), ids, k_args)
        Vs = mix(Ms, base)
        _, ss, _ = np.linalg.svd(Vs - Vs.mean(0), full_matrices=False)
        null.append(ss ** 2 / (ss ** 2).sum())
    null = np.mean(null, axis=0)
    deb = frs.debate.unique()
    held = [[] for _ in range(MAX_AXES)]
    if len(deb) >= 6:
        for sp_i in range(SPLITS):
            a_set = set(np.random.default_rng(50 + sp_i).permutation(deb)[:len(deb) // 2])
            fa, fb = frs[frs.debate.isin(a_set)], frs[~frs.debate.isin(a_set)]
            Ma, Mb = evidence_matrix(fa, ids, k_args), evidence_matrix(fb, ids, k_args)
            ok = (Ma.sum(1) >= 3) & (Mb.sum(1) >= 3)
            if ok.sum() < 12:
                continue
            Va, Vb = mix(Ma, base), mix(Mb, base)
            _, _, Ta = np.linalg.svd(Va - Va.mean(0), full_matrices=False)
            for d in range(MAX_AXES):
                c = np.corrcoef(Va[ok] @ Ta[d], Vb[ok] @ Ta[d])[0, 1]
                held[d].append(abs(c) if not np.isnan(c) else 0)
    axes, n_axes, stopped = [], 0, False
    eligible = fr.member.isin(ids).to_numpy()
    party_ids = members.party_abbr.reindex(ids).fillna("?").to_numpy()
    for d in range(MAX_AXES):
        h = float(np.mean(held[d])) if held[d] else None
        ratio = float(var[d] / null[d])
        passes = h is not None and h >= PASS_HELD and ratio >= PASS_RATIO
        grade = "strong" if (h is not None and h >= STRONG_HELD and ratio >= STRONG_RATIO) else ("moderate" if passes else "weak")
        if passes and not stopped:
            n_axes += 1
        else:
            stopped = True
        w = arg_vec.T @ Vt[d]
        lo, hi = use[np.argsort(w)[:9]], use[np.argsort(-w)[:9]]
        score = ((km.labels_[:, None] == np.arange(k_args)[None]).astype(float) @ Vt[d]) * fit_

        def diverse(order_, n=2):
            seen, got = set(), []
            for i in order_[:120]:
                m = df_idx.member.iloc[rows[i]]
                if eligible[i] and m not in seen and len(df_idx.text.iloc[rows[i]]) > 120:
                    seen.add(m)
                    got.append({"t": " ".join(df_idx.text.iloc[rows[i]].split()[:40]) + "…", "m": int(m), "h": df_idx.heading.iloc[rows[i]], "d": df_idx.date.iloc[rows[i]]})
                if len(got) == n:
                    break
            return got
        # How stance-like is this axis? Three unsupervised signals:
        # 1. Do both ends speak in the same debates, as in an argument, more than chance would put them together?
        es = score[eligible]
        er = np.where(eligible)[0]
        lo_i, hi_i = er[es <= np.quantile(es, 0.2)], er[es >= np.quantile(es, 0.8)]
        deb_lo, deb_hi = fr.debate.to_numpy()[lo_i], fr.debate.to_numpy()[hi_i]
        def together(dl, dh):
            shared = set(dl) & set(dh)
            return (np.isin(dl, list(shared)).mean() + np.isin(dh, list(shared)).mean()) / 2
        observed = together(deb_lo, deb_hi)
        pool = np.concatenate([deb_lo, deb_hi])
        rr = np.random.default_rng(3)
        expected = np.mean([together(*np.split(rr.permutation(pool), [len(deb_lo)])) for _ in range(20)])
        codebate = float(observed / max(expected, 1e-6))
        # 2. Does this axis dominate the next one?
        dominance = float(var[d] / max(var[d + 1], 1e-9)) if d + 1 < len(var) else None
        # 3. Do MPs split into two camps? (bimodality coefficient; above about 0.56 suggests two groups)
        zz = Z[:, d]
        n_ = len(zz)
        sk = float(pd.Series(zz).skew())
        ku = float(pd.Series(zz).kurt())
        bimodal = (sk ** 2 + 1) / (ku + 3 * (n_ - 1) ** 2 / ((n_ - 2) * (n_ - 3)))
        z = Z[:, d]
        g = pd.Series(z).groupby(party_ids)
        eta = float(((g.transform("mean") - z.mean()) ** 2).sum() / ((z - z.mean()) ** 2).sum())
        axes.append({"lo": terms[lo].tolist(), "hi": terms[hi].tolist(), "var": round(float(var[d]), 3), "null": round(float(null[d]), 3), "held": None if h is None else round(h, 2),
                     "grade": grade, "used": bool(passes and n_axes > d), "party": round(eta, 2),
                     "codebate": round(codebate, 2), "dominance": None if dominance is None else round(dominance, 2), "bimodal": round(float(bimodal), 3), "ex_lo": diverse(np.argsort(score)), "ex_hi": diverse(np.argsort(-score))})
    scale = np.percentile(np.abs(Z), 97, axis=0).clip(1e-6)
    # The combined space uses each passing axis, measured from all speech and, as a check, from each half of the debates
    for d in range(n_axes):
        col = pd.Series(Z[:, d] / scale[d], index=ids)
        halves = []
        for half_id in (0, 1):
            fh = frs[frs.debate.map(debate_half) == half_id]
            Mh = evidence_matrix(fh, ids, k_args)
            okh = Mh.sum(1) >= 3
            zh = ((mix(Mh, base) - Vmean) @ Vt[d]) / scale[d]
            halves.append(pd.Series(np.where(okh, zh, np.nan), index=ids))
        columns.append((t, d, col))
        half_cols.append(halves)
    quotes = {}
    for m_i, m in enumerate(ids):
        idx = np.where(fr.member.to_numpy() == m)[0]
        idx = idx[np.argsort(-(Vt[0][km.labels_[idx]] * np.sign(Z[m_i, 0] + 1e-9)))]
        quotes[int(m)] = {"t": " ".join(df_idx.text.iloc[rows[idx[0]]].split()[:32]) + "…", "h": df_idx.heading.iloc[rows[idx[0]]], "d": df_idx.date.iloc[rows[idx[0]]]}
    out.append({"t": t, "words": words, "n": int(len(rows)), "mps": int(len(ids)), "n_axes": n_axes, "axes": axes,
                "pts": [[int(m)] + [round(float(Z[i, d] / scale[d]), 3) for d in range(MAX_AXES)] + [int(e.sp[m])] for i, m in enumerate(ids)], "quotes": quotes})

done = [o for o in out if not o.get("skip")]
st = pd.DataFrame([{"topic": ", ".join(o["words"][:2]), "n_axes": o["n_axes"], "codebate": o["axes"][0]["codebate"], "dominance": o["axes"][0]["dominance"], "bimodal": o["axes"][0]["bimodal"],
                    "party": o["axes"][0]["party"], "lo": ", ".join(o["axes"][0]["lo"][:4]), "hi": ", ".join(o["axes"][0]["hi"][:4])} for o in done])
st["stance_score"] = (st.codebate.rank(pct=True) + st.dominance.rank(pct=True) + st.bimodal.rank(pct=True)) / 3
from scipy.stats import spearmanr
say("\nSTANCE-LIKENESS OF EACH TOPIC'S FIRST AXIS")
say(f"  how the three signals agree (rank correlation): same-debate vs dominance {spearmanr(st.codebate, st.dominance)[0]:+.2f}, same-debate vs two camps {spearmanr(st.codebate, st.bimodal)[0]:+.2f}, dominance vs two camps {spearmanr(st.dominance, st.bimodal)[0]:+.2f}")
say(f"  stance score vs how much party explains the axis: {spearmanr(st.stance_score, st.party)[0]:+.2f}")
for title, frame in (("MOST STANCE-LIKE", st.sort_values("stance_score", ascending=False).head(15)), ("MOST FOCUS-LIKE", st.sort_values("stance_score").head(10))):
    say(f"\n{title}")
    for r in frame.itertuples():
        say(f"  {r.topic:32} score {r.stance_score:.2f} | same debates {r.codebate:.2f} dominance {r.dominance:.2f} two camps {r.bimodal:.2f} party {r.party:.0%} | {r.lo}  <->  {r.hi}")
say("\nCHECKS ON TOPICS WE ALREADY UNDERSTAND")
for word in ["asylum", "gaza", "oil", "trans", "solar", "rail", "cancer", "gp", "pension", "dental"]:
    hit = st[st.topic.str.startswith(word)]
    for r in hit.itertuples():
        say(f"  {r.topic:32} score {r.stance_score:.2f} (rank {int((st.stance_score > r.stance_score).sum()) + 1} of {len(st)}) | same debates {r.codebate:.2f} dominance {r.dominance:.2f} two camps {r.bimodal:.2f}")
counts = pd.Series([o["n_axes"] for o in done]).value_counts().sort_index().to_dict()
say(f"\nAXES CHOSEN BY THE DATA (cap {MAX_AXES}): {counts}  (number of axes: number of topics)")
for n_ in range(MAX_AXES + 1):
    names_ = [", ".join(o["words"][:2]) for o in sorted(done, key=lambda o: -o["mps"]) if o["n_axes"] == n_]
    say(f"  {n_} axes ({len(names_)}): {names_[:16]}")
for o in [o for o in done if o["n_axes"] == 3][:8]:
    say(f"  3-axis topic {', '.join(o['words'][:3])}: " + " | ".join(f"{', '.join(a['lo'][:3])} <-> {', '.join(a['hi'][:3])}" for a in o["axes"]))


# Combined space: MPs x passing axes, with gaps left as gaps
all_ids = sorted({m for _, _, c in columns for m in c.index})
A = pd.concat([c.rename(f"{t}.{d}") for t, d, c in columns], axis=1).reindex(all_ids).to_numpy()
Ah = [pd.concat([h[i].rename(f"{j}") for j, h in enumerate(half_cols)], axis=1).reindex(all_ids).to_numpy() for i in (0, 1)]
cover = (~np.isnan(A)).sum(1)
keep = cover >= 4
all_ids = np.array(all_ids)[keep]
A, Ah = A[keep], [x[keep] for x in Ah]
party = members.party_abbr.reindex(all_ids).fillna("?").to_numpy()
say(f"\nCOMBINED SPACE: {len(all_ids)} MPs with positions on at least 4 axes; {A.shape[1]} axes from {len(set(t for t, _, _ in columns))} topics; {np.mean(~np.isnan(A)):.0%} of cells known")


def soft_impute(Mx, rank=3, iters=150, shrink=1.0):
    mask = ~np.isnan(Mx)
    mu, sd = np.nanmean(Mx, 0), np.nanstd(Mx, 0) + 1e-9
    Y = np.where(mask, (Mx - mu) / sd, 0.0)
    fill = Y.copy()
    for _ in range(iters):
        U_, s_, Vt_ = np.linalg.svd(fill, full_matrices=False)
        s_ = np.maximum(s_ - shrink, 0)
        fill = np.where(mask, Y, (U_[:, :rank] * s_[:rank]) @ Vt_[:rank])
    expl = np.array([np.sum(((U_[:, [r]] * s_[r]) @ Vt_[[r]])[mask] ** 2) for r in range(rank)]) / np.sum(Y[mask] ** 2)
    return Vt_[:rank].T, mu, sd, expl


def place(Mx, Lw, mu, sd, ridge=1.0):
    outp = np.full((len(Mx), Lw.shape[1]), np.nan)
    for i, row in enumerate(Mx):
        o = ~np.isnan(row)
        if o.sum() >= 3:
            Lo = Lw[o]
            outp[i] = np.linalg.solve(Lo.T @ Lo + ridge * np.eye(Lw.shape[1]), Lo.T @ ((row[o] - mu[o]) / sd[o]))
    return outp


Lw, mu, sd, expl = soft_impute(A)
P = place(A, Lw, mu, sd)
Sh = A.copy()
rr = np.random.default_rng(3)
for j in range(Sh.shape[1]):
    o = np.where(~np.isnan(Sh[:, j]))[0]
    Sh[o, j] = Sh[rr.permutation(o), j]
_, _, _, expl_null = soft_impute(Sh)
say(f"variance by dimension {np.round(expl, 3).tolist()} | shuffled {np.round(expl_null, 3).tolist()}")
Pa, Pb = place(Ah[0], Lw, mu, sd), place(Ah[1], Lw, mu, sd)
okab = ~np.isnan(Pa[:, 0]) & ~np.isnan(Pb[:, 0])
lab = party == "Lab"
dims_out = []
for d in range(3):
    c = np.corrcoef(Pa[okab, d], Pb[okab, d])[0, 1]
    cl = np.corrcoef(Pa[okab & lab, d], Pb[okab & lab, d])[0, 1]
    g = pd.Series(P[:, d]).groupby(party)
    eta = float(((g.transform("mean") - P[:, d].mean()) ** 2).sum() / ((P[:, d] - P[:, d].mean()) ** 2).sum())
    order_ = np.argsort(Lw[:, d])
    def end(idx, sign):
        res = []
        for j in idx[:5]:
            t, ax, _ = columns[j] if False else (columns[j][0], columns[j][1], None)
            o = next(x for x in out if x["t"] == t)
            words_ = o["axes"][ax]["hi" if (Lw[j, d] * sign) > 0 else "lo"][:3]
            res.append({"topic": ", ".join(o["words"][:2]), "words": words_})
        return res
    dims_out.append({"held": round(float(c), 2), "held_lab": round(float(cl), 2), "party": round(eta, 2), "expl": round(float(expl[d]), 3),
                     "lo": end(order_, -1), "hi": end(order_[::-1], 1)})
    o_ = np.argsort(P[:, d])
    say(f"\n dim {d}: separate debates {c:.2f} (within Labour {cl:.2f}, {int((okab & lab).sum())} MPs), party explains {eta:.0%}")
    say("   low MPs :", [f"{members.name[all_ids[i]]} ({party[i]})" for i in o_[:8]])
    say("   high MPs:", [f"{members.name[all_ids[i]]} ({party[i]})" for i in o_[-8:]])
    say("   low end :", [f"{e['topic']}: {', '.join(e['words'])}" for e in dims_out[-1]["lo"]])
    say("   high end:", [f"{e['topic']}: {', '.join(e['words'])}" for e in dims_out[-1]["hi"]])
scaleP = np.nanpercentile(np.abs(P), 97, axis=0).clip(1e-6)
combined = {"t": -1, "combined": True, "words": ["all topics combined"], "n": int(len(df)), "mps": int(len(all_ids)), "n_axes": 3,
            "dims": dims_out, "pts": [[int(m)] + [round(float(P[i, d] / scaleP[d]), 3) for d in range(3)] + [int(cover[keep][i])] for i, m in enumerate(all_ids)]}
votes = pd.read_parquet(ROOT / "data/clean/votes_long.parquet")
from sklearn.metrics import roc_auc_score
say("\nLabour votes it never saw (direction learned on other MPs; includes speech after the vote; 0.5 is chance)")
for div, nm in [(2406, "Asylum bill"), (2074, "Welfare bill"), (2078, "Palestine Action ban"), (2071, "Assisted dying")]:
    v = votes[votes.division_id == div].drop_duplicates("member_id", keep=False).set_index("member_id").vote
    m = np.array([i in v.index for i in all_ids]) & lab
    y = (v.reindex(all_ids[m]).to_numpy() == 1).astype(int)
    res = np.zeros(m.sum())
    perm = np.random.default_rng(1).permutation(m.sum())
    for fold in np.array_split(perm, 5):
        train = np.setdiff1d(perm, fold)
        wts = np.array([np.corrcoef(P[m][train, d], y[train])[0, 1] for d in range(3)])
        res[fold] = P[m][fold] @ np.nan_to_num(wts)
    say(f"  {nm:22} n={m.sum()} minority={min(y.sum(), (1 - y).sum())}: {roc_auc_score(y, res):.2f}")

# ---- Two-part space: what MPs care about, and where they stand ----
def describe(Pm, ids_, d, party_):
    g = pd.Series(Pm[:, d]).groupby(party_)
    return float(((g.transform("mean") - Pm[:, d].mean()) ** 2).sum() / ((Pm[:, d] - Pm[:, d].mean()) ** 2).sum())


def composite(Pm, Pa_, Pb_, ids_, ends, title, desc, cover_):
    party_ = members.party_abbr.reindex(ids_).fillna("?").to_numpy()
    labm = party_ == "Lab"
    ok_ = ~np.isnan(Pa_[:, 0]) & ~np.isnan(Pb_[:, 0])
    dims = []
    for d in range(3):
        c = float(np.corrcoef(Pa_[ok_, d], Pb_[ok_, d])[0, 1])
        cl = float(np.corrcoef(Pa_[ok_ & labm, d], Pb_[ok_ & labm, d])[0, 1])
        eta = describe(Pm, ids_, d, party_)
        dims.append({"held": round(abs(c), 2), "held_lab": round(abs(cl), 2), "party": round(eta, 2), "lo": ends[d][0], "hi": ends[d][1]})
        o_ = np.argsort(Pm[:, d])
        say(f"  dim {d}: separate debates {abs(c):.2f} (within Labour {abs(cl):.2f}), party explains {eta:.0%}")
        say(f"     one end : {[e['topic'] for e in ends[d][0]]}  MPs {[members.name[ids_[i]] for i in o_[:5]]}")
        say(f"     other   : {[e['topic'] for e in ends[d][1]]}  MPs {[members.name[ids_[i]] for i in o_[-5:]]}")
    sc_ = np.nanpercentile(np.abs(Pm), 97, axis=0).clip(1e-6)
    return {"t": -1, "combined": True, "title": title, "desc": desc, "words": [title.lower()], "n": int(len(df)), "mps": int(len(ids_)), "n_axes": 3, "dims": dims,
            "pts": [[int(m)] + [round(float(Pm[i, d] / sc_[d]), 3) for d in range(3)] + [int(cover_[i])] for i, m in enumerate(ids_)]}


topic_name = {o["t"]: ", ".join(o["words"][:2]) for o in out}
real_topics = [o["t"] for o in out if not o.get("skip")]

# What MPs care about: each MP's share of speeches across every topic, shrunk towards the average for quiet MPs
say("\nWHAT MPs CARE ABOUT")
CARE_SHRINK = 15


def care_matrix(frame, ids_):
    u = frame.drop_duplicates(["member", "speech", "thing"])
    M = u.groupby(["member", "thing"]).size().unstack(fill_value=0).reindex(index=ids_, columns=real_topics, fill_value=0).to_numpy(float)
    return M


sp_count = df.groupby("member").speech.nunique()
care_ids = sp_count.index[sp_count >= 15].to_numpy()
Mc = care_matrix(df, care_ids)
base_c = Mc.sum(0) / Mc.sum()
mixc = lambda M: np.sqrt((M + CARE_SHRINK * base_c) / (M.sum(1, keepdims=True) + CARE_SHRINK)) - np.sqrt(base_c)
Vc = mixc(Mc)
mu_c = Vc.mean(0)
_, sc_c, Vt_c = np.linalg.svd(Vc - mu_c, full_matrices=False)
Pc = (Vc - mu_c) @ Vt_c[:3].T
say(f"{len(care_ids)} MPs with 15+ speeches; variance on the first three dimensions {np.round(sc_c[:3] ** 2 / (sc_c ** 2).sum(), 3).tolist()}")
deb_half = df.debate.map(debate_half).to_numpy()
Pca = (mixc(care_matrix(df[deb_half == 0], care_ids)) - mu_c) @ Vt_c[:3].T
Pcb = (mixc(care_matrix(df[deb_half == 1], care_ids)) - mu_c) @ Vt_c[:3].T
enough_a = care_matrix(df[deb_half == 0], care_ids).sum(1) >= 6
enough_b = care_matrix(df[deb_half == 1], care_ids).sum(1) >= 6
Pca[~enough_a], Pcb[~enough_b] = np.nan, np.nan
ends_c = []
for d in range(3):
    o_ = np.argsort(Vt_c[d])
    ends_c.append(([{"topic": topic_name[real_topics[j]], "words": []} for j in o_[:4]], [{"topic": topic_name[real_topics[j]], "words": []} for j in o_[::-1][:4]]))
care = composite(Pc, Pca, Pcb, care_ids, ends_c, "What MPs care about",
                 "Each MP placed by their share of speeches across all {n} topics, including topics with no axis. MPs sit close when they talk about the same mix of things. This says nothing about which side they take.".format(n=len(real_topics)),
                 Mc.sum(1))

# Where MPs stand: only the main axis of the topics that look most like a stance
say("\nWHERE MPs STAND")
STANCE_TOPICS = 30
ranked = st.assign(t=[o["t"] for o in done]).sort_values("stance_score", ascending=False)
chosen_t = [t for t, n in zip(ranked.t, ranked.n_axes) if n >= 1][:STANCE_TOPICS]
pick = [i for i, (t, d, _) in enumerate(columns) if d == 0 and t in chosen_t]
say(f"using the main axis of {len(pick)} topics: {[topic_name[columns[i][0]] for i in pick]}")
As = pd.concat([columns[i][2].rename(str(i)) for i in pick], axis=1)
ids_s_all = As.index.to_numpy()
Hs = [pd.concat([half_cols[i][h].rename(str(i)) for i in pick], axis=1).reindex(ids_s_all).to_numpy() for h in (0, 1)]
As = As.to_numpy()
cov_s = (~np.isnan(As)).sum(1)
keep_s = cov_s >= 3
ids_s, As, Hs = ids_s_all[keep_s], As[keep_s], [h[keep_s] for h in Hs]
Lw_s, mu_s, sd_s, expl_s = soft_impute(As)
Ps = place(As, Lw_s, mu_s, sd_s)
Shs = As.copy()
for j in range(Shs.shape[1]):
    o = np.where(~np.isnan(Shs[:, j]))[0]
    Shs[o, j] = Shs[np.random.default_rng(5).permutation(o), j]
_, _, _, expl_sn = soft_impute(Shs)
say(f"{len(ids_s)} MPs with a position on 3+ of these axes; {np.mean(~np.isnan(As)):.0%} of cells known; variance {np.round(expl_s, 3).tolist()} vs shuffled {np.round(expl_sn, 3).tolist()}")
Psa, Psb = place(Hs[0], Lw_s, mu_s, sd_s), place(Hs[1], Lw_s, mu_s, sd_s)
ends_s = []
for d in range(3):
    o_ = np.argsort(Lw_s[:, d])
    def e_(j, sgn):
        t_, ax_, _ = columns[pick[j]]
        oo = next(x for x in out if x["t"] == t_)
        return {"topic": topic_name[t_] + ": " + ", ".join(oo["axes"][0]["hi" if Lw_s[j, d] * sgn > 0 else "lo"][:2]), "words": []}
    ends_s.append(([e_(j, -1) for j in o_[:4]], [e_(j, 1) for j in o_[::-1][:4]]))
stance = composite(Ps, Psa, Psb, ids_s, ends_s, "Where MPs stand",
                   f"MPs placed only on the main axis of the {len(pick)} topics that look most like a stance rather than a focus. MPs sit close when they fall on similar sides of those divides. Coverage is thin, so many positions rest on few topics.",
                   cov_s[keep_s])
say("\nLabour votes they never saw (direction learned on other MPs; includes speech after the vote; 0.5 is chance)")
for label_, Pm_, ids_ in (("care", Pc, care_ids), ("stance", Ps, ids_s)):
    party_ = members.party_abbr.reindex(ids_).fillna("?").to_numpy()
    for div, nm in [(2406, "Asylum bill"), (2074, "Welfare bill"), (2071, "Assisted dying")]:
        v = votes[votes.division_id == div].drop_duplicates("member_id", keep=False).set_index("member_id").vote
        mm = np.array([i in v.index for i in ids_]) & (party_ == "Lab") & ~np.isnan(Pm_[:, 0])
        y = (v.reindex(ids_[mm]).to_numpy() == 1).astype(int)
        if min(y.sum(), (1 - y).sum()) < 4:
            continue
        res = np.zeros(mm.sum())
        perm = np.random.default_rng(1).permutation(mm.sum())
        for fold in np.array_split(perm, 5):
            train = np.setdiff1d(perm, fold)
            wts = np.array([np.corrcoef(Pm_[mm][train, d], y[train])[0, 1] for d in range(3)])
            res[fold] = Pm_[mm][fold] @ np.nan_to_num(wts)
        say(f"  {label_:6} {nm:16} n={mm.sum()} minority={min(y.sum(), (1 - y).sum())}: {roc_auc_score(y, res):.2f}")
combined["title"] = "All divides combined"
combined["desc"] = None
(ROOT / "data/quick/topic_axes.json").write_text(json.dumps({"topics": out, "combined": combined, "care": care, "stance": stance}, ensure_ascii=False))
(ROOT / "data/quick/topic_axes_report.txt").write_text("\n".join(lines))
say(f"\nDONE {time.time() - t0:.0f}s")
