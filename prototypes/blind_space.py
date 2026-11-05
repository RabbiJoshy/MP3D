"""A map of MPs built blind. No human-written axes, and no votes, party or names go into the fit.

1. Inside each topic, find the main axes of difference in word use (backbench and opposition text only).
2. Keep only axes that line up with other topics. A sub-topic like "dentists vs pharmacies" lines up with nothing
   and drops out; a position that recurs across subjects survives. Selection is repeated, and the same selection
   is run on shuffled data so its inflation can be measured.
3. Grow the map through time: MPs are placed using only what they had said by each date.
4. Only then check it against what it wasn't taught: party, rebellions, Jev's independent reading, axes' own words.

Human choices that remain: which words count as procedure, the topic clustering settings, the number of dimensions.
Out: data/quick/blind.npz, blind_report.txt
"""
import os
import pathlib

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

ROOT = pathlib.Path(__file__).resolve().parent.parent
AXES_PER_TOPIC, MIN_PASSAGES, MIN_AXES, RANK, SEED = 3, 3, 6, 3, 7
KEEP_ROUNDS = [0.6, 0.6, 0.6]
rng = np.random.default_rng(SEED)
STOP = """hon right friend friends member members gentleman lady minister ministers secretary state government
house bill clause clauses amendment amendments committee debate speaker sir madam people country uk british
britain think want wants know need needs make makes made said say says going just like does did ask asked agree
important new way ensure work working year years time today constituency constituents local support issue issues
point points absolutely thank thanks labour conservative conservatives liberal democrat democrats snp reform party
parties tory tories opposition lord lords chancellor prime shadow previous last plaid cymru green greens
sinn dup farage badenoch starmer""".split()

lines = []
def say(*a):
    t = " ".join(str(x) for x in a)
    print(t, flush=True)
    lines.append(t)

q = np.load(ROOT / "data/quick/topic_space.npz", allow_pickle=True)
df = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "member_id", "text", "role", "date"])
df["topic"] = q["topic"]
real = np.where(~q["is_chatter"])[0]
back = df[df.role != "minister"]
members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
votes_long = pd.read_parquet(ROOT / "data/clean/votes_long.parquet")

models = {}
for k in real:
    sub = back[back.topic == k]
    n = sub.member_id.value_counts()
    ids_k = n.index[n >= MIN_PASSAGES]
    if len(ids_k) < 25:
        continue
    docs = sub[sub.member_id.isin(ids_k)].groupby("member_id").text.apply(" ".join)
    vec = TfidfVectorizer(stop_words=sorted(ENGLISH_STOP_WORDS | set(STOP)), ngram_range=(1, 2), min_df=3, max_df=0.6, max_features=6000,
                          sublinear_tf=True, token_pattern=r"[a-zA-Z][a-zA-Z\-]{2,}")
    X = normalize(vec.fit_transform(docs)).toarray()
    models[k] = (vec, PCA(AXES_PER_TOPIC, random_state=SEED).fit(X), np.array(vec.get_feature_names_out()))
axis_topic = np.repeat(list(models), AXES_PER_TOPIC)
axis_index = np.tile(np.arange(AXES_PER_TOPIC), len(models))


def score(frame, ids=None):
    ids = np.array(sorted(frame.member_id.unique())) if ids is None else ids
    pos = {m: i for i, m in enumerate(ids)}
    M = np.full((len(ids), len(axis_topic)), np.nan)
    for t, k in enumerate(models):
        vec, pca, _ = models[k]
        sub = frame[(frame.topic == k) & frame.member_id.isin(pos)]
        n = sub.member_id.value_counts()
        keep = n.index[n >= MIN_PASSAGES]
        if not len(keep):
            continue
        docs = sub[sub.member_id.isin(keep)].groupby("member_id").text.apply(" ".join)
        Z = pca.transform(normalize(vec.transform(docs)).toarray())
        counts = n.reindex(docs.index).to_numpy()
        M[[pos[m] for m in docs.index], t * AXES_PER_TOPIC:(t + 1) * AXES_PER_TOPIC] = Z * (counts / (counts + 4))[:, None]
    return ids, M


def factorise(M, rank=RANK, iters=120, shrink=2.0):
    mask = ~np.isnan(M)
    mean, sd = np.nanmean(M, axis=0), np.nanstd(M, axis=0) + 1e-9
    Y = np.where(mask, (M - mean) / sd, 0.0)
    fill = Y.copy()
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(fill, full_matrices=False)
        s = np.maximum(s - shrink, 0)
        fill = np.where(mask, Y, (U[:, :rank] * s[:rank]) @ Vt[:rank])
    low = (U[:, :rank] * s[:rank]) @ Vt[:rank]
    fit = np.array([np.corrcoef(Y[mask[:, j], j], low[mask[:, j], j])[0, 1] if mask[:, j].sum() > 10 and low[mask[:, j], j].std() > 0 else 0 for j in range(Y.shape[1])])
    expl = np.array([np.sum(((U[:, [r]] * s[r]) @ Vt[[r]])[mask] ** 2) for r in range(rank)]) / np.sum(Y[mask] ** 2)
    return {"L": Vt[:rank].T, "s": s[:rank], "mean": mean, "sd": sd, "expl": expl, "fit": np.nan_to_num(fit)}


def select(M, label):
    """Keep the axes the shared structure explains best, repeatedly."""
    columns = np.arange(M.shape[1])
    for keep_share in KEEP_ROUNDS:
        f = factorise(M[:, columns], iters=60)
        columns = columns[np.argsort(-f["fit"])[:max(12, int(len(columns) * keep_share))]]
    f = factorise(M[:, columns])
    say(f"{label}: {len(columns)} axes kept; variance by dimension {np.round(f['expl'], 3).tolist()}, total {f['expl'].sum():.3f}")
    return columns, f


def project(M, f, columns, ridge=1.0):
    out = np.zeros((len(M), f["L"].shape[1]))
    X = M[:, columns]
    for i, row in enumerate(X):
        o = ~np.isnan(row)
        if o.any():
            L = f["L"][o]
            out[i] = np.linalg.solve(L.T @ L + ridge * np.eye(L.shape[1]), L.T @ ((row[o] - f["mean"][columns][o]) / f["sd"][columns][o]))
    return out


ids_all, M_all = score(back)
keep_mp = (~np.isnan(M_all)).sum(1) >= MIN_AXES
ids, M = ids_all[keep_mp], M_all[keep_mp]
party = members.party_abbr.reindex(ids).fillna("?").to_numpy()
say(f"MPs placed: {len(ids)}")
columns, f = select(M, "real data")
Z = project(M, f, columns)

S = M.copy()
for j in range(S.shape[1]):
    o = np.where(~np.isnan(S[:, j]))[0]
    S[o, j] = S[rng.permutation(o), j]
select(S, "shuffled data, same selection")

half = rng.random(len(back)) < 0.5
ids_a, Ma = score(back[half], ids)
ids_b, Mb = score(back[~half], ids)
A, B = project(Ma, f, columns), project(Mb, f, columns)
say("split-half correlation by dimension:", [round(float(np.corrcoef(A[:, d], B[:, d])[0, 1]), 2) for d in range(RANK)])


def poles(d):
    """Each end of a dimension, described by its own words across all the axes that feed it."""
    weights = {}
    for j, col in enumerate(columns):
        k, a = axis_topic[col], axis_index[col]
        vec, pca, terms = models[k]
        w = pca.components_[a] * f["L"][j, d]
        for i in np.argsort(-np.abs(w))[:15]:
            weights[terms[i]] = weights.get(terms[i], 0) + w[i]
    s = pd.Series(weights).sort_values()
    return s.index[:14].tolist(), s.index[::-1][:14].tolist()


say("\nDIMENSIONS (words and MPs at each end, chosen by the data)")
pole_words = []
for d in range(RANK):
    lo, hi = poles(d)
    pole_words.append((lo, hi))
    g = pd.Series(Z[:, d]).groupby(party)
    eta = float(((g.transform("mean") - Z[:, d].mean()) ** 2).sum() / ((Z[:, d] - Z[:, d].mean()) ** 2).sum())
    means = g.mean()[[p for p in ["Lab", "Con", "LD", "RUK", "SNP", "Green"] if p in set(party)]].round(2).to_dict()
    order = np.argsort(Z[:, d])
    say(f"\n-- dim {d}: party explains {eta:.0%} | {means}")
    say("   low words :", lo)
    say("   high words:", hi)
    say("   low MPs :", [f"{members.name[ids[i]]} ({party[i]})" for i in order[:8]])
    say("   high MPs:", [f"{members.name[ids[i]]} ({party[i]})" for i in order[-8:]])
say("topics feeding the map:", pd.Series([q["top_terms"][axis_topic[c]][0] for c in columns]).value_counts().head(25).to_dict())


def best_axis_auc(z, y):
    # Direction is taken from the training folds, so the sign can't be chosen with hindsight
    idx = rng.permutation(len(y))
    out = np.zeros(len(y))
    for fold in np.array_split(idx, 5):
        train = np.setdiff1d(idx, fold)
        c = np.array([np.corrcoef(z[train, d], y[train])[0, 1] for d in range(z.shape[1])])
        out[fold] = z[fold] @ np.nan_to_num(c)
    return roc_auc_score(y, out)


TESTS = [(2071, "Assisted dying (free)"), (2058, "Abortion (free)"), (2074, "Welfare bill"), (2078, "Palestine Action ban"), (2406, "Asylum bill"), (2253, "Two-child limit")]
say("\nCHECKS AGAINST VOTES IT NEVER SAW (held-out direction; 0.5 is chance)")
for div, name in TESTS:
    v = votes_long[votes_long.division_id == div].drop_duplicates("member_id", keep=False).set_index("member_id").vote
    m = np.array([i in v.index for i in ids])
    y = (v.reindex(ids[m]).to_numpy() == 1).astype(int)
    row = f"{name:24} all {best_axis_auc(Z[m], y):.2f}"
    for p in ["Lab", "Con"]:
        pm = party[m] == p
        if min(y[pm].sum(), (1 - y[pm]).sum()) >= 8:
            row += f" | {p} n={pm.sum()} minority={min(y[pm].sum(), (1 - y[pm]).sum())} {best_axis_auc(Z[m][pm], y[pm]):.2f}"
    say(row)

# Inside a party: a separate blind map, where government vs opposition can't explain anything
lab = party == "Lab"
good = (~np.isnan(M[lab])).sum(0) >= 25
lab_cols = np.where(good)[0]
fl = factorise(M[lab][:, lab_cols])
sel = lab_cols[np.argsort(-fl["fit"])[:int(len(lab_cols) * 0.3)]]
fl = factorise(M[lab][:, sel])
Zl = project(M[lab], fl, sel)
lab_ids = ids[lab]
say("\nINSIDE LABOUR: variance", np.round(fl["expl"], 3).tolist(), f"{len(sel)} axes")
for d in range(RANK):
    order = np.argsort(Zl[:, d])
    say(f"  dim {d} low :", [members.name[lab_ids[i]] for i in order[:8]])
    say(f"  dim {d} high:", [members.name[lab_ids[i]] for i in order[-8:]])
    row = []
    for div, name in TESTS:
        v = votes_long[votes_long.division_id == div].drop_duplicates("member_id", keep=False).set_index("member_id").vote
        m = np.array([i in v.index for i in lab_ids])
        y = (v.reindex(lab_ids[m]).to_numpy() == 1).astype(int)
        if min(y.sum(), (1 - y).sum()) >= 8:
            a = roc_auc_score(y, Zl[m, d])
            row.append(f"{name} {a:.2f}")
    say(f"  dim {d} vs votes (unflipped):", row)

# Independent check: does the blind map agree with Jev reading the asylum passages?
jev_path = ROOT / "data/quick/stance_asylum_jev.jsonl"
if jev_path.exists():
    j = pd.DataFrame(json.loads(l) for l in jev_path.read_text().splitlines())
    j = j.rename(columns={"id": "passage_id"}).merge(df[["passage_id", "member_id"]], on="passage_id").dropna(subset=["stance"])
    c = j.member_id.value_counts()
    jm = j[j.member_id.isin(c.index[c >= 3])].groupby("member_id").stance.mean()
    # Exclude the asylum topic's own axes, so agreement can't come from the same passages
    no_asylum = columns[axis_topic[columns] != 28]
    fx = factorise(M[:, no_asylum])
    Zx = project(M, fx, no_asylum)
    idx = [i for i, m in enumerate(ids) if m in jm.index]
    y = jm.reindex(ids[idx]).to_numpy()
    say("\nJEV'S ASYLUM STANCE vs BLIND MAP BUILT WITHOUT ANY ASYLUM TEXT:", f"{len(idx)} MPs")
    for d in range(RANK):
        lab_here = party[idx] == "Lab"
        say(f"  dim {d}: correlation all {np.corrcoef(Zx[idx, d], y)[0, 1]:+.2f} | within Labour {np.corrcoef(Zx[idx, d][lab_here], y[lab_here])[0, 1]:+.2f} ({lab_here.sum()} MPs)")

# Grow the map through time, placing MPs with only what they had said by each date
dates = sorted(back.date.unique())
cuts = [dates[int(len(dates) * x) - 1] for x in (0.08, 0.16, 0.28, 0.42, 0.58, 0.74, 0.88, 1.0)]
frames = []
for cut in cuts:
    _, Mt = score(back[back.date <= cut], ids)
    Zt = project(Mt, f, columns)
    known = (~np.isnan(Mt[:, columns])).any(1).mean()
    frames.append(Zt)
    lc = np.isin(party, ["Lab", "Con"])
    a = roc_auc_score(party[lc] == "Con", Zt[lc, 0])
    say(f"up to {cut}: {known:.0%} of MPs have any position yet | spread {Zt.std():.2f} | Lab-vs-Con on dim 0 {max(a, 1 - a):.2f}")
np.savez(ROOT / "data/quick/blind.npz", ids=ids, Z=Z, Zl=Zl, lab_ids=lab_ids, frames=np.array(frames), cuts=np.array(cuts), expl=f["expl"],
         poles=np.array(pole_words, dtype=object))
(ROOT / "data/quick/blind_report.txt").write_text("\n".join(lines))
