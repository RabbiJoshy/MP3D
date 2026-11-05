"""Place MPs by what they argue, not what they talk about. Unsupervised: votes and party are only used to test it.

1. Inside each topic, find the main axes of difference in word use between MPs (backbench/opposition text only).
2. Score each MP on each axis, leaving it unknown if they've barely spoken on that topic.
3. Find the shared structure: low-rank factorisation of MPs x axes that handles missing values. Axes that only
   describe sub-topics don't co-vary with anything, so they fade out. Positions that travel together survive.
4. Test: chance baseline, split-half reliability, and cross-validated prediction of votes not used in the fit,
   including within a party and with the vote's own topic held out.

Out: data/quick/polarity.npz, polarity_report.txt
"""
import os
import pathlib
import sys

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import normalize

ROOT = pathlib.Path(__file__).resolve().parent.parent
AXES_PER_TOPIC, MIN_PASSAGES, MIN_AXES, SEED = 3, 3, 12, 7
RANK = 6
rng = np.random.default_rng(SEED)

# Party names and "the House" style words are removed so a party can't be spotted by name alone
STOP = """hon right friend friends member members gentleman lady minister ministers secretary state government
house bill clause clauses amendment amendments committee debate speaker sir madam people country uk british
britain think want wants know need needs make makes made said say says going just like does did ask asked agree
important new way ensure work working year years time today constituency constituents local support issue issues
point points absolutely thank thanks labour conservative conservatives liberal democrat democrats snp reform party
parties tory tories opposition lord lords chancellor prime shadow previous last plaid cymru green greens
sinn dup farage badenoch starmer""".split()

q = np.load(ROOT / "data/quick/topic_space.npz", allow_pickle=True)
df = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["member_id", "text", "role"])
df["topic"] = q["topic"]
real = np.where(~q["is_chatter"])[0]
members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
votes_long = pd.read_parquet(ROOT / "data/clean/votes_long.parquet")
back = df[df.role != "minister"]
print(f"backbench/opposition passages: {len(back)} of {len(df)}", flush=True)

lines = []
def say(*a):
    t = " ".join(str(x) for x in a)
    print(t, flush=True)
    lines.append(t)

# Step 1: find the axes inside each topic
models = {}
for k in real:
    sub = back[back.topic == k]
    n = sub.member_id.value_counts()
    ids = n.index[n >= MIN_PASSAGES]
    if len(ids) < 25:
        continue
    docs = sub[sub.member_id.isin(ids)].groupby("member_id").text.apply(" ".join)
    vec = TfidfVectorizer(stop_words=sorted(ENGLISH_STOP_WORDS | set(STOP)), ngram_range=(1, 2), min_df=3, max_df=0.6, max_features=6000,
                          sublinear_tf=True, token_pattern=r"[a-zA-Z][a-zA-Z\-]{2,}")
    X = normalize(vec.fit_transform(docs)).toarray()
    pca = PCA(AXES_PER_TOPIC, random_state=SEED).fit(X)
    models[k] = (vec, pca, np.array(vec.get_feature_names_out()))
say(f"topics with enough speakers: {len(models)} of {len(real)}")
axis_topic = np.repeat(list(models), AXES_PER_TOPIC)
axis_index = np.tile(np.arange(AXES_PER_TOPIC), len(models))


def score(frame):
    """MPs x axes matrix for some passages. NaN means too little text to say."""
    ids = np.array(sorted(frame.member_id.unique()))
    pos = {m: i for i, m in enumerate(ids)}
    M = np.full((len(ids), len(axis_topic)), np.nan)
    for t, k in enumerate(models):
        vec, pca, _ = models[k]
        sub = frame[frame.topic == k]
        n = sub.member_id.value_counts()
        keep = n.index[n >= MIN_PASSAGES]
        if not len(keep):
            continue
        docs = sub[sub.member_id.isin(keep)].groupby("member_id").text.apply(" ".join)
        Z = pca.transform(normalize(vec.transform(docs)).toarray())
        Z = Z * (n.reindex(docs.index).to_numpy() / (n.reindex(docs.index).to_numpy() + 4))[:, None]  # shrink thin evidence
        M[[pos[m] for m in docs.index], t * AXES_PER_TOPIC:(t + 1) * AXES_PER_TOPIC] = Z
    return ids, M


def factorise(M, rank=RANK, iters=120, shrink=2.0):
    """Soft-impute: low-rank structure from an incomplete matrix, without treating silence as neutral."""
    mask = ~np.isnan(M)
    mean = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0) + 1e-9
    Y = np.where(mask, (M - mean) / sd, 0.0)
    fill = Y.copy()
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(fill, full_matrices=False)
        s = np.maximum(s - shrink, 0)
        low = (U[:, :rank] * s[:rank]) @ Vt[:rank]
        fill = np.where(mask, Y, low)
    # Share of observed variance each dimension explains
    expl = np.array([np.sum(((U[:, [r]] * s[r]) @ Vt[[r]])[mask] ** 2) for r in range(rank)]) / np.sum(Y[mask] ** 2)
    return {"L": Vt[:rank].T, "s": s[:rank], "mean": mean, "sd": sd, "expl": expl}


def project(M, f, ridge=1.0):
    """Place each MP using only the axes they have values for."""
    out = np.zeros((len(M), f["L"].shape[1]))
    for i, row in enumerate(M):
        o = ~np.isnan(row)
        if o.any():
            L = f["L"][o]
            out[i] = np.linalg.solve(L.T @ L + ridge * np.eye(L.shape[1]), L.T @ ((row[o] - f["mean"][o]) / f["sd"][o]))
    return out


def auc(score_, y):
    return roc_auc_score(y, score_)


def fold_auc(Z, y):
    if min((y == 0).sum(), (y == 1).sum()) < 8:
        return float("nan")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = cross_val_predict(LogisticRegression(C=0.5, max_iter=500, class_weight="balanced"), Z, y, cv=cv, method="predict_proba")[:, 1]
    return auc(p, y)


ids_all, M_all = score(back)
coverage = (~np.isnan(M_all)).sum(1)
keep_mp = coverage >= MIN_AXES
ids, M = ids_all[keep_mp], M_all[keep_mp]
say(f"MPs placed: {len(ids)} (others spoke on too few topics outside government posts); share of MP x axis cells known: {(~np.isnan(M)).mean():.1%}")
party = members.party_abbr.reindex(ids).fillna("?").to_numpy()
say("MPs placed by party:", pd.Series(party).value_counts().head(8).to_dict())
f = factorise(M)
Z = project(M, f)
say("variance explained by each dimension:", np.round(f["expl"], 3).tolist(), "total", round(float(f["expl"].sum()), 3))

# Chance baseline: shuffle MPs within each axis, which keeps every axis but destroys positions travelling together
null = []
for _ in range(3):
    S = M.copy()
    for j in range(S.shape[1]):
        o = np.where(~np.isnan(S[:, j]))[0]
        S[o, j] = S[rng.permutation(o), j]
    null.append(factorise(S, iters=60)["expl"])
null = np.mean(null, axis=0)
say("same, with positions shuffled (chance):", np.round(null, 3).tolist(), "total", round(float(null.sum()), 3))

# Split-half: do two separate samples of an MP's text put them in the same place?
half = rng.random(len(back)) < 0.5
ids_a, Ma = score(back[half])
ids_b, Mb = score(back[~half])
A = pd.DataFrame(project(Ma, f), index=ids_a)
B = pd.DataFrame(project(Mb, f), index=ids_b)
common = A.index.intersection(B.index).intersection(ids)
say("split-half correlation by dimension:", [round(float(np.corrcoef(A.loc[common, d], B.loc[common, d])[0, 1]), 2) for d in range(RANK)], f"({len(common)} MPs)")

# How much of each dimension is just party?
say("\nDIMENSIONS")
for d in range(RANK):
    g = pd.Series(Z[:, d]).groupby(party)
    ss = (g.transform("mean") - Z[:, d].mean()) ** 2
    eta = float(ss.sum() / ((Z[:, d] - Z[:, d].mean()) ** 2).sum())
    means = g.mean()[[p for p in ["Lab", "Con", "LD", "RUK", "SNP", "Green"] if p in set(party)]].round(2).to_dict()
    say(f"\n-- dim {d}: party explains {eta:.0%} | party means {means}")
    order = np.argsort(Z[:, d])
    say("   low MPs :", [f"{members.name[ids[i]]} ({party[i]})" for i in order[:7]])
    say("   high MPs:", [f"{members.name[ids[i]]} ({party[i]})" for i in order[-7:]])
    load = f["L"][:, d]
    for sign, label in ((-1, "low"), (1, "high")):
        say(f"   drives {label}:")
        for j in np.argsort(-sign * load)[:6]:
            k, a = axis_topic[j], axis_index[j]
            vec, pca, terms = models[k]
            w = pca.components_[a] * np.sign(load[j]) * sign
            say(f"      t{k:<3} {q['top_terms'][k][0]:<14} {', '.join(terms[np.argsort(-w)[:6]])}")

# Votes never entered the fit, so these are real checks
TESTS = [(2071, "Assisted dying (free)", [48, 54, 53]), (1877, "Assisted dying 2nd reading (free)", [48, 54, 53]),
         (2058, "Abortion decriminalisation (free)", [87, 2]), (2074, "Welfare bill", [61, 58]), (2078, "Palestine Action ban", [34, 66, 62]),
         (2406, "Immigration & Asylum Bill", [28]), (1874, "Tobacco bill", [31]), (2253, "Two-child limit", [61])]
dummy = pd.get_dummies(party).to_numpy().astype(float)
say("\nVOTE TESTS (cross-validated AUC; 0.5 is chance. 'no topic' drops the vote's own topic from the space)")
for div, name, drop in TESTS:
    v = votes_long[votes_long.division_id == div].drop_duplicates("member_id", keep=False).set_index("member_id").vote
    idx = np.array([i for i, m in enumerate(ids) if m in v.index])
    y = (v.reindex(ids[idx]).to_numpy() == 1).astype(int)
    columns = ~np.isin(axis_topic, drop)
    f2 = factorise(M[:, columns])
    Z2 = project(M[:, columns], f2)
    row = f"{name:36} all: party {fold_auc(dummy[idx], y):.2f} space {fold_auc(Z[idx], y):.2f} no-topic {fold_auc(Z2[idx], y):.2f}"
    for p in ["Lab", "Con"]:
        pm = party[idx] == p
        if pm.sum() > 40 and min(y[pm].sum(), (1 - y[pm]).sum()) >= 8:
            row += f" | {p} n={pm.sum()} minority={min(y[pm].sum(), (1 - y[pm]).sum())}: space {fold_auc(Z[idx][pm], y[pm]):.2f} no-topic {fold_auc(Z2[idx][pm], y[pm]):.2f}"
    say(row)

# Fit inside Labour alone, where government-vs-opposition can't be the explanation
lab = party == "Lab"
fl = factorise(M[lab][:, (~np.isnan(M[lab])).sum(0) >= 25])
keep_cols = (~np.isnan(M[lab])).sum(0) >= 25
Zl = project(M[lab][:, keep_cols], fl)
say("\nLABOUR-ONLY SPACE: variance", np.round(fl["expl"], 3).tolist())
lab_ids = ids[lab]
for d in range(3):
    order = np.argsort(Zl[:, d])
    say(f"  dim {d} low :", [members.name[lab_ids[i]] for i in order[:8]])
    say(f"  dim {d} high:", [members.name[lab_ids[i]] for i in order[-8:]])
axis_cols = np.where(keep_cols)[0]
for d in range(2):
    for sign, label in ((-1, "low"), (1, "high")):
        for j in np.argsort(-sign * fl["L"][:, d])[:4]:
            full = axis_cols[j]
            k, a = axis_topic[full], axis_index[full]
            vec, pca, terms = models[k]
            w = pca.components_[a] * np.sign(fl["L"][j, d]) * sign
            say(f"  lab dim {d} {label} t{k} {q['top_terms'][k][0]}: {', '.join(terms[np.argsort(-w)[:6]])}")
for d in range(3):
    row = []
    for div, name, _ in TESTS[:6]:
        v = votes_long[votes_long.division_id == div].drop_duplicates("member_id", keep=False).set_index("member_id").vote
        idx = np.array([i for i, m in enumerate(lab_ids) if m in v.index])
        y = (v.reindex(lab_ids[idx]).to_numpy() == 1).astype(int)
        if min(y.sum(), (1 - y).sum()) >= 8:
            a = auc(Zl[idx, d], y)
            row.append(f"{name[:14]} {max(a, 1 - a):.2f}")
    say(f"  lab dim {d} vs votes:", row)

# The 2024 map's failure: Corbyn was indistinguishable from average Labour
centre = Z[lab].mean(0)
dist = np.linalg.norm(Z - centre, axis=1)
scale = np.median(dist[lab])
say("\nDISTANCE FROM LABOUR AVERAGE (1.0 = typical Labour MP)")
for nm in ["Jeremy Corbyn", "Zarah Sultana", "Richard Burgon", "Diane Abbott", "John McDonnell", "Rosie Duffield", "Nigel Farage", "Kemi Badenoch", "Ed Davey", "Danny Kruger", "Suella Braverman", "Robert Jenrick"]:
    i = next((j for j, m in enumerate(ids) if nm in members.name[m]), None)
    say(f"  {nm:18}", "not placed" if i is None else f"{dist[i] / scale:.2f} ({party[i]})")
unit = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
sim = unit @ unit.T
np.fill_diagonal(sim, -1)
say("\nNEAREST MPs")
for nm in ["Jeremy Corbyn", "Zarah Sultana", "Nigel Farage", "Kemi Badenoch", "Tim Farron", "Rosie Duffield", "Danny Kruger", "Kim Leadbeater", "Jim Shannon"]:
    i = next((j for j, m in enumerate(ids) if nm in members.name[m]), None)
    if i is not None:
        say(f"  {nm:16}", [f"{members.name[ids[j]]} ({party[j]})" for j in np.argsort(-sim[i])[:7]])
# Same-party share among nearest MPs: high means the space is mostly a party map
nn = np.argsort(-sim, axis=1)[:, :10]
big = np.isin(party, ["Lab", "Con", "LD"])
say("10 nearest in own party:", {p: round(float((party[nn[party == p]] == p).mean()), 2) for p in ["Lab", "Con", "LD"]}, "| random:", {p: round(float((party == p).mean()), 2) for p in ["Lab", "Con", "LD"]})

np.savez(ROOT / "data/quick/polarity.npz", lab_ids=lab_ids, Zl=Zl, ids=ids, Z=Z, L=f["L"], axis_topic=axis_topic, axis_index=axis_index, expl=f["expl"])
(ROOT / "data/quick/polarity_report.txt").write_text("\n".join(lines))
