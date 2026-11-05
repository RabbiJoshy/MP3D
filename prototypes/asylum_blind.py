"""A small, blind map of one debate: asylum. No paid calls, no handwritten axes, stopwords or phrase rules.

Text:  original Hansard speeches, not the heavily cleaned passages. Stage directions and motion text are dropped using
       Hansard's own markup. Common words are dropped by how often they appear across a sample of all Hansard, and
       names, constituencies and parties are dropped using Parliament's own member data.
Shape: each sentence becomes a low-dimensional vector (TF-IDF then SVD), and sentences group into argument clusters.
Space: MPs are placed by which arguments they use more than expected. Votes, party and Jev never enter the fit.
Tests: shuffled baseline, axes fitted on some debates and tested on others, removing any one debate, speech made
       before a vote compared with that vote, and agreement with Jev's separate reading.

Out: data/quick/asylum_blind.npz, asylum_blind_report.txt
"""
import collections
import json
import os
import pathlib
import re
import xml.etree.ElementTree as ET

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/hansard"
TOPIC, SEED = 28, 7
K_ARGUMENTS, SVD_DIMS, SHRINK, BACKGROUND_DF = 30, 40, 8, 0.012
MIN_SPEECHES, MIN_DATES = 3, 2
rng = np.random.default_rng(SEED)
lines = []
def say(*a):
    t = " ".join(str(x) for x in a)
    print(t, flush=True)
    lines.append(t)

q = np.load(ROOT / "data/quick/topic_space.npz", allow_pickle=True)
p = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "member_id", "role", "date", "major", "minor", "source"])
p["topic"] = q["topic"]
a = p[(p.topic == TOPIC) & (p.role != "minister")].copy()
a["speech_id"] = a.passage_id.str.split("#").str[0]
speeches = a.drop_duplicates("speech_id").set_index("speech_id")
wanted = set(speeches.index)
say(f"asylum speeches from backbench and opposition MPs: {len(wanted)}, by {a.member_id.nunique()} MPs")

SPLIT = re.compile(r"(?<=[.?!])\s+(?=[\"“‘A-Z(])")
def speech_text(el):
    return [" ".join("".join(x.itertext()).split()) for x in el.findall("p") if x.get("class") != "italic" and x.get("pwmotiontext") != "yes"]

days = sorted(speeches.date.unique())
rows, background = [], []
all_files = sorted((RAW / "debates").glob("*.xml")) + sorted((RAW / "westminhall").glob("*.xml")) + sorted((RAW / "standing").glob("*.xml"))
sample_every = max(1, len(all_files) // 120)
for i, f in enumerate(all_files):
    day = re.search(r"\d{4}-\d{2}-\d{2}", f.name).group()
    take_background = i % sample_every == 0
    if day not in days and not take_background:
        continue
    for el in ET.parse(f).getroot().iter("speech"):
        sid = el.get("id")
        if sid in wanted:
            for para in speech_text(el):
                for sent in SPLIT.split(re.sub(r"\bhon\.\s", "hon ", para)):
                    rows.append((sid, sent))
        elif take_background and el.get("person_id"):
            background.append(" ".join(speech_text(el)))
s = pd.DataFrame(rows, columns=["speech_id", "text"])
s = s.join(speeches[["member_id", "date", "major", "minor", "source"]], on="speech_id")
s["debate"] = s.date + "|" + s.source + "|" + s.major.fillna("") + "|" + s.minor.fillna("")
say(f"sentences: {len(s)}; background speeches for common-word filter: {len(background)}")

# Words to ignore, all from data: common across Hansard, or part of a member, seat or party name
background_df = CountVectorizer(ngram_range=(1, 2), binary=True, min_df=5, token_pattern=r"[a-zA-Z][a-zA-Z\-']+", lowercase=True).fit(background)
bg = background_df.transform(background)
share = np.asarray(bg.mean(0)).ravel()
common = set(background_df.get_feature_names_out()[share > BACKGROUND_DF])
members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
identity = set()
for col in ["name", "constituency", "party"]:
    for value in members[col].dropna():
        for w in re.findall(r"[a-zA-Z][a-zA-Z\-']+", str(value).lower()):
            identity.add(w)
say(f"data-derived ignore list: {len(common)} common terms, {len(identity)} identity words. Most common kept out: {sorted(common, key=len)[:12]}")

vec = CountVectorizer(ngram_range=(1, 2), min_df=6, max_df=0.08, token_pattern=r"[a-zA-Z][a-zA-Z\-']+", lowercase=True)
C = vec.fit_transform(s.text)
terms = np.array(vec.get_feature_names_out())
def banned(t):
    parts = t.split()
    return t in common or all(w in common for w in parts) or any(w in identity for w in parts)
keep_terms = np.array([not banned(t) for t in terms])
C, terms = C[:, keep_terms], terms[keep_terms]
has = np.asarray((C > 0).sum(1)).ravel() >= 2
s, C = s[has].reset_index(drop=True), C[has]
say(f"vocabulary {len(terms)}; sentences with enough signal {len(s)}")

X = TfidfTransformer(sublinear_tf=True).fit_transform(C)
svd = TruncatedSVD(SVD_DIMS, random_state=SEED)
L = normalize(svd.fit_transform(X))
km = MiniBatchKMeans(K_ARGUMENTS, n_init=10, batch_size=2048, random_state=SEED, max_iter=300)
s["arg"] = km.fit_predict(L)
s["fit"] = (L * normalize(km.cluster_centers_)[s.arg]).sum(1)
say(f"each sentence reduced to {SVD_DIMS} numbers, then one of {K_ARGUMENTS} argument clusters")

overall_terms = np.asarray(X.mean(0)).ravel()
arg_words, arg_examples = [], []
for k in range(K_ARGUMENTS):
    m = (s.arg == k).to_numpy()
    mean = np.asarray(X[m].mean(0)).ravel()
    arg_words.append(terms[np.argsort(-(mean * np.log1p(mean / (overall_terms + 1e-6))))[:9]].tolist())
    arg_examples.append(s[m].sort_values("fit", ascending=False).text.iloc[:2].tolist())


def unit_counts(frame):
    """One piece of evidence per MP, per speech, per argument, so one long speech can't dominate."""
    return frame.drop_duplicates(["member_id", "speech_id", "arg"]).groupby(["member_id", "arg"]).size().unstack(fill_value=0).reindex(columns=range(K_ARGUMENTS), fill_value=0)


def mix(counts, base):
    c = counts.to_numpy(float)
    smooth = (c + SHRINK * base) / (c.sum(1, keepdims=True) + SHRINK)
    return np.sqrt(smooth) - np.sqrt(base)


def axes(V, k=3):
    U, sv, Vt = np.linalg.svd(V - V.mean(0), full_matrices=False)
    return Vt[:k], sv


evidence = s.groupby("member_id").agg(speeches=("speech_id", "nunique"), dates=("date", "nunique"))
ids = evidence.index[(evidence.speeches >= MIN_SPEECHES) & (evidence.dates >= MIN_DATES)]
placed = s[s.member_id.isin(ids)]
counts = unit_counts(placed).reindex(ids, fill_value=0)
base = counts.sum(0).to_numpy(float) / counts.sum().sum()
V = mix(counts, base)
Vt, sv = axes(V)
Z = (V - V.mean(0)) @ Vt.T
party = members.party_abbr.reindex(ids).fillna("?").to_numpy()
say(f"\nMPs placed (at least {MIN_SPEECHES} speeches on {MIN_DATES} dates): {len(ids)}", pd.Series(party).value_counts().head(7).to_dict())
share_v = sv ** 2 / np.sum(sv ** 2)
say("variance by dimension:", np.round(share_v[:5], 3).tolist())

# 1. Chance: give every sentence to a random MP within the same data
null = []
for _ in range(20):
    shuffled = placed.copy()
    shuffled["member_id"] = rng.permutation(shuffled.member_id.to_numpy())
    cn = unit_counts(shuffled).reindex(ids, fill_value=0)
    _, svn = axes(mix(cn, base))
    null.append(svn[:5] ** 2 / np.sum(svn ** 2))
say("same, sentences shuffled between MPs:", np.round(np.mean(null, axis=0), 3).tolist())

# 2. Fit axes on half the debates; place MPs from the other half, which the axes never saw
debates = placed.debate.unique()
corrs = []
for rep in range(10):
    r = np.random.default_rng(100 + rep)
    in_a = set(r.permutation(debates)[:len(debates) // 2])
    fa, fb = placed[placed.debate.isin(in_a)], placed[~placed.debate.isin(in_a)]
    ca, cb = unit_counts(fa), unit_counts(fb)
    good = ca.index.intersection(cb.index)
    good = good[(ca.loc[good].sum(1) >= 8) & (cb.loc[good].sum(1) >= 8)]
    Va, Vb = mix(ca.loc[good], base), mix(cb.loc[good], base)
    Ta, _ = axes(mix(ca, base))
    corrs.append([abs(np.corrcoef(Va @ Ta[d], Vb @ Ta[d])[0, 1]) for d in range(3)] + [len(good)])
corrs = np.array(corrs)
say(f"held-out debates: axes from half, MPs from the other half; correlation by dimension {np.round(corrs[:, :3].mean(0), 2).tolist()} ({int(corrs[:, 3].mean())} MPs, 10 random splits)")

# 3. Remove any one big debate
by_size = placed.debate.value_counts().index[:15]
worst = [1.0, 1.0, 1.0]
worst_name = ["", "", ""]
for deb in by_size:
    rest = placed[placed.debate != deb]
    cr = unit_counts(rest).reindex(ids, fill_value=0)
    Vr = mix(cr, base)
    Tr, _ = axes(Vr)
    Zr = (Vr - Vr.mean(0)) @ Tr.T
    for d in range(3):
        c = abs(np.corrcoef(Z[:, d], Zr[:, d])[0, 1])
        if c < worst[d]:
            worst[d], worst_name[d] = c, deb
say("removing any one of the 15 biggest debates, lowest correlation with the full map:", np.round(worst, 2).tolist(), [w[:60] for w in worst_name])

# 4. What the data put at each end
say("\nWHAT EACH DIMENSION IS (all chosen by the data)")
for d in range(3):
    order = np.argsort(Vt[d])
    g = pd.Series(Z[:, d]).groupby(party)
    eta = float(((g.transform("mean") - Z[:, d].mean()) ** 2).sum() / ((Z[:, d] - Z[:, d].mean()) ** 2).sum())
    means = g.mean()[[x for x in ["Lab", "Con", "LD", "RUK", "SNP", "Green"] if x in set(party)]].round(3).to_dict()
    say(f"\n-- dim {d}: {share_v[d]:.1%} of variance, party explains {eta:.0%} of positions | party means {means}")
    for label, idx in (("low", order[:4]), ("high", order[::-1][:4])):
        say(f"   {label} end arguments:")
        for k in idx:
            say(f"     [{k}] {', '.join(arg_words[k][:8])}")
            say(f"          e.g. {arg_examples[k][0][:170]!r}")
    o = np.argsort(Z[:, d])
    say("   low MPs :", [f"{members.name[ids[i]]} ({party[i]})" for i in o[:9]])
    say("   high MPs:", [f"{members.name[ids[i]]} ({party[i]})" for i in o[-9:]])

# 5. Votes: speech before the vote only, axes refitted on that speech, direction chosen on training folds
votes = pd.read_parquet(ROOT / "data/clean/votes_long.parquet")
say("\nVOTES IT NEVER SAW (speech before the vote only; direction learned on other MPs; 0.5 is chance)")
for div, name in [(2406, "Asylum bill, July 2026"), (2074, "Welfare bill, July 2025"), (2071, "Assisted dying, June 2025"), (2078, "Palestine Action ban, July 2025")]:
    dv = votes[votes.division_id == div]
    date = dv.date.iloc[0]
    v = dv.drop_duplicates("member_id", keep=False).set_index("member_id").vote
    pre = s[s.date < date]
    e = pre.groupby("member_id").agg(sp=("speech_id", "nunique"), dt=("date", "nunique"))
    mps = e.index[(e.sp >= MIN_SPEECHES) & (e.dt >= MIN_DATES)]
    cp = unit_counts(pre[pre.member_id.isin(mps)]).reindex(mps, fill_value=0)
    bp = cp.sum(0).to_numpy(float) / cp.sum().sum()
    Vp = mix(cp, bp)
    Tp, _ = axes(Vp)
    Zp = (Vp - Vp.mean(0)) @ Tp.T
    for group in ["all", "Lab"]:
        pp = members.party_abbr.reindex(mps).to_numpy()
        m = np.array([i in v.index for i in mps]) & ((pp == group) if group != "all" else True)
        y = (v.reindex(mps[m]).to_numpy() == 1).astype(int)
        minority = min(y.sum(), (1 - y).sum()) if m.sum() else 0
        if minority < 5:
            say(f"  {name:32} {group:3} n={m.sum()} minority={minority}: too few")
            continue
        scores = []
        for rep in range(20):
            order_ = np.random.default_rng(rep).permutation(m.sum())
            out = np.zeros(m.sum())
            for fold in np.array_split(order_, 5):
                train = np.setdiff1d(order_, fold)
                w = np.array([np.corrcoef(Zp[m][train, d], y[train])[0, 1] for d in range(3)])
                out[fold] = Zp[m][fold] @ np.nan_to_num(w)
            scores.append(roc_auc_score(y, out))
        say(f"  {name:32} {group:3} n={m.sum()} minority={minority}: {np.mean(scores):.2f}")

# 6. Jev read the same subject separately. Agreement is a check, and it wasn't used to build anything.
j = pd.DataFrame(json.loads(l) for l in (ROOT / "data/quick/stance_asylum_jev.jsonl").read_text().splitlines()).rename(columns={"id": "passage_id"})
j = j.merge(a[["passage_id", "member_id"]], on="passage_id").dropna(subset=["stance"])
jm = j[j.member_id.isin(ids)].groupby("member_id").stance.mean()
idx = [i for i, m in enumerate(ids) if m in jm.index]
y = jm.reindex(ids[idx]).to_numpy()
say(f"\nAGREEMENT WITH JEV'S SEPARATE READING ({len(idx)} MPs)")
for d in range(3):
    lab = party[idx] == "Lab"
    say(f"  dim {d}: all {np.corrcoef(Z[idx, d], y)[0, 1]:+.2f} | within Labour {np.corrcoef(Z[idx, d][lab], y[lab])[0, 1]:+.2f} ({lab.sum()} MPs)")

np.savez(ROOT / "data/quick/asylum_blind.npz", ids=ids, Z=Z, Vt=Vt, base=base, arg_words=np.array(arg_words, dtype=object), V=V)
s[["speech_id", "member_id", "date", "debate", "arg", "fit", "text"]].to_parquet(ROOT / "data/quick/asylum_sentences.parquet", index=False)
(ROOT / "data/quick/asylum_blind_report.txt").write_text("\n".join(lines))
