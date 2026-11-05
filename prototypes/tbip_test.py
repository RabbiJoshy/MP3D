"""Text-based ideal points on about 15 subjects: subject and stance learned together, with no labels.

Each document is one MP in one debate. Its word counts are modelled as
    words ~ Poisson( sum over topics of  share of topic x topic's neutral words x exp(MP's position x topic's stance words) )
So one position per MP has to explain how that MP's language shifts across every topic. Votes and party never enter.

This is a simplified fit (the most likely values, with priors) of the model in Vafa, Naidu and Blei (2020), not their full
variational inference, so it gives no uncertainty for each MP yet.

Choices still mine: number of subjects and topics, vocabulary size, training length.
Out: data/quick/tbip.npz, tbip_report.txt
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

import torch
from sklearn.metrics import roc_auc_score
torch.set_num_threads(2)
torch.manual_seed(SEED)
N_SUBJECTS, K, VOCAB, STEPS, BATCH = 15, 15, 5000, 2500, 256

df["thing"] = thing
coh = np.array([np.mean(L[thing == k] @ normalize(L[thing == k].mean(0, keepdims=True))[0]) for k in range(K_THINGS)])
mps_per = df.groupby("thing").member.nunique()
candidates = [k for k in mps_per.sort_values(ascending=False).index if coh[k] >= np.median(coh)][:N_SUBJECTS]
overall = np.asarray(X.mean(0)).ravel()
for k in candidates:
    mean = np.asarray(X[thing == k].mean(0)).ravel()
    say(f"  subject: {mps_per[k]} MPs, {(thing == k).sum()} passages | {', '.join(terms[np.argsort(-(mean * np.log1p(mean / (overall + 1e-6))))[:6]])}")
sel = np.isin(thing, candidates)
d = df[sel].reset_index(drop=True)
Cs = C[sel]
total = np.asarray(Cs.sum(0)).ravel()
top = np.argsort(-total)[:VOCAB]
Cs, words = Cs[:, top], terms[top]
speech_n = d.groupby("member").speech.nunique()
mps = speech_n.index[speech_n >= 5].to_numpy()
pos = {m: i for i, m in enumerate(mps)}
party = members.party_abbr.reindex(mps).fillna("?").to_numpy()
keep_row = d.member.isin(pos).to_numpy()
d, Cs = d[keep_row].reset_index(drop=True), Cs[keep_row]
doc_key, doc_names = pd.factorize(d.member.astype(str) + "|" + d.debate)
G = sp.csr_matrix((np.ones(len(d)), (doc_key, np.arange(len(d)))), shape=(len(doc_names), len(d)))
Y = (G @ Cs).tocsr().astype("float32")
info = d.groupby(doc_key).agg(member=("member", "first"), date=("date", "first"), debate=("debate", "first"), thing=("thing", "first"))
doc_author = info.member.map(pos).to_numpy()
doc_date = info.date.to_numpy()
doc_debate = info.debate.to_numpy()
say(f"\n{Y.shape[0]} documents (one MP in one debate), {Y.shape[1]} words, {len(mps)} MPs", pd.Series(party).value_counts().head(6).to_dict())


def fit(rows, authors, n_authors, steps=STEPS, label=""):
    """Fit on the given document rows, returning each author's position and the topics."""
    t = time.time()
    Ys = Y[rows]
    a = torch.tensor(authors[rows], dtype=torch.long)
    n_docs = Ys.shape[0]
    th = torch.nn.Parameter(torch.randn(n_docs, K) * 0.1 - 1)
    be = torch.nn.Parameter(torch.randn(K, Y.shape[1]) * 0.1 - 3)
    et = torch.nn.Parameter(torch.randn(K, Y.shape[1]) * 0.01)
    xs = torch.nn.Parameter(torch.randn(n_authors) * 0.01)
    opt = torch.optim.Adam([th, be, et, xs], lr=0.02)
    sp_ = torch.nn.functional.softplus
    r = np.random.default_rng(SEED)
    warm = steps // 4
    for step in range(steps):
        idx = r.integers(0, n_docs, BATCH)
        y = torch.tensor(Ys[idx].toarray())
        theta, beta = sp_(th[idx]) + 1e-6, sp_(be) + 1e-6
        ideal = xs[a[idx]][:, None, None] if step >= warm else torch.zeros(BATCH, 1, 1)
        rate = (theta[:, :, None] * beta[None] * torch.exp(ideal * et[None])).sum(1) + 1e-9
        scale = n_docs / BATCH
        like = scale * (y * torch.log(rate) - rate).sum()
        prior = scale * ((0.3 - 1) * torch.log(theta) - 0.3 * theta).sum() + ((0.3 - 1) * torch.log(beta) - 0.3 * beta).sum() - 0.5 * (et ** 2).sum() - 0.5 * (xs ** 2).sum()
        loss = -(like + prior) / n_docs
        opt.zero_grad()
        loss.backward()
        opt.step()
    say(f"   fitted {label}: {n_docs} docs, {time.time() - t:.0f}s")
    return xs.detach().numpy(), sp_(be).detach().numpy(), et.detach().numpy()


all_rows = np.arange(Y.shape[0])
seen = np.bincount(doc_author, minlength=len(mps))
x, beta, eta = fit(all_rows, doc_author, len(mps), label="all MPs")
say(f"position spread {x.std():.2f}")

say("\nTOPICS AND WHAT THE POSITION MEANS INSIDE EACH (from the data)")
neutral_rank = np.argsort(-beta, axis=1)
for k in np.argsort(-beta.sum(1)):
    w_neg = beta[k] * np.exp(-1.5 * eta[k])
    w_pos = beta[k] * np.exp(1.5 * eta[k])
    say(f" neutral : {', '.join(words[neutral_rank[k][:7]])}")
    say(f"    minus: {', '.join(words[np.argsort(-w_neg)[:7]])}")
    say(f"    plus : {', '.join(words[np.argsort(-w_pos)[:7]])}")

order = np.argsort(x)
names = lambda idx: [f"{members.name[mps[i]]} ({party[i]})" for i in idx]
say("\nminus end:", names(order[:14]))
say("plus end :", names(order[-14:]))
g = pd.Series(x).groupby(party)
eta2 = float(((g.transform("mean") - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum())
say(f"party explains {eta2:.0%} of positions; party means", g.mean()[[p for p in ["Lab", "Con", "LD", "RUK", "SNP", "Green"] if p in set(party)]].round(2).to_dict())

# Is this real? Same MPs fitted on separate debates, compared with shuffled speakers.
r = np.random.default_rng(SEED)
deb = np.unique(doc_debate)
in_a = set(r.permutation(deb)[:len(deb) // 2])
ra = np.array([i for i in all_rows if doc_debate[i] in in_a])
rb = np.array([i for i in all_rows if doc_debate[i] not in in_a])
xa, _, _ = fit(ra, doc_author, len(mps), label="half A of debates")
xb, _, _ = fit(rb, doc_author, len(mps), label="half B of debates")
sa, sb = np.bincount(doc_author[ra], minlength=len(mps)), np.bincount(doc_author[rb], minlength=len(mps))
both = (sa >= 3) & (sb >= 3)
say(f"separate debates, same MPs: correlation {abs(np.corrcoef(xa[both], xb[both])[0, 1]):.2f} ({both.sum()} MPs)")
lab_both = both & (party == "Lab")
say(f"  within Labour: {abs(np.corrcoef(xa[lab_both], xb[lab_both])[0, 1]):.2f} ({lab_both.sum()} MPs)")
shuffled_author = r.permutation(doc_author)
xsa, _, _ = fit(ra, shuffled_author, len(mps), label="half A, shuffled speakers")
xsb, _, _ = fit(rb, shuffled_author, len(mps), label="half B, shuffled speakers")
sa2, sb2 = np.bincount(shuffled_author[ra], minlength=len(mps)), np.bincount(shuffled_author[rb], minlength=len(mps))
both2 = (sa2 >= 3) & (sb2 >= 3)
say(f"chance, shuffled speakers: correlation {abs(np.corrcoef(xsa[both2], xsb[both2])[0, 1]):.2f} ({both2.sum()} MPs)")

# Inside Labour alone, government vs opposition can't explain anything
lab_i = np.where(party == "Lab")[0]
lab_pos = {m: j for j, m in enumerate(lab_i)}
lab_rows = np.array([i for i in all_rows if doc_author[i] in lab_pos])
lab_author = np.full(len(doc_author), -1)
lab_author[lab_rows] = [lab_pos[doc_author[i]] for i in lab_rows]
xl, bl, el = fit(lab_rows, lab_author, len(lab_i), label="Labour only")
o = np.argsort(xl)
say("\nINSIDE LABOUR\n minus end:", [members.name[mps[lab_i[i]]] for i in o[:14]])
say(" plus end :", [members.name[mps[lab_i[i]]] for i in o[-14:]])
for k in np.argsort(-bl.sum(1))[:8]:
    say(f"   {', '.join(words[np.argsort(-bl[k])[:4]])} | minus: {', '.join(words[np.argsort(-(bl[k] * np.exp(-1.5 * el[k])))[:6]])} | plus: {', '.join(words[np.argsort(-(bl[k] * np.exp(1.5 * el[k])))[:6]])}")

votes = pd.read_parquet(ROOT / "data/clean/votes_long.parquet")
say("\nVOTES IT NEVER SAW, within Labour (0.5 is chance)")
def vote_auc(pos_vec, ids, div):
    v = votes[votes.division_id == div].drop_duplicates("member_id", keep=False).set_index("member_id").vote
    m = np.array([i in v.index for i in ids])
    y = (v.reindex(ids[m]).to_numpy() == 1).astype(int)
    return (np.nan, m.sum(), 0) if min(y.sum(), (1 - y).sum()) < 4 else (roc_auc_score(y, pos_vec[m]), m.sum(), min(y.sum(), (1 - y).sum()))
for div, nm in [(2406, "Asylum bill"), (2074, "Welfare bill"), (2078, "Palestine Action ban"), (2071, "Assisted dying"), (2058, "Abortion")]:
    a_, n_, mn = vote_auc(xl, mps[lab_i], div)
    say(f"  all speech, including after the vote: {nm:22} n={n_} minority={mn}: raw {a_:.2f}  (direction not chosen: read 0.5 as chance, either side as signal)")
for div, nm in [(2406, "Asylum bill"), (2074, "Welfare bill")]:
    date = votes[votes.division_id == div].date.iloc[0]
    pre = np.array([i for i in lab_rows if doc_date[i] < date])
    xp, _, _ = fit(pre, lab_author, len(lab_i), label=f"Labour before {nm}")
    counts = np.bincount(lab_author[pre], minlength=len(lab_i))
    enough = counts >= 3
    a_, n_, mn = vote_auc(xp[enough], mps[lab_i][enough], div)
    # Keep the direction of the full Labour fit, so it can't be flipped to whichever scores better
    sign = np.sign(np.corrcoef(xp[enough], xl[enough])[0, 1])
    a2, _, _ = vote_auc(xp[enough] * sign, mps[lab_i][enough], div)
    say(f"  speech before the vote only: {nm:22} n={n_} minority={mn}: {a2:.2f}")

import json as _json
jev = ROOT / "data/quick/stance_asylum_jev.jsonl"
if jev.exists():
    j = pd.DataFrame(_json.loads(l) for l in jev.read_text().splitlines()).dropna(subset=["stance"])
    p_all = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "member_id"])
    j = j.merge(p_all, left_on="id", right_on="passage_id")
    c = j.member_id.value_counts()
    jm = j[j.member_id.isin(c.index[c >= 3])].groupby("member_id").stance.mean()
    ix = [i for i, m in enumerate(mps) if m in jm.index]
    y = jm.reindex(mps[ix]).to_numpy()
    lab = party[ix] == "Lab"
    say(f"\nAGREEMENT WITH JEV'S ASYLUM READING: all {np.corrcoef(x[ix], y)[0, 1]:+.2f} ({len(ix)} MPs) | within Labour {np.corrcoef(x[ix][lab], y[lab])[0, 1]:+.2f} ({lab.sum()} MPs)")
il = [i for i, m in enumerate(mps[lab_i]) if m in jm.index]
yl = jm.reindex(mps[lab_i][il]).to_numpy()
say(f"LABOUR-ONLY POSITION vs JEV: {np.corrcoef(xl[il], yl)[0, 1]:+.2f} ({len(il)} MPs)")
np.savez(ROOT / "data/quick/tbip.npz", x=x, mps=mps, xl=xl, lab_ids=mps[lab_i], beta=beta, eta=eta, words=words)
(ROOT / "data/quick/tbip_report.txt").write_text("\n".join(lines))
say(f"\nDONE {time.time() - t0:.0f}s")
