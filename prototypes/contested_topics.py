"""Which topics are argued over? Rated from Commons votes, without using them to place anyone.

For every division recorded in the chamber, the debate it ended is linked to the topics of that debate's speech, using
the same 110 topics as the Fault Lines tool. Each topic then gets:
  votes      divisions in debates where it makes up a good share of the speech
  closeness  how near the vote was to 50/50
  partisan   how far the parties' Aye shares differed
  rebellion  share of MPs voting against their own party's majority
Party is only used here to describe the votes, never to place MPs.
Out: data/quick/contested_topics.json, contested_topics_report.txt
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

MIN_SHARE = 0.25
df["thing"] = thing
overall = np.asarray(X.mean(0)).ravel()
names = {}
for k in range(K_THINGS):
    mean = np.asarray(X[thing == k].mean(0)).ravel()
    names[k] = ", ".join(terms[np.argsort(-(mean * np.log1p(mean / (overall + 1e-6))))[:3]])

# Divisions as Hansard records them, with the heading of the debate they closed
divs = []
for f in sorted((RAW / "hansard" / "debates").glob("*.xml")):
    root = ET.parse(f).getroot()
    major = minor = ""
    for el in root.iter():
        if el.tag in ("major-heading", "oral-heading"):
            major, minor = tidy(el), ""
        elif el.tag == "minor-heading":
            minor = tidy(el)
        elif el.tag == "division":
            divs.append({"date": el.get("divdate"), "number": int(el.get("divnumber") or 0), "debate": (el.get("divdate") or "") + "|" + (minor or major)})
divs = pd.DataFrame(divs).drop_duplicates(["date", "number"])
say(f"divisions recorded in chamber Hansard: {len(divs)}")

# Votes from the Parliament data, matched on date and division number
meta, rows = [], []
for f in (RAW / "divisions").glob("*.json"):
    d = json.loads(f.read_text())
    meta.append({"division_id": d["DivisionId"], "date": d["Date"][:10], "number": d["Number"], "ayes": d["AyeCount"], "noes": d["NoCount"], "title": d["Title"]})
meta = pd.DataFrame(meta)
votes = pd.read_parquet(ROOT / "data/clean/votes_long.parquet")
votes["p"] = votes.party.str.replace(r" \(Co-op\)", "", regex=True)
per = []
for div, g in votes.groupby("division_id"):
    share = g.groupby("p").vote.apply(lambda v: (v == 1).mean())
    size = g.groupby("p").size()
    big = share[size >= 5]
    partisan = float(np.average(np.abs(big - np.average(big, weights=size[big.index])), weights=size[big.index])) * 2 if len(big) >= 2 else np.nan
    maj = g.groupby("p").vote.transform(lambda v: np.sign(v.sum()) if v.sum() != 0 else 0)
    rebellion = float(((g.vote * maj) < 0).mean())
    per.append({"division_id": div, "partisan": partisan, "rebellion": rebellion})
meta = meta.merge(pd.DataFrame(per), on="division_id", how="left")
meta["closeness"] = 1 - (meta.ayes - meta.noes).abs() / (meta.ayes + meta.noes).clip(1)
m = divs.merge(meta, on=["date", "number"], how="inner")
say(f"matched to vote data: {len(m)} of {len(divs)}")

# Topic mix of each debate's speech
mix = df.groupby(["debate", "thing"]).size().unstack(fill_value=0)
mix = mix.div(mix.sum(1), axis=0)
m = m[m.debate.isin(mix.index)]
say(f"divisions whose debate has backbench or opposition speech in our data: {len(m)}")
passages_per_topic = pd.Series(thing).value_counts()
out = []
for k in range(K_THINGS):
    if k not in mix.columns:
        continue
    w = mix.reindex(m.debate)[k].fillna(0).to_numpy()
    use = w >= MIN_SHARE
    debates_with_topic = (mix[k] >= MIN_SHARE).sum()
    if use.sum() == 0:
        out.append({"t": k, "name": names[k], "divisions": 0, "debates": int(debates_with_topic), "per_100_debates": 0.0, "closeness": None, "partisan": None, "rebellion": None, "titles": []})
        continue
    mm, ww = m[use], w[use]
    out.append({"t": k, "name": names[k], "divisions": int(use.sum()), "debates": int(debates_with_topic),
                "per_100_debates": round(100 * mm.debate.nunique() / max(1, debates_with_topic), 1),
                "closeness": round(float(np.average(mm.closeness, weights=ww)), 2), "partisan": round(float(np.nanmean(mm.partisan)), 2),
                "rebellion": round(float(np.average(mm.rebellion, weights=ww)), 3), "titles": mm.sort_values("closeness", ascending=False).title.head(3).tolist()})
R = pd.DataFrame(out)
R["contest"] = (R.per_100_debates.rank(pct=True) + R.closeness.fillna(0).rank(pct=True) + R.partisan.fillna(0).rank(pct=True)) / 3
R.loc[R.divisions == 0, "contest"] = 0
R = R.sort_values("contest", ascending=False)
(ROOT / "data/quick/contested_topics.json").write_text(R.to_json(orient="records"))
pd.set_option("display.width", 250)
say(f"\nMOST CONTESTED (votes, closeness and party disagreement combined)")
for r in R.head(22).itertuples():
    say(f"  {r.name:42} divisions {r.divisions:3d} | {r.per_100_debates:5.1f} per 100 debates | close {r.closeness} | partisan {r.partisan} | rebels {r.rebellion:.1%} | e.g. {r.titles[0][:60] if r.titles else ''}")
say(f"\nNEVER VOTED ON ({(R.divisions == 0).sum()} topics): " + "; ".join(R[R.divisions == 0].name.head(25)))
say(f"\nMOST REBELLIOUS (share of MPs voting against their own party)")
for r in R[R.divisions >= 3].sort_values("rebellion", ascending=False).head(8).itertuples():
    say(f"  {r.name:42} rebels {r.rebellion:.1%} over {r.divisions} divisions | e.g. {r.titles[0][:60] if r.titles else ''}")

axes = json.loads((ROOT / "data/quick/topic_axes.json").read_text())["topics"]
A = pd.DataFrame([{"t": a["t"], "n_axes": a.get("n_axes", 0), "party_axis1": a["axes"][0]["party"] if a.get("axes") else None} for a in axes if not a.get("skip")])
J = R.merge(A, on="t", how="left")
from scipy.stats import spearmanr
say(f"\nAGAINST THE SPEECH AXES: contest score vs number of axes found, rank correlation {spearmanr(J.contest, J.n_axes, nan_policy='omit')[0]:+.2f}; vs how much party explains axis 1, {spearmanr(J.contest, J.party_axis1, nan_policy='omit')[0]:+.2f}")
(ROOT / "data/quick/contested_topics_report.txt").write_text("\n".join(lines))
