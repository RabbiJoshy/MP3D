"""Generic topic axis test. Same code for every topic; only the topic name changes.

    python axis_test.py "Social welfare"

1. Every passage gets one CAP home (bge match to topic descriptions; 'No policy' bucket; runner-up kept).
2. Inside the topic: MPs with >= MIN_PASSAGES passages; MP point = mean of their passage embeddings.
3. Subtract the topic mean; the axes are the main directions of difference between MPs (PCA).
4. Report: ends of each axis (real passages), focuses (k-means clusters) placed on the axis, MPs by party,
   and how much of the axis party explains. Output: axis_<topic>.json
"""
import json, pathlib, re, sys
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
HERE = pathlib.Path(__file__).parent
TOPIC = sys.argv[1]
MIN_PASSAGES, MIN_MPS, N_FOCUS, N_AXES = 5, 100, 8, 3

src = (HERE / "cap_test.py").read_text()
CAP = eval(re.search(r"CAP = (\{.*?\})\n\nbase", src, re.S).group(1))
CAP["5"] = ("Employment", "jobs, unemployment, wages, the minimum wage, workers' rights, trade unions, strikes, workplace pensions")
CAP["18"] = ("Foreign trade", "international trade deals, tariffs, exports and imports, trade with other countries")
CAP["0"] = ("No policy", "parliamentary procedure, thanking or congratulating colleagues, tributes and condolences, points of order, chairing the debate, attacks on another party's record")

# 1. homes
p = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "member_id", "date", "major", "minor", "speech_type", "role", "text", "words"])
E = np.load(ROOT / "data/embed/passages.npy", mmap_mode="r")
assert (np.linalg.norm(E[::997], axis=1) > 0).all(), "embeddings incomplete"
bge = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True, device="mps" if torch.backends.mps.is_available() else "cpu")
keys = list(CAP)
Q = bge.encode(["Represent this sentence for searching relevant passages: " + CAP[k][1] for k in keys], normalize_embeddings=True)
sims = np.vstack([E[i:i + 50000] @ Q.T for i in range(0, len(E), 50000)])
o = np.argsort(-sims, 1)
names = np.array([CAP[k][0] for k in keys])
p["home"], p["runner_up"] = names[o[:, 0]], names[o[:, 1]]
p["home_margin"] = sims[np.arange(len(p)), o[:, 0]] - sims[np.arange(len(p)), o[:, 1]]
(HERE / "homes_summary.json").write_text(p.home.value_counts().to_json())

# 2. topic passages and MP points
chair = p.speech_type.str.contains("Chair", na=False)
t = p[(p.home == TOPIC) & ~chair & (p.words >= 25)].copy()
cnt = t.member_id.value_counts()
mps = cnt[cnt >= MIN_PASSAGES].index
t = t[t.member_id.isin(mps)]
if len(mps) < MIN_MPS:
    sys.exit(f"{TOPIC}: only {len(mps)} MPs with >= {MIN_PASSAGES} passages; rule says move to next topic")
X = np.asarray(E[t.index.values])
mu = X.mean(0)
mp_ids = np.array(sorted(mps))
M = np.vstack([X[(t.member_id == m).values].mean(0) for m in mp_ids]) - mu

# 3. axes = main directions of difference between MPs
U, S, Vt = np.linalg.svd(M, full_matrices=False)
var = (S ** 2) / (S ** 2).sum()
mem = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
mp = pd.DataFrame({"member_id": mp_ids, "n": cnt[mp_ids].values}).join(mem[["name", "party", "party_abbr", "party_colour"]], on="member_id")
role = t.groupby("member_id").role.agg(lambda r: (r == "minister").mean())
mp["minister_share"] = role[mp_ids].values

def party_r2(score):
    g = pd.Series(score).groupby(mp.party.values)
    return float(1 - (g.transform(lambda s: s - s.mean()) ** 2).sum() / ((score - score.mean()) ** 2).sum())

def short(s, n=330):
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"

# 4. focuses
km = KMeans(N_FOCUS, n_init=4, random_state=0).fit(X - mu)
t["focus"] = km.labels_
cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_df=1.0, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
docs = t.groupby("focus").text.apply(" ".join)
C = cv.fit_transform(docs).toarray().astype(float)
tf = C / C.sum(1, keepdims=True)
idf = np.log(1 + C.sum() / C.sum(0))
terms = np.array(cv.get_feature_names_out())
focus_terms = {f: list(terms[np.argsort(-(tf * idf)[i])[:6]]) for i, f in enumerate(docs.index)}

axes = []
for a in range(N_AXES):
    v = Vt[a]
    mp_score = M @ v
    if np.corrcoef(mp_score, mp.minister_share)[0, 1] > 0:  # orient: government-ish end on the right, only for consistent reading
        v, mp_score = -v, -mp_score
    ps = (X - mu) @ v
    t["s"] = ps
    ends = {}
    for side, asc in (("left", True), ("right", False)):
        pool = t[t.words >= 60].sort_values("s", ascending=asc)
        picked, seen = [], set()
        for r in pool.itertuples():
            if r.member_id in seen:
                continue
            seen.add(r.member_id)
            mm = mem.loc[r.member_id] if r.member_id in mem.index else None
            picked.append({"mp": mm["name"] if mm is not None else "?", "party": mm["party_abbr"] if mm is not None else "?",
                           "date": r.date, "debate": r.major or r.minor, "text": short(r.text)})
            if len(picked) == 5:
                break
        ends[side] = picked
    foc = []
    for f, g in t.groupby("focus"):
        foc.append({"focus": int(f), "n": int(len(g)), "terms": focus_terms[f], "pos": float(g.s.mean() / np.abs(ps).std()),
                    "example": short(g.loc[(g.s - g.s.mean()).abs().idxmin(), "text"], 200)})
    z = mp_score / mp_score.std()
    axes.append({"axis": a + 1, "var_share": float(var[a]), "party_r2": party_r2(mp_score),
                 "corr_minister": float(np.corrcoef(mp_score, mp.minister_share)[0, 1]),
                 "ends": ends, "focuses": sorted(foc, key=lambda x: x["pos"]),
                 "mps": [{"name": r.name, "party": r.party_abbr, "colour": r.party_colour, "n": int(r.n), "z": round(float(zz), 3)}
                         for r, zz in zip(mp.itertuples(), z)],
                 "party_means": {k: round(float(v_), 2) for k, v_ in pd.Series(z).groupby(mp.party_abbr.values).mean().items()
                                 if (mp.party_abbr == k).sum() >= 3}})

out = {"topic": TOPIC, "passages": int(len(t)), "mps": int(len(mps)), "axes": axes}
(HERE / f"axis_{TOPIC.replace(' ', '_')}.json").write_text(json.dumps(out, indent=1))
print(TOPIC, len(t), "passages", len(mps), "MPs")
for a in axes:
    print(f"\nAXIS {a['axis']}: var {a['var_share']:.2f}  party R2 {a['party_r2']:.2f}  corr minister {a['corr_minister']:.2f}  parties {a['party_means']}")
    for side in ("left", "right"):
        print(f"  [{side}]")
        for e in a["ends"][side][:3]:
            print(f"    {e['mp']} ({e['party']}) {e['debate']}: {e['text'][:220]}")
    for f in a["focuses"]:
        print(f"   pos {f['pos']:+.2f} n={f['n']} {', '.join(f['terms'])}")
