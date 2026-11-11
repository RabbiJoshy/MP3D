"""Test C: votes supervise, speeches carry. Generic for any topic.
For each vote axis: ridge on passage embeddings (topic passages only) -> MP's vote-axis score.
Cross-validated by MP. Receipts = passages with the most extreme predicted score (distinct MPs).
Also: within-Labour test (does speech predict Labour MPs' rebel position?)."""
import json, pathlib, re, sys
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D"); HERE = pathlib.Path(__file__).parent
TOPIC = sys.argv[1]; tag = TOPIC.replace(" ", "_")
homes = HERE / "homes.parquet"
p = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "member_id", "date", "major", "minor", "speech_type", "role", "text", "words"])
E = np.load(ROOT / "data/embed/passages.npy", mmap_mode="r")
if not homes.exists():
    src = (HERE / "axis_test.py").read_text()
    exec(re.search(r"(src = \(HERE.*?CAP\[\"0\"\] = .*?\n)", src, re.S).group(1))
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True, device="mps")
    keys = list(CAP); Q = bge.encode(["Represent this sentence for searching relevant passages: " + CAP[k][1] for k in keys], normalize_embeddings=True)
    sims = np.vstack([E[i:i + 50000] @ Q.T for i in range(0, len(E), 50000)])
    pd.DataFrame({"home": np.array([CAP[k][0] for k in keys])[sims.argmax(1)]}).to_parquet(homes)
p["home"] = pd.read_parquet(homes).home.values
va = pd.read_parquet(HERE / f"vote_axis_{tag}.parquet")
mem = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
t = p[(p.home == TOPIC) & ~p.speech_type.str.contains("Chair", na=False) & (p.words >= 40) & p.member_id.isin(va.index)].copy()
X = np.asarray(E[t.index.values]); g = t.member_id.values
print(f"{TOPIC}: {len(t)} passages from {t.member_id.nunique()} MPs with votes")

def short(s, n=300):
    s = re.sub(r"\s+", " ", s).strip(); return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"

def run(name, rows, target):
    y = target.reindex(t.member_id[rows]).values
    Xr, gr = X[rows], g[rows]
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xr, y, gr):
        m = RidgeCV(alphas=np.logspace(-1, 3, 9)).fit(Xr[tr], y[tr]); pred[te] = m.predict(Xr[te])
    tt = t[rows].assign(pred=pred, y=y)
    mp = tt.groupby("member_id").agg(pred=("pred", "mean"), y=("y", "first"), n=("pred", "size"))
    mp = mp[mp.n >= 3]
    r = np.corrcoef(mp.pred, mp.y)[0, 1]
    print(f"\n== {name}: held-out MP-level correlation {r:.2f} ({len(mp)} MPs)")
    res = {"name": name, "r": float(r), "n_mps": int(len(mp)), "ends": {}}
    for side, asc in (("low", True), ("high", False)):
        seen, out = set(), []
        for x in tt[tt.words >= 60].sort_values("pred", ascending=asc).itertuples():
            if x.member_id in seen: continue
            seen.add(x.member_id); mm = mem.loc[x.member_id]
            out.append({"mp": mm["name"], "party": mm["party_abbr"], "debate": x.major or x.minor, "date": x.date, "text": short(x.text)})
            if len(out) == 6: break
        res["ends"][side] = out
        print(f"  [{side}]")
        for o in out[:4]: print(f"    {o['mp']} ({o['party']}) {str(o['debate'])[:45]}: {o['text'][:230]}")
    return res

allrows = np.ones(len(t), bool)
results = [run("vote axis 2 (all MPs)", allrows, va["1"] if "1" in va else va[1]),
           run("vote axis 1 (all MPs)", allrows, va["0"] if "0" in va else va[0])]
lab = (va.party == "Labour")
labrows = t.member_id.isin(va.index[lab]).values
results.append(run("within Labour: vote axis 1 (loyal vs rebel)", labrows, (va["0"] if "0" in va else va[0])[lab]))
(HERE / f"probe_{tag}.json").write_text(json.dumps(results, indent=1))
