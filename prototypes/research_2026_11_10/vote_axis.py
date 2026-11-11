"""Test A: sides from Commons votes. Generic: every division gets a CAP home from its title (same descriptions as
passages); inside a topic, MP x division matrix (aye=+1, no=-1, absent=missing) -> axes; party share; rebels."""
import json, pathlib, re, sys
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D"); HERE = pathlib.Path(__file__).parent
TOPIC = sys.argv[1]
src = (HERE / "axis_test.py").read_text()
exec(re.search(r"(src = \(HERE.*?CAP\[\"0\"\] = .*?\n)", src, re.S).group(1))
d = pd.read_parquet(ROOT / "data/clean/divisions.parquet")
v = pd.read_parquet(ROOT / "data/clean/votes_long.parquet")
bge = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True, device="mps" if torch.backends.mps.is_available() else "cpu")
keys = list(CAP)
Q = bge.encode(["Represent this sentence for searching relevant passages: " + CAP[k][1] for k in keys], normalize_embeddings=True)
T = bge.encode(d.title.tolist(), normalize_embeddings=True)
d["home"] = [CAP[keys[i]][0] for i in (T @ Q.T).argmax(1)]
print(d.home.value_counts().to_string())
dd = d[d.home == TOPIC]
print(f"\n{TOPIC}: {len(dd)} divisions"); print(dd[["date", "title"]].to_string(max_colwidth=90))
vv = v[v.division_id.isin(dd.division_id)]
print("\nvote values:", vv.vote.value_counts().to_dict())
vv = vv.assign(x=vv.vote.astype(float))
M = vv.pivot_table(index="member_id", columns="division_id", values="x")
M = M[M.notna().sum(1) >= max(3, len(dd) // 3)]
party = vv.groupby("member_id").party.first().reindex(M.index)
# fill missing with column mean (standard quick PCA on roll calls), then centre
X = M.apply(lambda c: c.fillna(c.mean())).values; X = X - X.mean(0)
U, S, Vt = np.linalg.svd(X, full_matrices=False); var = S**2 / (S**2).sum()
out = {}
for a in range(2):
    s = pd.Series(U[:, a] * S[a], index=M.index)
    g = s.groupby(party.values)
    r2 = 1 - ((s - g.transform("mean"))**2).sum() / ((s - s.mean())**2).sum()
    within_lab = s[party == "Labour"].std() / s.std() if (party == "Labour").sum() > 5 else np.nan
    load = pd.Series(Vt[a], index=M.columns).sort_values()
    print(f"\nVOTE AXIS {a+1}: var {var[a]:.2f}  party R2 {r2:.2f}  Labour spread/overall {within_lab:.2f}")
    print("  party means:", g.mean().round(2).to_dict())
    tt = d.set_index("division_id").title
    for did in list(load.index[:3]) + list(load.index[-3:]):
        print(f"   {load[did]:+.2f} {tt[did][:100]}")
    out[a] = s
pd.DataFrame(out).assign(party=party).to_parquet(HERE / f"vote_axis_{TOPIC.replace(' ', '_')}.parquet")
print("\nrebels in topic divisions:", vv.groupby("member_id").rebel.sum().gt(0).sum(), "MPs rebelled at least once")
