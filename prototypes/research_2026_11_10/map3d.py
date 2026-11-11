"""All topics -> one 3D space. Contested debates (>=2 MPs each side) across every topic; MP x debate matrix
(2*p_agree-1, missing=0); PCA on MPs in >= MIN_DEB debates. Axes named by top-loading debates. No party used."""
import json, re, numpy as np, pandas as pd
MIN_SIDE, MIN_DEB = 2, 2
c = pd.read_json("all_jev.jsonl", lines=True)
c = c[c.jev != "UNRELATED"].assign(s=lambda x: 2 * x.p_agree - 1)
c["key"] = c.topic + " | " + c.proposition
side = c.groupby("key").apply(lambda g: pd.Series({"a": g[g.jev == "AGREE"].member_id.nunique(), "d": g[g.jev == "DISAGREE"].member_id.nunique()}))
deb = side[(side.a >= MIN_SIDE) & (side.d >= MIN_SIDE)].index.tolist()
c = c[c.key.isin(deb)]
print(len(deb), "contested debates in", c.topic.nunique(), "topics")
M = c.groupby(["member_id", "key"]).s.mean().unstack()[deb]
keep = M.notna().sum(axis=1) >= MIN_DEB
X = M[keep].fillna(0).values; X = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(X, full_matrices=False); var = S**2 / (S**2).sum()
mem = c.drop_duplicates("member_id").set_index("member_id")[["mp", "party", "role"]]
mps = pd.DataFrame(U[:, :3] * S[:3], index=M.index[keep], columns=["x", "y", "z"]).join(mem)
mps["n"] = M[keep].notna().sum(axis=1).values
axes = []
for a in range(3):
    L = pd.Series(Vt[a], index=deb).sort_values()
    axes.append({"var": float(var[a]), "neg": [[k, round(float(L[k]), 2)] for k in L.index[:4]], "pos": [[k, round(float(L[k]), 2)] for k in L.index[-4:][::-1]]})
    print(f"\nAXIS {a+1} ({var[a]:.0%})"); [print(f"   {L[k]:+.2f} {k}") for k in list(L.index[:4]) + list(L.index[-4:])]
pm = mps.groupby("party")[["x", "y", "z"]].mean().round(2); cnt = mps.party.value_counts()
print("\nparty means (>=3 MPs):\n", pm[cnt.reindex(pm.index) >= 3].to_string()); print("MPs placed:", len(mps))
json.dump({"axes": axes, "n_debates": len(deb), "mps": mps.reset_index().round(3).to_dict("records")}, open("all_map3d.json", "w"), indent=1, default=str)
