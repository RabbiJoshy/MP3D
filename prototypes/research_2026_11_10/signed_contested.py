"""Unsupervised: issues = policy clusters (already grouped); sign from the claim's own Supports/Opposes;
keep issues where >=K MPs support AND >=K oppose; MP x issue mean sign; PCA. No propositions, no Jev, no party."""
import sys, re, json, numpy as np, pandas as pd
tag, K = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3
c = pd.read_json(f"{tag}_issues.jsonl", lines=True)
rx = re.compile(r"^\s*(?:the speaker\s+)?(supports?|supporting|backs|favou?rs|welcomes|opposes?|opposing|rejects|criticis\w+)\b", re.I)
c["sign"] = c.position.map(lambda s: (lambda m: None if not m else (-1 if m[1].lower().startswith(("oppos", "reject", "critic")) else 1))(rx.match(s or "")))
c = c[c.sign.notna()]
st = c.groupby("issue").apply(lambda g: pd.Series({"sup": g[g.sign > 0].member_id.nunique(), "opp": g[g.sign < 0].member_id.nunique(), "n": len(g)}))
deb = st[(st.sup >= K) & (st.opp >= K)].index
print(f"{len(deb)} contested issues of {len(st)} (>= {K} MPs on each sign)")
c = c[c.issue.isin(deb)]
M = c.groupby(["member_id", "issue"]).sign.mean().unstack()
keep = M.notna().sum(axis=1) >= 2
X = M[keep].fillna(0).values; X = X - X.mean(0)
U, S, Vt = np.linalg.svd(X, full_matrices=False); var = S**2 / (S**2).sum()
info = c.drop_duplicates("member_id").set_index("member_id")[["mp", "party"]]
out = pd.DataFrame(U[:, :2] * S[:2], index=M.index[keep], columns=["x", "y"]).join(info)
lab = c.groupby("issue").policy.agg(lambda s: s.value_counts().index[0])
for a in range(2):
    L = pd.Series(Vt[a], index=M.columns).sort_values()
    print(f"\nAXIS {a+1} ({var[a]:.0%})")
    for k in list(L.index[:3]) + list(L.index[-3:]): print(f"   {L[k]:+.2f} [{st.loc[k,'sup']}+/{st.loc[k,'opp']}-] {lab[k]}")
g = out.groupby("party")[["x", "y"]].mean().round(2); n = out.party.value_counts()
print("\nparty means:\n", g[n.reindex(g.index) >= 3].to_string(), "\nMPs placed:", len(out))
j = pd.DataFrame(json.load(open(f"{tag}_map2d.json"))["mps"]).set_index("member_id")
k = out.join(j[["x", "y"]], rsuffix="_jev", how="inner")
print(f"\ncorrelation with Jev map on {len(k)} shared MPs:", np.round(np.corrcoef(k.x, k.x_jev)[0, 1], 2), "(axis 1 vs Jev axis 1)",
      np.round(np.corrcoef(k.y, k.y_jev)[0, 1], 2), "(axis 2 vs Jev axis 2)")
cl = {"Lab": "a", "Con": "b", "LD": "c", "Green": "e", "RUK": "f"}
s = out.x / out.x.abs().quantile(.98)
print("DOTS" + json.dumps(" ".join(f"{cl.get(p,'h')}{max(-1,min(1,v)):.2f}" for p, v in zip(out.party, s))))
