"""Unsupervised stance axes from claims. Each claim 'Supports/Opposes <object>' -> sign * embedding(object).
MP vector = mean of their signed claim vectors; centre; PCA. Axis ends named by the objects whose (signed) embeddings
project furthest. No propositions, no Jev, no party. python signed_axes.py HousingFull [min_claims]"""
import sys, re, json, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
tag = sys.argv[1]; MINC = int(sys.argv[2]) if len(sys.argv) > 2 else 3
c = pd.read_json(f"{tag}_claims.jsonl", lines=True)
c = c[c.position.notna()].copy()
rx = re.compile(r"^\s*(?:the speaker\s+)?(supports?|supporting|backs|favou?rs|welcomes|opposes?|opposing|rejects|criticis\w+)\s+(.*)", re.I)
def parse(s):
    m = rx.match(s or "")
    if not m: return None, None
    v = m[1].lower()
    return (-1 if v.startswith(("oppos", "reject", "critic")) else 1), m[2].strip(" .")
c[["sign", "obj"]] = c.position.apply(lambda s: pd.Series(parse(s)))
c = c[c.sign.notna() & (c.obj.str.len() > 8)]
print(f"{len(c)} signed claims ({(c.sign < 0).mean():.0%} 'opposes')")
m = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True, device="mps")
E = m.encode(c.obj.tolist(), normalize_embeddings=True, batch_size=64)
E = E - E.mean(0)                      # remove what every claim in the topic shares
V = E * c.sign.values[:, None]
c["i"] = range(len(c))
cnt = c.member_id.value_counts(); mps = cnt[cnt >= MINC].index
Mv = np.vstack([V[c.i[c.member_id == k]].mean(0) for k in mps]); Mv = Mv - Mv.mean(0)
U, S, Vt = np.linalg.svd(Mv, full_matrices=False); var = S**2 / (S**2).sum()
info = c.drop_duplicates("member_id").set_index("member_id")[["mp", "party", "role"]]
out = pd.DataFrame(U[:, :3] * S[:3], index=mps, columns=["x", "y", "z"]).join(info)
axes = []
for a in range(3):
    proj = V @ Vt[a]                    # where each signed claim pushes along this axis
    lo, hi = c.assign(p=proj).sort_values("p"), c.assign(p=proj).sort_values("p", ascending=False)
    axes.append({"var": float(var[a]), "neg": lo.position.head(8).tolist(), "pos": hi.position.head(8).tolist()})
    print(f"\nAXIS {a+1} ({var[a]:.0%})\n  [-]", *lo.position.head(6), sep="\n      ")
    print("  [+]", *hi.position.head(6), sep="\n      ")
    print("  MPs -:", ", ".join(f"{r.mp} ({r.party})" for r in out.sort_values(['x','y','z'][a]).head(6).itertuples()))
    print("  MPs +:", ", ".join(f"{r.mp} ({r.party})" for r in out.sort_values(['x','y','z'][a], ascending=False).head(6).itertuples()))
g = out.groupby("party")[["x", "y", "z"]].mean().round(2); n = out.party.value_counts()
print("\nparty means (>=3):\n", g[n.reindex(g.index) >= 3].to_string(), "\nMPs placed:", len(out))
# compare with Jev-built 2D map
try:
    j = pd.DataFrame(json.load(open(f"{tag}_map2d.json"))["mps"]).set_index("member_id")
    k = out.join(j[["x", "y"]], rsuffix="_jev", how="inner")
    cor = np.corrcoef(np.c_[k[["x", "y", "z"]].values, k[["x_jev", "y_jev"]].values].T)[:3, 3:]
    print(f"\ncorrelation with Jev map on {len(k)} shared MPs (rows: unsupervised x,y,z; cols: Jev x,y):\n", np.round(cor, 2))
except FileNotFoundError: pass
json.dump({"axes": axes, "mps": out.reset_index().round(3).to_dict("records")}, open(f"{tag}_signed.json", "w"), indent=1, default=str)
