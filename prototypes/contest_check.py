"""Check the contested targets: are MPs' positions stable, beyond party, and linked across targets?

An MP's position on a target is their share of support minus opposition across passages where they took a side.
Votes and party never entered the scoring; they're only used here to test it.
Out: data/quick/contest_report.txt, contest_positions.parquet
"""
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
MIN_PASSAGES_MP, MIN_MPS_PER_SIDE = 2, 6
lines = []
def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)

c = pd.read_parquet(ROOT / "data/quick/contest_candidates.parquet", columns=["topic", "target", "pid", "member", "date", "debate"])
a = pd.DataFrame(json.loads(l) for l in (ROOT / "data/quick/contest_answers.jsonl").read_text().splitlines() if l.strip())
a = a.drop_duplicates(["pid", "target"]).merge(c.drop_duplicates(["pid", "target"]), on=["pid", "target"])
a["side"] = np.where(a.support >= 0.5, 1, np.where(a.oppose >= 0.5, -1, 0))
members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
votes = pd.read_parquet(ROOT / "data/clean/votes_long.parquet")
deep_counts = a.groupby("target").size()
deep = deep_counts.index[deep_counts > 16]
a = a[a.target.isin(deep)]
say(f"{len(deep)} targets scored in depth, {len(a)} answers, {a.member.nunique()} MPs")

rng = np.random.default_rng(7)
debates = a.debate.unique()
half = dict(zip(debates, rng.random(len(debates)) < 0.5))
a["half"] = a.debate.map(half)
pos = a[a.side != 0]
rows, positions = [], {}
for t, g in pos.groupby("target"):
    per = g.groupby("member").side.agg(["mean", "size"])
    per = per[per["size"] >= MIN_PASSAGES_MP]
    sup, opp = (per["mean"] > 0).sum(), (per["mean"] < 0).sum()
    ha, hb = g[g.half].groupby("member").side.mean(), g[~g.half].groupby("member").side.mean()
    both = ha.index.intersection(hb.index)
    rel = np.corrcoef(ha[both], hb[both])[0, 1] if len(both) >= 12 and ha[both].std() > 0 and hb[both].std() > 0 else np.nan
    party = members.party_abbr.reindex(per.index).fillna("?")
    pm = per["mean"].groupby(party.to_numpy()).mean()
    lab = per["mean"][party.to_numpy() == "Lab"]
    rows.append({"target": t, "topic": int(g.topic.iloc[0]), "passages": int(a[a.target == t].shape[0]), "with_side": round(len(g) / max(1, (a.target == t).sum()), 2),
                 "mps": len(per), "support_mps": int(sup), "oppose_mps": int(opp), "contest": round(min(sup, opp) / max(1, sup + opp), 2), "halves": None if np.isnan(rel) else round(rel, 2), "halves_mps": len(both),
                 "Lab": round(pm.get("Lab", np.nan), 2), "Con": round(pm.get("Con", np.nan), 2), "LD": round(pm.get("LD", np.nan), 2), "RUK": round(pm.get("RUK", np.nan), 2),
                 "lab_split": round(float(min((lab > 0).mean(), (lab < 0).mean())), 2) if len(lab) >= 8 else None})
    positions[t] = per["mean"]
R = pd.DataFrame(rows)
real = R[(R.support_mps >= MIN_MPS_PER_SIDE) & (R.oppose_mps >= MIN_MPS_PER_SIDE)].sort_values("contest", ascending=False)
say(f"\nTARGETS WITH AT LEAST {MIN_MPS_PER_SIDE} MPs ON EACH SIDE: {len(real)} of {len(R)}")
say(real.to_string(index=False))
say("\nEVERYTHING ELSE (fewer than that on one side)")
say(", ".join(R[~R.target.isin(real.target)].sort_values("contest", ascending=False).target.tolist()))

P = pd.DataFrame(positions)[real.target.tolist()] if len(real) else pd.DataFrame()
P.to_parquet(ROOT / "data/quick/contest_positions.parquet")
if len(real) >= 2:
    say(f"\nMPs with a position on 2+ contested targets: {int((P.notna().sum(1) >= 2).sum())}; on 4+: {int((P.notna().sum(1) >= 4).sum())}")
    topic_of = dict(zip(R.target, R.topic))
    pairs = []
    cols = P.columns.tolist()
    for i, x in enumerate(cols):
        for y in cols[i + 1:]:
            ok = P[x].notna() & P[y].notna()
            if ok.sum() >= 12 and P.loc[ok, x].std() > 0 and P.loc[ok, y].std() > 0:
                r = np.corrcoef(P.loc[ok, x], P.loc[ok, y])[0, 1]
                lab_ok = ok & (members.party_abbr.reindex(P.index).to_numpy() == "Lab")
                rl = np.corrcoef(P.loc[lab_ok, x], P.loc[lab_ok, y])[0, 1] if lab_ok.sum() >= 10 and P.loc[lab_ok, x].std() > 0 and P.loc[lab_ok, y].std() > 0 else np.nan
                pairs.append((r, rl, int(ok.sum()), x, y, topic_of[x] == topic_of[y]))
    say(f"\nLINKS BETWEEN TARGETS THROUGH SHARED MPs: {len(pairs)} pairs with 12+ MPs in common, {sum(not p[5] for p in pairs)} across different topics")
    for r, rl, n, x, y, same in sorted(pairs, key=lambda p: -abs(p[0]))[:25]:
        say(f"  {r:+.2f} (within Labour {rl:+.2f}) n={n} {'same topic ' if same else ''}[{x}] with [{y}]")
    # Shuffled check: break the link between MPs by permuting each target's positions
    null = []
    for s in range(50):
        Q = P.apply(lambda col: pd.Series(np.random.default_rng(s).permutation(col.dropna().to_numpy()), index=col.dropna().index).reindex(P.index))
        for i, x in enumerate(cols):
            for y in cols[i + 1:]:
                ok = Q[x].notna() & Q[y].notna()
                if ok.sum() >= 12 and Q.loc[ok, x].std() > 0 and Q.loc[ok, y].std() > 0:
                    null.append(abs(np.corrcoef(Q.loc[ok, x], Q.loc[ok, y])[0, 1]))
    real_abs = np.array([abs(p[0]) for p in pairs if not p[5]])
    say(f"cross-topic links: median |r| {np.median(real_abs):.2f}, share above shuffled 95th percentile ({np.quantile(null, .95):.2f}): {(real_abs > np.quantile(null, .95)).mean():.0%} (5% expected by chance)")

say("\nVOTES THEY NEVER SAW, Labour MPs: rebels' average position vs other Labour MPs (target positions only)")
for div, nm in [(2406, "Asylum bill"), (2074, "Welfare bill"), (2078, "Palestine Action ban"), (2071, "Assisted dying")]:
    v = votes[votes.division_id == div].drop_duplicates("member_id", keep=False).set_index("member_id").vote
    best = []
    for t in P.columns:
        col = P[t].dropna()
        ids = [m for m in col.index if m in v.index and members.party_abbr.get(m) == "Lab"]
        y = (v.reindex(ids).to_numpy() == 1).astype(int)
        if len(ids) >= 15 and min(y.sum(), (1 - y).sum()) >= 4 and col[ids].std() > 0:
            best.append((roc_auc_score(y, col[ids]), len(ids), min(y.sum(), (1 - y).sum()), t))
    for auc, n, mn, t in sorted(best, key=lambda b: -abs(b[0] - 0.5))[:4]:
        say(f"  {nm:22} [{t}] n={n} minority={mn}: {auc:.2f}")
(ROOT / "data/quick/contest_report.txt").write_text("\n".join(lines))
