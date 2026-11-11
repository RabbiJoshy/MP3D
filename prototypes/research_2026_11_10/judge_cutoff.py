"""Does a different cut-off on saved judge scores recover wrongly dropped claims? Free, local."""
import json, pathlib, collections
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
HERE = ROOT / "prototypes/research_2026_11_10"; R = HERE / "results"
S = pd.read_json(ROOT / "data/quick/research_2026_11_10/edu_jev2_sides.jsonl", lines=True)
S["label_n"] = S.label.str.strip()

def find(pid, mp, side, label):
    c = S[(S.passage_id == pid) & (S.mp == mp)]
    if len(c) > 1:
        c2 = c[c.stance == side]
        if len(c2): c = c2
    if len(c) > 1:
        c2 = c[c.label_n == (label or "").strip()]
        if len(c2): c = c2
    return None if c.empty else c.iloc[0]

hand = []  # dicts: source, pid, mp, side, good
h1 = json.load(open(R / "edu_handcheck.json"))
for g in h1["groups"]:
    for c in g["claims"]:
        hand.append(dict(source="handcheck1", pid=c["passage_id"], mp=c["mp"], side=c["label_side"], label=g["label"],
                         good=bool(c["same_measure"]) and c["side_correct"] == "yes"))
blind = {b["row_id"]: b for b in json.load(open(R / "edu_handcheck2_blind.json"))}
key = json.load(open(R / "edu_handcheck2_key.json"))
rows = json.load(open(R / "edu_handcheck2.json"))["rows"]
strat = {"A": "stratum A", "B": "stratum B", "H": "Hinds"}
for r in rows:
    k = key[str(r["row_id"])]; b = blind[r["row_id"]]
    hand.append(dict(source=strat[k["stratum"]], pid=b["passage_id"], mp=b["mp"], side=k["extractor_side"], label=b["group_label"],
                     good=r["position_on_label"] in ("FOR", "AGAINST") and r["position_on_label"] == k["extractor_side"]))
n_raw = len(hand)
seen = {}; dups = 0
for h in hand:
    kk = (h["pid"], h["mp"], h["side"])
    if kk in seen:
        dups += 1
        if seen[kk]["good"] != h["good"]: seen[kk]["good"] = False  # conflict: treat as BAD (reported below)
        continue
    seen[kk] = h
hand = list(seen.values())
miss = 0; H = []
for h in hand:
    c = find(h["pid"], h["mp"], h["side"], h["label"])
    if c is None: miss += 1; continue
    s = c.p_for if h["side"] == "FOR" else c.p_against; o = c.p_against if h["side"] == "FOR" else c.p_for
    H.append(dict(h, s=float(s), o=float(o), g=int(c.g), stance_sjoin=c.stance))
H = pd.DataFrame(H)
H["stance_match"] = H.side == H.stance_sjoin
print("raw", n_raw, "dups removed", dups, "unmatched", miss, "n", len(H), "stance mismatch", (~H.stance_match).sum())

# all-claim side scores
S["s"] = np.where(S.stance == "FOR", S.p_for, S.p_against); S["o"] = np.where(S.stance == "FOR", S.p_against, S.p_for)
def rule_t(df, t): return (df.s >= t) & (df.s > df.o)
def rule_m(df, m): return (df.s - df.o >= m)
def hand_counts(f):
    k = H[f(H)]; out = {"good": int(k.good.sum()), "bad": int((~k.good).sum())}
    for src, d in H.groupby("source"):
        kk = d[f(d)]; out[src] = {"good": int(kk.good.sum()), "bad": int((~kk.good).sum()), "n_good": int(d.good.sum()), "n_bad": int((~d.good).sum())}
    return out
def groups(f):
    a = S[f(S)]; contested = 0; contested_ex = 0; per = {}
    for g, d in S.groupby("g"):
        k = a[a.g == g]; F = set(k[k.stance == "FOR"].member_id); A = set(k[k.stance == "AGAINST"].member_id)
        both = F & A; Fx, Ax = F - both, A - both
        c = len(F) >= 2 and len(A) >= 2; cx = len(Fx) >= 2 and len(Ax) >= 2
        contested += c; contested_ex += cx
        per[int(g)] = dict(label=d.label.iloc[0], mps_for=len(F), mps_against=len(A), both=len(both), contested=bool(c), contested_ex=bool(cx))
    return contested, contested_ex, per, int(len(a))
def table(name, grid, rule):
    T = []
    for v in grid:
        f = lambda df, v=v: rule(df, v)
        hc = hand_counts(f); c, cx, per, n = groups(f)
        T.append(dict(param=round(float(v), 2), hand=hc, n_claims_kept=n, contested=c, contested_ex_both=cx, per_group=per))
    return T
tg = np.round(np.arange(0.05, 0.951, 0.05), 2); mg = np.round(np.arange(0, 0.901, 0.1), 2)
Tt = table("t", tg, rule_t); Tm = table("m", mg, rule_m)
cur = dict(hand=hand_counts(lambda d: d.s.ge(0.5) & d.s.gt(d.o)), contested=groups(lambda d: d.s.ge(.5) & d.s.gt(d.o))[:2])
# score summary for wrongly-dropped GOOD claims (current rule)
dropped = H[H.good & ~(H.s.ge(.5) & H.s.gt(H.o))]
zero_bad = [r["param"] for r in Tt if r["hand"]["bad"] == 0]
zero_bad_m = [r["param"] for r in Tm if r["hand"]["bad"] == 0]
out = dict(n_raw=n_raw, dups_removed=dups, unmatched=miss, n=len(H), n_good=int(H.good.sum()), n_bad=int((~H.good).sum()),
           by_source={k: dict(n=int(len(d)), good=int(d.good.sum()), bad=int((~d.good).sum())) for k, d in H.groupby("source")},
           current_rule=cur, threshold_table=Tt, margin_table=Tm, t_with_zero_bad=zero_bad, m_with_zero_bad=zero_bad_m,
           good_dropped_by_current=dict(n=len(dropped), s_quantiles=[round(float(x), 3) for x in dropped.s.quantile([0, .25, .5, .75, 1])], s_list=sorted(round(float(x), 3) for x in dropped.s)),
           s_quantiles_all_364=[round(float(x), 3) for x in S.s.quantile([0, .25, .5, .75, 1])],
           frac_all_s_below_0_05=float((S.s < 0.05).mean()),
           hand_stance_mismatch=int((~H.stance_match).sum()))
json.dump(out, open(R / "edu_judge_cutoff.json", "w"), indent=1)

# chart
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 11})
BL, RD = "#2a78d6", "#e34948"
fig = plt.figure(figsize=(14, 7), facecolor="white")
gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1.5, 1], hspace=0.14, wspace=0.22, left=0.06, right=0.98, top=0.9, bottom=0.11)
ax = fig.add_subplot(gs[:, 0]); rng = np.random.default_rng(1)
for good, y, c, mk in [(True, 1, BL, "o"), (False, 0, RD, "x")]:
    d = H[H.good == good]
    ax.scatter(d.s, y + rng.uniform(-.17, .17, len(d)), c=c, marker=mk, s=38 if good else 42, alpha=.75, linewidths=1.6 if not good else 0)
ax.axvline(.5, ls="--", c="#444", lw=1.2); ax.text(.51, 1.42, "Current cut-off 0.5", fontsize=10, color="#444")
ax.set_yticks([0, 1]); ax.set_yticklabels([f"Bad ({int((~H.good).sum())})", f"Good ({int(H.good.sum())})"]); ax.set_ylim(-.5, 1.6); ax.set_xlim(-.02, 1.02)
ax.set_xlabel("Judge score for the extractor's side"); ax.set_title("Judge score for the extractor's side, 96 hand-read claims".replace("96", str(len(H))), loc="left", fontsize=12.5)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, 1], sharex=a1)
x = [r["param"] for r in Tt]
a1.step(x, [r["contested"] for r in Tt], where="mid", c="#222", lw=2, label="Contested groups")
a1.step(x, [r["contested_ex_both"] for r in Tt], where="mid", c="#888", lw=2, ls="--", label="Excluding MPs on both sides")
a1.axvline(.5, ls="--", c="#bbb", lw=1); a1.set_ylabel("Contested groups (of 20)"); a1.legend(frameon=False, fontsize=9.5, loc="upper right")
a1.set_title("Lowering the threshold: contested groups and good claims kept", loc="left", fontsize=12.5)
a2.step(x, [100 * r["hand"]["good"] / out["n_good"] for r in Tt], where="mid", c=BL, lw=2, label="Good kept")
a2.step(x, [100 * r["hand"]["bad"] / out["n_bad"] for r in Tt], where="mid", c=RD, lw=2, label="Bad kept")
a2.axvline(.5, ls="--", c="#bbb", lw=1); a2.set_ylim(-3, 103); a2.set_ylabel("Hand-read kept (%)"); a2.set_xlabel("Threshold t on the extractor-side score"); a2.legend(frameon=False, fontsize=9.5)
plt.setp(a1.get_xticklabels(), visible=False)
for a in (a1, a2):
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.savefig(R / "edu_judge_cutoff.png", dpi=100, facecolor="white")

def row(r): h = r["hand"]; return f"{r['param']:.2f} good {h['good']}/{out['n_good']} bad {h['bad']}/{out['n_bad']} contested {r['contested']} ex {r['contested_ex_both']} claims {r['n_claims_kept']}"
print("by source", out["by_source"]); print("current", cur)
for r in Tt: print("t", row(r))
for r in Tm: print("m", row(r))
print("zero-bad t", zero_bad, "m", zero_bad_m); print("dropped good", out["good_dropped_by_current"]); print("all s q", out["s_quantiles_all_364"], out["frac_all_s_below_0_05"])
