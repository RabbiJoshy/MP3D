"""Compare speech-based fight positions with actual division votes. Free, local.
Usage: vote_check.py [fights.json] [--out-prefix results/vote_check]"""
import sys, json, re, argparse
from collections import defaultdict
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/joshuathomasamar/PycharmProjects/MP3D"
HERE = ROOT + "/prototypes/research_2026_11_10"
ap = argparse.ArgumentParser()
ap.add_argument("fights", nargs="?", default=HERE + "/results/fp_fights_final3.json")
ap.add_argument("--out-prefix", default=HERE + "/results/vote_check")
ap.add_argument("--jaccard", type=float, default=0.5)
ap.add_argument("--min-triples", type=int, default=8)
a = ap.parse_args()
SEED, NSPLIT = 20261110, 20

STRIP = set("""bill bills second third reading readings clause clauses amendment amendments committee
stage report new lords commons consideration motion disagree disagreement insist agree agreed
the of and a an to in for on by with at from as is be or opposition day cclm programme
schedule schedules lord ccl no nos""".split())
def toks(s):
    s = re.sub(r"\[.*?\]|\(.*?\)", " ", str(s).lower())
    return {t for t in re.findall(r"[a-z]+", s) if t not in STRIP and len(t) > 1}
def jac(x, y):
    return len(x & y) / len(x | y) if x and y else 0.0
def dtype(t):
    t = t.lower()
    if re.search(r"second reading|third reading", t): return "second/third reading"
    if re.search(r"amendment|clause|schedule|lords", t): return "amendment/clause"
    return "other"

fights = json.load(open(a.fights))["fights"]
P = pd.read_parquet(ROOT + "/data/clean/passages.parquet", columns=["passage_id", "date", "major"]).set_index("passage_id")
D = pd.read_parquet(ROOT + "/data/clean/divisions.parquet")
V = pd.read_parquet(ROOT + "/data/clean/votes_long.parquet", columns=["division_id", "member_id", "vote", "party"])
NDUP = int(V.duplicated(["division_id", "member_id"]).sum()); V = V.drop_duplicates(["division_id", "member_id"], keep="first")
D["date"] = pd.to_datetime(D["date"]); D["tok"] = D.title.map(toks)
divs_by_date = defaultdict(list)
for r in D.itertuples(): divs_by_date[r.date.strftime("%Y-%m-%d")].append(r)

# 1. linking
links, unlinked, fight_info = [], [], {}
for f in fights:
    mps = [dict(m, side=1) for m in f["mps_for"]] + [dict(m, side=-1) for m in f["mps_against"]]
    pids = {c["passage_id"] for m in mps for c in m["claims"]}
    pp = P.reindex(list(pids)).dropna()
    dm = defaultdict(set)
    for pid, r in pp.iterrows(): dm[str(r.date)[:10]].add(r.major)
    found = {}
    for d, majors in dm.items():
        for off in (0, 1):
            dd = (pd.Timestamp(d) + pd.Timedelta(days=off)).strftime("%Y-%m-%d")
            for dv in divs_by_date.get(dd, []):
                s = max(jac(dv.tok, toks(m)) for m in majors)
                if s >= a.jaccard:
                    found[dv.division_id] = max(s, found.get(dv.division_id, 0))
    fight_info[f["fight_id"]] = f
    if not found: unlinked.append({"fight_id": f["fight_id"], "label": f["label"], "majors": sorted({m for s in dm.values() for m in s})[:3]})
    for did, s in found.items():
        t = D.loc[D.division_id == did, "title"].iloc[0]
        links.append({"fight_id": f["fight_id"], "label": f["label"], "division_id": int(did), "title": t,
                      "date": str(D.loc[D.division_id == did, "date"].iloc[0])[:10], "score": round(s, 3), "type": dtype(t)})
L = pd.DataFrame(links)

# party majority per division
Vv = V.copy()
pm = Vv.groupby(["division_id", "party"]).vote.agg(n="size", aye=lambda x: (x == 1).sum()).reset_index()
pm["maj"] = np.where(pm.aye * 2 >= pm.n, 1, -1)
pm["minshare"] = np.minimum(pm.aye, pm.n - pm.aye) / pm.n
pmi = pm.set_index(["division_id", "party"])
vote_of = V.set_index(["division_id", "member_id"]).vote

# triples
rows = []
for l in links:
    f = fight_info[l["fight_id"]]
    for m in f["mps_for"] + f["mps_against"]:
        if m["stance"] == 0: continue
        key = (l["division_id"], m["member_id"])
        if key not in vote_of.index: continue
        v = int(vote_of[key])
        rows.append(dict(fight_id=l["fight_id"], label=l["label"], division_id=l["division_id"], dtype=l["type"],
                         member_id=m["member_id"], name=m["name"], spp=m["party"], speech=int(np.sign(m["stance"])),
                         stance=m["stance"], focus=m["focus"], vote=v,
                         passages=[c["passage_id"] for c in m["claims"]]))
T = pd.DataFrame(rows)
pv = V.drop_duplicates(["division_id", "member_id"]).set_index(["division_id", "member_id"]).party
T["vparty"] = [pv[(d, m)] for d, m in zip(T.division_id, T.member_id)]
T = T.join(pmi[["maj", "minshare", "n"]], on=["division_id", "vparty"]).rename(columns={"maj": "party_maj", "n": "party_n"})

# 2a orientation (per fight-division pair)
T["pair"] = T.fight_id.astype(str) + "_" + T.division_id.astype(str)
def best_sign(g):
    m = (g.speech * g.vote).mean() if len(g) else 0
    return 1 if m >= 0 else -1
orient_a = T.groupby("pair").apply(best_sign)
T["osign_a"] = T.pair.map(orient_a)
T["ok_a"] = (T.speech * T.vote * T.osign_a) > 0
# 2b held-out
rng = np.random.default_rng(SEED)
score_b = defaultdict(list)
for pair, g in T.groupby("pair"):
    if len(g) < 8: continue
    idx = g.index.values
    for _ in range(NSPLIT):
        perm = rng.permutation(len(idx)); h = len(idx) // 2
        A, B = idx[perm[:h]], idx[perm[h:]]
        ma = (T.loc[A, "speech"] * T.loc[A, "vote"]).mean()
        for i in B:
            c = T.at[i, "speech"] * T.at[i, "vote"]
            score_b[i].append(0.5 if ma == 0 else float((ma > 0) == (c > 0)))
T["ok_b"] = pd.Series({i: np.mean(s) for i, s in score_b.items()})

# 4 party-line context (orientation a)
T["vote_with_party"] = T.vote == T.party_maj
T["rebel"] = (~T.vote_with_party) | (T.minshare >= 0.2)
T["pure_rebel"] = ~T.vote_with_party
# majority speech side per party per fight (>=2 positioned MPs, not tied)
ss = T.drop_duplicates(["fight_id", "member_id"])
fm = {}
for (fid, p), g in ss.groupby(["fight_id", "spp"]):
    s = g.speech.sum()
    if len(g) >= 2 and s != 0: fm[(fid, p)] = int(np.sign(s))
T["party_speech_maj"] = [fm.get((f, p)) for f, p in zip(T.fight_id, T.spp)]

T["focus_g"] = np.where(T.focus >= 2, "2+ speeches", "1 speech")
T["stance_g"] = np.where(T.stance.abs() == 1, "exactly ±1", "mixed")
def pct(x): return None if len(x) == 0 else round(float(np.mean(x)) * 100, 1)
res = {"duplicate_vote_rows_dropped": NDUP, "fights_file": a.fights, "jaccard_threshold": a.jaccard, "n_fights": len(fights),
       "n_fights_linked": int(L.fight_id.nunique()) if len(L) else 0, "n_links": len(L), "unlinked_fights": unlinked,
       "links": links, "n_triples": len(T), "n_fight_division_pairs": int(T.pair.nunique())}
res["agreement_a_upper"] = {"pct": pct(T.ok_a), "n": len(T)}
tb = T.dropna(subset=["ok_b"])
res["agreement_b_heldout"] = {"pct": pct(tb.ok_b), "n": len(tb), "n_pairs": int(tb.pair.nunique())}
# a restricted to same triples as b, for fair comparison
res["agreement_a_on_b_triples"] = {"pct": pct(tb.ok_a), "n": len(tb)}
def grp(col, df=T, key="ok_a"):
    return {str(k): {"pct": pct(g[key]), "n": len(g)} for k, g in df.groupby(col)}
res["by_type_a"] = grp("dtype"); res["by_type_b"] = grp("dtype", tb, "ok_b")
res["by_focus_a"] = grp("focus_g"); res["by_focus_b"] = grp("focus_g", tb, "ok_b")
res["by_stance_a"] = grp("stance_g"); res["by_stance_b"] = grp("stance_g", tb, "ok_b")
pf = []
for fid, g in T.groupby("fight_id"):
    if len(g) < a.min_triples: continue
    gb = g.dropna(subset=["ok_b"])
    pf.append({"fight_id": int(fid), "label": g.label.iloc[0], "n": len(g), "agree_a": pct(g.ok_a),
               "agree_b": pct(gb.ok_b) if len(gb) else None, "n_b": len(gb)})
res["per_fight"] = sorted(pf, key=lambda x: -x["agree_a"])
res["party_line"] = {
    "vote_with_party_majority": {"pct": pct(T.vote_with_party), "n": len(T)},
    "speech_agrees_when_votes_with_party_a": {"pct": pct(T[T.vote_with_party].ok_a), "n": int(T.vote_with_party.sum())},
    "speech_agrees_rebel_or_free_a": {"pct": pct(T[T.rebel].ok_a), "n": int(T.rebel.sum())},
    "speech_agrees_pure_rebel_a": {"pct": pct(T[T.pure_rebel].ok_a), "n": int(T.pure_rebel.sum())},
}
tbr = tb[tb.rebel]
res["party_line"]["speech_agrees_rebel_or_free_b"] = {"pct": pct(tbr.ok_b), "n": len(tbr)}
res["party_line"]["speech_agrees_all_b_same_filter"] = {"pct": pct(tb.ok_b), "n": len(tb)}
c = T.dropna(subset=["party_speech_maj"]); c = c[c.speech != c.party_speech_maj]
res["party_line"]["speech_against_party_speech_majority__vote_also_against_party"] = {
    "pct": pct(~c.vote_with_party), "n": len(c)}
big = T[T.rebel].label.value_counts().index[0]
xr = T[T.rebel & (T.label != big)]
res["party_line"]["speech_agrees_rebel_or_free_a_excluding_largest_fight"] = {"fight_excluded": big, "pct": pct(xr.ok_a), "n": len(xr), "n_pure_rebel": int(xr.pure_rebel.sum()), "pct_pure_rebel": pct(xr[xr.pure_rebel].ok_a)}
res["party_line"]["rebel_triples_by_fight"] = T[T.rebel].label.value_counts().head(8).to_dict()
rl = T[T.rebel].copy(); rl["correct"] = rl.ok_a
rl = rl.sort_values("focus", ascending=False).head(15)
res["rebel_examples"] = [dict(name=r.name, party=r.spp, fight=r.label, speech="FOR" if r.speech > 0 else "AGAINST",
                              vote="Aye" if r.vote > 0 else "No", division=int(r.division_id), correct=bool(r.correct),
                              pure_rebel=bool(r.pure_rebel)) for r in rl.itertuples()]
dis = T[~T.ok_a].sort_values(["focus", "stance"], key=lambda s: s if s.name == "focus" else -s.abs(), ascending=False).head(15)
res["top_disagreements"] = [dict(name=r.name, party=r.spp, fight=r.label, focus=int(r.focus), stance=r.stance,
                                 speech="FOR" if r.speech > 0 else "AGAINST", vote="Aye" if r.vote > 0 else "No",
                                 division=int(r.division_id), title=D.loc[D.division_id == r.division_id, "title"].iloc[0],
                                 oriented_aye_means="FOR" if r.osign_a > 0 else "AGAINST", passages=r.passages) for r in dis.itertuples()]
json.dump(res, open(a.out_prefix + ".json", "w"), indent=1, default=str)
T.drop(columns=["passages"]).to_csv(a.out_prefix + "_triples.csv", index=False)

# chart
plt.rcParams["font.family"] = "sans-serif"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=100, gridspec_kw={"width_ratios": [1.7, 1]})
fig.patch.set_facecolor("white")
pfs = sorted(res["per_fight"], key=lambda x: x["agree_a"])
if pfs:
    lab = [(x["label"][:45] + ("…" if len(x["label"]) > 45 else "")) + f"  (n={x['n']})" for x in pfs]
    ax1.barh(range(len(pfs)), [x["agree_a"] for x in pfs], color="#3b6fb6")
    ax1.set_yticks(range(len(pfs))); ax1.set_yticklabels(lab, fontsize=8)
ov = res["agreement_a_upper"]["pct"]
ax1.axvline(ov, color="#c0392b", ls="--"); ax1.text(ov, len(pfs) - 0.4, f" overall {ov}%", color="#c0392b", fontsize=9, va="bottom")
ax1.set_xlim(0, 105); ax1.set_xlabel("Speech position matches vote (%), data-driven orientation")
ax1.set_title("Agreement per fight (fights with 8+ MPs)", loc="left", fontsize=11)
bars = [("All positions", res["agreement_a_upper"]), ("Votes with\nthe party", res["party_line"]["speech_agrees_when_votes_with_party_a"]),
        ("Rebel / free-\nvote votes", res["party_line"]["speech_agrees_rebel_or_free_a"])]
ax2.bar(range(3), [b[1]["pct"] or 0 for b in bars], color=["#3b6fb6", "#7f8c8d", "#e67e22"])
for i, b in enumerate(bars): ax2.text(i, (b[1]["pct"] or 0) + 1.5, f"{b[1]['pct']}%\nn={b[1]['n']}", ha="center", fontsize=9)
ax2.set_xticks(range(3)); ax2.set_xticklabels([b[0] for b in bars], fontsize=9); ax2.set_ylim(0, 115)
ax2.set_ylabel("Speech matches vote (%)"); ax2.axhline(50, color="#aaa", lw=0.8, ls=":")
ax2.set_title("Speech vs vote, by party-line context", loc="left", fontsize=11)
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(a.out_prefix + ".png", facecolor="white")
print(json.dumps({k: res[k] for k in ["n_fights", "n_fights_linked", "n_links", "n_triples", "n_fight_division_pairs", "agreement_a_upper", "agreement_b_heldout", "agreement_a_on_b_triples", "by_type_a", "by_type_b", "by_focus_a", "by_stance_a", "party_line"]}, indent=1))
print("per-fight", [(x["label"][:30], x["n"], x["agree_a"], x["agree_b"]) for x in res["per_fight"]])
print("unlinked", len(unlinked))
