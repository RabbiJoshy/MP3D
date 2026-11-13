"""Build the canonical Commons hierarchy app from cached local evidence.
Writes app_hierarchy.html, the sk13 compatibility copy, and docs/index.html / stance.html.
No upload or publication: deployment still requires a separate git push.
"""
import json, pathlib, collections, csv, re, sys
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score, silhouette_score

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "research_2026_11_10" / "results"
ROOT = HERE.parent
ev = json.loads((RES / "fp2_mp_evidence.json").read_text())
ts = json.loads((RES / "fp2_topic_space.json").read_text())
_fj = json.loads((RES / "fp2_fights.json").read_text()); FALL = _fj["fights"]; _tot = _fj["totals"]      # all 242 fights (82 two-sided, 42 one_short, 118 one_sided)
def _merge_fight_claims(f, es, party_of=None):
    acc = {}
    for k in ("mps_for", "mps_against", "mps_both_sides"):
        for m in f[k]: acc[m["member_id"]] = dict(m, claims=list(m["claims"]))
    added = 0
    for e in es:
        mid = int(e["member_id"])
        m = acc.setdefault(mid, {"member_id": mid, "name": e["mp"], "party": e.get("party") or (party_of(mid) if party_of else ""), "claims": []})
        if not any(c["passage_id"] == e["passage_id"] for c in m["claims"]):      # dedupe speech x fight
            m["claims"].append({"passage_id": e["passage_id"], "measure": f["label"], "side": e["verdict"]}); added += 1
    f["mps_for"], f["mps_against"], f["mps_both_sides"] = [], [], []
    for m in acc.values():
        net = sum(1 if c["side"] == "FOR" else -1 for c in m["claims"])
        m["stance"] = round(net / len(m["claims"]), 3); m["focus"] = len(m["claims"])
        f["mps_for" if net > 0 else "mps_against" if net < 0 else "mps_both_sides"].append(m)
    nf, na = len(f["mps_for"]), len(f["mps_against"])
    f["status"] = "fight" if nf >= 2 and na >= 2 else "one_short" if min(nf, na) == 1 and max(nf, na) >= 2 else "one_sided"
    return added

# ---- pilot positions: strict-judge FOR/AGAINST verdicts on claims matched to fp2 fights (pl_core.inject_factory rule), added to the dense verdicts
def inject_pilot(F):
    j = pd.read_json(ROOT / "data/quick/research_2026_11_10/pl/judge_out.jsonl", lines=True)
    j = j[j.verdict.isin(["FOR", "AGAINST"])]
    by = collections.defaultdict(list)
    for r in j.itertuples(): by[int(r.fight)].append({"member_id": r.member_id, "mp": r.mp, "party": r.party, "passage_id": r.passage_id, "verdict": r.verdict})
    added = sum(_merge_fight_claims(F[fi], es) for fi, es in by.items())
    print("pilot positions added:", added)
    return F

# ---- bridge positions: verified targeted-reading additions (research_2026_11_12)
def inject_bridge(F, members_df):
    bf = ROOT / "data/quick/research_2026_11_12/bridge/accepted.json"
    if not bf.exists(): return F
    by_label = {f["label"]: f for f in F}
    by = collections.defaultdict(list)
    for r in json.loads(bf.read_text()):
        if r.get("verdict") in ("FOR", "AGAINST") and r.get("label") in by_label:
            by[r["label"]].append(r)
    party_of = lambda mid: str(members_df.loc[mid, "party"]) if mid in members_df.index else ""
    added = sum(_merge_fight_claims(by_label[lbl], es, party_of) for lbl, es in by.items())
    print("bridge positions added:", added)
    return F

USE_PILOT = "--no-pilot" not in sys.argv
if USE_PILOT: FALL = inject_pilot(FALL)
EVF = {f["i"]: f for f in ev["fights"]}                                  # two-sided fights with per-MP claims + stance
PASS = ev["P"]
members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
for c in ("party", "party_abbr", "constituency", "name"):
    if c in members: members[c] = members[c].fillna("")
members["party_colour"] = members.party_colour.fillna("888888")
FALL = inject_bridge(FALL, members)

# ---------------- merge same-bill duplicate fights (generic rules, no hand edits) ----------------
topic_of = {int(k): v for k, v in ts["topic_of"].items()}
def stance_of(f):
    d = {}
    for k in ("mps_for", "mps_against", "mps_both_sides"):
        for x in f[k]: d[x["member_id"]] = x["stance"]
    return d
pairs = {}   # (a,b) -> rule
for r in csv.DictReader(open(RES / "mr_judged_pairs.csv")):
    if r["verdict"] == "SAME_SIDE_DUPLICATE" and float(r["similarity"]) >= 0.85:
        pairs[tuple(sorted((int(r["fightA"]), int(r["fightB"]))))] = "judged duplicate (mr_judged_pairs.csv, sim>=0.85)"
n_judged = len(pairs)
# Second generic rule for pairs of two-sided fights, which the judged list did not cover: same embedding text (sim>=0.85 on the
# local bge model, as in mr_cands.py) AND the MPs on both fights mostly take the same side (>=5 common MPs, >=80% same sign).
from sentence_transformers import SentenceTransformer
_bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu", local_files_only=True)
_txt = lambda f: f["label"] + ". " + (f.get("description") or "") + " FOR: " + (f.get("for_means") or "") + " AGAINST: " + (f.get("against_means") or "")
_E = _bge.encode([_txt(f) for f in FALL], normalize_embeddings=True)
two = [i for i, f in enumerate(FALL) if f["status"] == "fight"]
for a in two:
    for b in two:
        if a < b and float(_E[a] @ _E[b]) >= 0.85:
            A_, B_ = stance_of(FALL[a]), stance_of(FALL[b])
            com = [m for m in A_ if m in B_ and A_[m] != 0 and B_[m] != 0]
            if len(com) >= 5 and np.mean([np.sign(A_[m]) == np.sign(B_[m]) for m in com]) >= 0.8 and (a, b) not in pairs:
                pairs[(a, b)] = "auto: sim>=0.85 and same side on >=80% of >=5 common MPs"
n_auto = len(pairs) - n_judged
par = list(range(len(FALL)))
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for a, b in pairs: par[find(a)] = find(b)
comp = collections.defaultdict(list)
for i in range(len(FALL)): comp[find(i)].append(i)
n_members = lambda i: len(FALL[i]["mps_for"]) + len(FALL[i]["mps_against"]) + len(FALL[i]["mps_both_sides"])

MPNP = {x["member_id"]: (x["name"], x["party"]) for f in FALL for k in ("mps_for", "mps_against", "mps_both_sides") for x in f[k]}
def merged_mps(ids):
    """union of MPs over fights `ids`; claims deduped; stance = mean of +1/-1 over all claims, side by net sign (exactly the fp2_analyse rule)."""
    acc = {}
    for i in ids:
        for k in ("mps_for", "mps_against", "mps_both_sides"):
            for x in FALL[i][k]:
                d = acc.setdefault(x["member_id"], {})
                for c in x["claims"]: d[(c["passage_id"], c["side"])] = c      # (no reversed members exist: only same-side duplicates are merged)
    out = {"for": [], "against": [], "both": []}
    for mid, d in acc.items():
        cl = [{"p": c["passage_id"], "s": c["side"]} for c in d.values()]
        net = sum(1 if c["s"] == "FOR" else -1 for c in cl)
        out["for" if net > 0 else "against" if net < 0 else "both"].append({"id": int(mid), "claims": cl, "stance": round(net / len(cl), 3)})
    return out
def fstatus(nf, na):      # fp2_analyse.status
    return "fight" if nf >= 2 and na >= 2 else "one_short" if min(nf, na) == 1 and max(nf, na) >= 2 else "one_sided"

# merged fights in the fp2_fights.json format, then the SAME topic_space.build() that made fp2_topic_space.json is re-run on them
import sys
sys.path.insert(0, str(HERE / "research_2026_11_10"))
import topic_space as TS
RECS = []
for grp in sorted(comp.values(), key=min):
    tw = [i for i in grp if FALL[i]["status"] == "fight"]
    anchor = max(tw or grp, key=n_members)
    mm = merged_mps(grp)
    RECS.append({"anchor": anchor, "grp": grp, "mm": mm, "status": fstatus(len(mm["for"]), len(mm["against"]))})
MF = []
for r in RECS:
    a_ = FALL[r["anchor"]]; mm = r["mm"]
    pk = lambda lst: [{"member_id": x["id"], "name": MPNP[x["id"]][0], "party": MPNP[x["id"]][1], "stance": x["stance"], "focus": len(x["claims"]),
                       "claims": [{"passage_id": c["p"], "measure": a_["label"], "side": c["s"]} for c in x["claims"]]} for x in lst]
    MF.append({"fight_id": len(MF), "topic": a_["topic"], "label": a_["label"], "status": r["status"],
               "mps_for": pk(mm["for"]), "mps_against": pk(mm["against"]), "mps_both_sides": pk(mm["both"])})
ts_old = ts
ts = TS.build(MF, log=lambda *a: None); ts.pop("holdout", None)
topic_of = {int(k): v for k, v in ts["topic_of"].items()}
MAPF, POOL = [], []      # merged two-sided fights (drawn as lines) and merged one-sided fights ("took a side" pool)
for idx, r in enumerate(RECS):
    A = FALL[r["anchor"]]
    rec = {"label": A["label"], "subj": topic_of.get(idx), "fm": A.get("for_means") or "", "am": A.get("against_means") or "", "desc": A.get("description") or "", "orig": sorted(r["grp"]), "n_merged": len(r["grp"]), **r["mm"]}
    (MAPF if r["status"] == "fight" else POOL).append(rec)
n_before = len(FALL); n_after = len(MAPF) + len(POOL)
print(f"fights {n_before} -> {n_after}; judged pairs {n_judged}, auto pairs {n_auto}; two-sided fights {len(two)} -> {len(MAPF)}")
FIGHTS = MAPF

# Inject scored cross-party causes (speech-only, no whipped division votes)
_causes_f = HERE / "research_2026_11_12/causes_scored.json"
if _causes_f.exists():
    _cs = json.loads(_causes_f.read_text()).get("causes", [])
    _existing_labels = {f["label"].lower() for f in FIGHTS}
    _added_c = 0
    for c in _cs:
        if c["label"].lower() in _existing_labels: continue
        FIGHTS.append({"label": c["label"], "subj": c["subj"], "fm": c["fm"], "am": c["am"], "desc": c["label"],
                       "orig": [], "n_merged": 1, "for": c["for"], "against": c["against"], "both": c["both"]})
        _added_c += 1
    print(f"added {_added_c} cross-party speech causes from causes_scored.json -> {len(FIGHTS)} total fights")

# ---------------- direct argument-level 3D space (unified MP & visitor placement) ----------------
SUBJ = sorted(t for t, s in ts["scales"].items() if s["scale"])
NT = len(SUBJ)
scores = {t: {int(m): v for m, v in ts["scales"][t]["mps"].items()} for t in SUBJ}
ids = sorted({m for t in SUBJ for m in scores[t] if m in members.index})
mi = {m: i for i, m in enumerate(ids)}
N = len(ids)
S = np.full((N, NT), np.nan)
for k, t in enumerate(SUBJ):
    for m, v in scores[t].items():
        if m in mi: S[mi[m], k] = v

fight_y = {}; fight_w = {}; fight_claims = collections.defaultdict(list)
for fi, f in enumerate(FIGHTS):
    y = np.full(N, np.nan)
    w = np.ones(N, dtype=float)
    for side in ("for", "against", "both"):
        for m in f[side]:
            if m["id"] in mi:
                i_m = mi[m["id"]]
                y[i_m] = float(m["stance"])
                w[i_m] = min(2.0, float(np.sqrt(max(1, len(m["claims"])))))
                fight_claims[i_m].append([fi, side, [[c["p"], c["s"]] for c in m["claims"][:3]]])
    fight_y[fi] = y; fight_w[fi] = w

Y_mat = np.column_stack([fight_y[fi] for fi in range(len(FIGHTS))]) if FIGHTS else np.zeros((N, 0))
W_mat = np.column_stack([fight_w[fi] for fi in range(len(FIGHTS))]) if FIGHTS else np.ones((N, 0))
fight_vote_title = {}
obs_Y = np.isfinite(Y_mat)
LAM_DIRECT = 2.5

def _project_direct(y_row, w_row, V_mat, b_vec, lam):
    ix = np.isfinite(y_row)
    if not ix.any(): return np.zeros(3)
    Vx = V_mat[ix]; Vw = Vx * w_row[ix][:, None]
    return np.linalg.solve(Vw.T @ Vx + lam * np.eye(3), Vw.T @ (y_row[ix] - b_vec[ix]))

def _fit_direct(Y, W, rank=3, lam=LAM_DIRECT, seed=17, iters=70):
    M, T = Y.shape
    mask = np.isfinite(Y)
    b = np.nan_to_num(np.nanmean(Y, axis=0))
    rng = np.random.default_rng(seed)
    V = rng.normal(0, 0.2, (T, rank))
    U = np.zeros((M, rank))
    eye = np.eye(rank)
    for _ in range(iters):
        for i in range(M): U[i] = _project_direct(Y[i], W[i], V, b, lam)
        for j in range(T):
            ix = mask[:, j]
            if ix.any():
                A = U[ix]; Aw = A * W[ix, j][:, None]
                V[j] = np.linalg.solve(Aw.T @ A + lam * eye, Aw.T @ (Y[ix, j] - b[j]))
    U = np.array([_project_direct(Y[i], W[i], V, b, lam) for i in range(M)])
    return U, V, b

U_dir, V_dir, b_dir = _fit_direct(Y_mat, W_mat, rank=3, lam=LAM_DIRECT, seed=17, iters=70)
_, _, _rot = np.linalg.svd(U_dir, full_matrices=False)
Q_rot = _rot.T
U_dir = U_dir @ Q_rot
V_dir = V_dir @ Q_rot
# Orthogonal Varimax rotation on subject loadings so the 3 axes align with distinct policy clusters
_on0 = obs_Y.any(1)
_L = np.zeros((NT, 3))
for k in range(NT):
    _ok = _on0 & np.isfinite(S[:, k])
    if _ok.sum() >= 12 and np.std(S[_ok, k]) > 1e-8:
        for ax in range(3):
            if np.std(U_dir[_ok, ax]) > 1e-8: _L[k, ax] = float(np.corrcoef(U_dir[_ok, ax], S[_ok, k])[0, 1])
_Rv, _d = np.eye(3), 0.0
for _ in range(50):
    _d_old, _Lam = _d, _L @ _Rv
    _u, _s, _vt = np.linalg.svd(_L.T @ (_Lam**3 - (1.0 / max(1, NT)) * _Lam @ np.diag(np.diag(_Lam.T @ _Lam))))
    _Rv, _d = _u @ _vt, float(np.sum(_s))
    if _d_old != 0 and _d / _d_old < 1 + 1e-7: break
U_dir = U_dir @ _Rv
V_dir = V_dir @ _Rv
for ax in range(3):
    if U_dir[:, ax].sum() < 0:
        U_dir[:, ax] *= -1; V_dir[:, ax] *= -1
# Put the two cleanest ideological axes (1 & 3) on the default front X/Y plane
_perm = [0, 2, 1]
U_dir = U_dir[:, _perm]
V_dir = V_dir[:, _perm]
_scale = float(np.percentile(np.linalg.norm(U_dir[obs_Y.any(1)], axis=1), 98)) or 1.0
Z = U_dir / _scale
vv = V_dir * _scale
ridge_direct = float(LAM_DIRECT * (_scale ** 2))
on_map = obs_Y.sum(1) >= 1
replay_err = float(np.max(np.abs(np.array([_project_direct(Y_mat[i], W_mat[i], vv, b_dir, ridge_direct) for i in range(N)]) - Z)))
assert replay_err < 1e-8, f"direct replay mismatch: {replay_err}"
print(f"{N} MPs ({int(on_map.sum())} placed directly from {len(FIGHTS)} arguments, {int((~on_map).sum())} unplaced; replay err {replay_err:.2e})")

def fit_diameter(Z3, y, min_n=12):
    obs = ~np.isnan(y)
    if obs.sum() < min_n or np.std(y[obs]) < 1e-8: return None, 0.0, int(obs.sum())
    Zo = Z3[obs] - Z3[obs].mean(0); yo = (y[obs] - y[obs].mean()) / y[obs].std()
    beta, *_ = np.linalg.lstsq(Zo, yo, rcond=None)
    nrm = float(np.linalg.norm(beta))
    if nrm < 1e-8: return None, 0.0, int(obs.sum())
    pred = Zo @ beta
    r = float(np.corrcoef(pred, yo)[0, 1]) if pred.std() > 1e-8 else 0.0
    return beta / nrm, abs(r), int(obs.sum())

# ---------------- fights (directions directly from shared 3D model V) ----------------
fights_out = []
for fi, f in enumerate(FIGHTS):
    y = Y_mat[:, fi]
    v = vv[fi]
    vnorm = float(np.linalg.norm(v))
    n_obs = int(np.isfinite(y).sum())
    u = (v / vnorm) if vnorm > 1e-8 else None
    r = 0.0
    if u is not None and n_obs >= 5 and np.std(y[np.isfinite(y)]) > 1e-8:
        pr = Z[np.isfinite(y)] @ u
        if pr.std() > 1e-8: r = abs(float(np.corrcoef(pr, y[np.isfinite(y)])[0, 1]))
    sj = f["subj"]
    small = n_obs < 12
    fights_out.append({"i": fi, "r_in": round(r, 3), "r_ho": round(r, 3), "label": f["label"], "subj": sj, "n": n_obs,
                       "nfit": n_obs, "u": None if u is None else [round(float(x), 4) for x in u], "r": round(r, 3),
                       "sc": [None if np.isnan(val) else round(float(val), 2) for val in y],
                       "nfor": int((y > 0).sum()), "nag": int((y < 0).sum()), "merged": f["n_merged"], "small": small, "fm": f["fm"], "am": f["am"], "desc": f["desc"],
                       **({"vote_title": fight_vote_title[fi]} if fi in fight_vote_title else {})})
FB = {f["i"]: f for f in fights_out}

# ---------------- subjects (planes spanned directly by their arguments' vectors vv[ix]) ----------------
subj_out = []
for k, t in enumerate(SUBJ):
    fl = [f for f in fights_out if f["subj"] == t]
    ix = [f["i"] for f in fl if f["u"]]
    u, r, nfit = fit_diameter(Z, np.where(on_map, S[:, k], np.nan))
    plane = None; capture = None
    if len(ix) >= 2:
        M_s = vv[ix]
        _, sv, Vt = np.linalg.svd(M_s, full_matrices=False)
        e1, e2 = Vt[0].copy(), Vt[1].copy()
        if u is not None and float(e1 @ u) < 0: e1 = -e1
        if u is None:
            u = e1 / (np.linalg.norm(e1) or 1.0)
            ok_s = ~np.isnan(S[:, k]) & on_map
            if ok_s.sum() >= 5 and np.std(S[ok_s, k]) > 1e-8:
                r = abs(float(np.corrcoef(Z[ok_s] @ u, S[ok_s, k])[0, 1]))
        capture = float((sv[0] ** 2 + sv[1] ** 2) / (sv ** 2).sum()) if (sv ** 2).sum() > 0 else 1.0
        plane = [[round(float(x), 4) for x in e1], [round(float(x), 4) for x in e2]]
    elif len(ix) == 1 and u is None:
        u = np.array(fl[0]["u"]); r = fl[0]["r"]
    L = sorted(ts["scales"][t]["loadings"], key=lambda l: -abs(l["v"]))
    top = L[0]
    hi = ("Supports: " if top["v"] > 0 else "Opposes: ") + top["label"]
    lo = ("Opposes: " if top["v"] > 0 else "Supports: ") + top["label"]
    subj_out.append({"k": k, "r_in": round(r, 3), "name": t, "n": int((~np.isnan(S[:, k])).sum()), "nfit": nfit, "u": None if u is None else [round(float(x), 4) for x in u],
                     "r": round(r, 3), "plane": plane, "capture": None if capture is None else round(capture, 3),
                     "nlines": len(ix), "nfights": len(fl), "hi": hi, "lo": lo,
                     "sc": [None if np.isnan(v) else round(float(v), 3) for v in S[:, k]]})

# ---------------- semantic super-topics (planes spanned directly by their arguments' vectors vv[ix]) ----------------
SUPER_TOPICS = [
    ("Economy & Work", ["Macroeconomics", "Domestic commerce", "Employment", "Foreign trade"]),
    ("Public Services & Welfare", ["Education", "Health", "Social welfare", "Housing", "Culture"]),
    ("Environment, Energy & Infrastructure", ["Environment", "Energy", "Agriculture", "Public lands", "Transport"]),
    ("Rights, Justice & Borders", ["Civil rights", "Law and crime", "Immigration", "Conscience (free votes)"]),
    ("State, Defence & Constitution", ["Government operations", "Defence"]),
]
subj_idx_map = {t: k for k, t in enumerate(SUBJ)}
groups = []
for g_label, g_names in SUPER_TOPICS:
    ks = [subj_idx_map[nm] for nm in g_names if nm in subj_idx_map]
    if not ks: continue
    names = [SUBJ[k] for k in ks]
    z = np.where(np.isnan(S[:, ks]), np.nan, (S[:, ks] - np.nanmean(S[:, ks], 0)) / np.nanstd(S[:, ks], 0))
    zs = np.array([np.nan if np.all(np.isnan(row)) else np.nanmean(row) for row in z])
    ix_f = [f["i"] for f in fights_out if f["subj"] in names and f["u"]]
    sv_ = [(k, np.array(subj_out[k]["u"]) * subj_out[k]["r"]) for k in ks if subj_out[k]["u"]]
    gplane = gcap = gcap_all = gax = gload = None
    gr = float(np.mean([subj_out[k]["r"] for k, _ in sv_])) if sv_ else 0.0
    if len(ix_f) >= 2:
        Mg = vv[ix_f]
        _, sg, Vg = np.linalg.svd(Mg, full_matrices=False)
        if len(sg) > 1 and sg[1] > 1e-9:
            e1, e2 = Vg[0].copy(), Vg[1].copy()
            if sv_ and sum(float(v @ e1) for _, v in sv_) < 0: e1 = -e1
            if e2[np.argmax(np.abs(e2))] < 0: e2 = -e2
            Ms = np.array([v for _, v in sv_]) if sv_ else Mg
            tot_s = float((Ms ** 2).sum())
            gcap = float((((Ms @ e1) ** 2).sum() + ((Ms @ e2) ** 2).sum()) / tot_s) if tot_s > 0 else None
            gcap_all = float((sg[0] ** 2 + sg[1] ** 2) / (sg ** 2).sum())
            gplane = [[round(float(x), 4) for x in e1], [round(float(x), 4) for x in e2]]
            gload = [{"k": k, "a": round(float(v @ e1), 3), "b": round(float(v @ e2), 3)} for k, v in sv_]
            def axlab(key):
                top = sorted(gload, key=lambda d: -abs(d[key]))[:2]
                return " / ".join(("+" if d[key] > 0 else "-") + SUBJ[d["k"]] for d in top)
            gax = [axlab("a"), axlab("b")] if gload else None
    groups.append({"label": g_label, "plane": gplane, "capture": None if gcap is None else round(gcap, 3), "capture_all": None if gcap_all is None else round(gcap_all, 3), "r": round(gr, 3),
                   "ax": gax, "load": gload, "name": g_label, "subjects": ks,
                   "n": int((~np.isnan(zs)).sum()), "sc": [None if np.isnan(v) else round(float(v), 2) for v in zs]})
stab_val = 1.0
groups_ok = True
print("semantic super-topics:", [(g["label"], len(g["subjects"]), g["capture_all"]) for g in groups])
os_out = []; mp_one = collections.defaultdict(list)
for pi, f in enumerate(POOL):
    nf = len(f["for"]); na = len(f["against"])
    # inferred direction: centroid of all placed MPs -> mean position of the backers on the majority side (>=3 placed backers)
    u_os = None; r_os = 0.0; side_os = None; bk = []
    for sd, lst in (("for", f["for"]), ("against", f["against"])):
        ix = [mi[m["id"]] for m in lst if m["id"] in mi]
        if len(ix) >= 3 and (side_os is None or len(ix) > len(bk)): side_os, bk = sd, ix
    if side_os:
        off = Z[bk] - Z.mean(0); mean = off.mean(0); nm = float(np.linalg.norm(mean))
        se = float(np.sqrt(off.var(0, ddof=1).sum() / len(bk)))      # standard error of the mean offset (all axes)
        if nm > 1e-8 and se > 1e-8:
            snr = nm / se; r_os = snr / (snr + 2.0); u_os = (mean / nm) * (1 if side_os == "for" else -1)   # filled end = FOR end
    os_out.append({"label": f["label"], "subj": f["subj"], "nfor": nf, "nag": na, "merged": f["n_merged"], "side": side_os, "nb": len(bk), "bk": bk,
                   "u": None if u_os is None else [round(float(x), 4) for x in u_os], "r": round(r_os, 3), "fm": f["fm"], "am": f["am"], "desc": f["desc"]})
    for side in ("for", "against"):
        for m in f[side]:
            if m["id"] in mi: mp_one[mi[m["id"]]].append([pi, side, [[c["p"], c["s"]] for c in m["claims"][:3]]])

# ---------------- MPs ----------------
mp_out = []
for m in ids:
    r = members.loc[m]
    mp_out.append({"id": int(m), "n": r["name"] or str(m), "p": r.party_abbr or "Ind", "c": r.constituency, "col": "#" + str(r.party_colour)})
used_p = {p for v in list(fight_claims.values()) + list(mp_one.values()) for fc in v for p, _ in fc[2]}

# passages: reuse fp2's excerpts where present, otherwise cut an excerpt from data/clean/passages.parquet (same method as fp2_build_space_page.py)
def url(pid):
    m = re.match(r"uk\.org\.publicwhip/(debate|westminhall|standing)/([^#]+)", pid)
    if not m: return None
    sec = {"debate": "debates", "westminhall": "whall", "standing": "pbc"}[m.group(1)]
    return f"https://www.theyworkforyou.com/{sec}/?id={m.group(2)}"
W = lambda s: set(re.findall(r"[a-z]{4,}", s.lower()))
def excerpt(text, measure, n=380):
    t = " ".join(text.split())
    if len(t) <= n: return t
    sents = re.split(r"(?<=[.!?])\s+", t); mw = W(measure); best, bi = -1, 0
    for i, s in enumerate(sents):
        sc = len(W(s) & mw)
        if sc > best: best, bi = sc, i
    out = ""
    for s in sents[bi:]:
        if len(out) + len(s) > n and out: break
        out += (" " if out else "") + s
    out = out[:n].rsplit(" ", 1)[0] if len(out) > n else out
    return ("… " if bi > 0 else "") + out + ("…" if len(out) < len(t) else "")
label_of_p = {}
for f in FIGHTS + POOL:
    for side in ("for", "against", "both"):
        for m in f.get(side, []):
            for c in m["claims"]: label_of_p.setdefault(c["p"], f["label"])
missing = [p for p in used_p if p not in PASS]
pas = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "date", "major", "minor", "text"])
pas = pas[pas.passage_id.isin(missing)].drop_duplicates("passage_id").set_index("passage_id")
Ps = {}
for p in used_p:
    if p in PASS: Ps[p] = {"d": PASS[p]["date"], "h": PASS[p]["debate"], "t": PASS[p]["ex"], "u": PASS[p]["url"]}
    elif p in pas.index:
        r = pas.loc[p]
        Ps[p] = {"d": str(r["date"])[:10], "h": " — ".join(x for x in [r["major"], r["minor"]] if isinstance(x, str) and x), "t": excerpt(r["text"] or "", label_of_p.get(p, "")), "u": url(p)}
print("passages", len(Ps), "of", len(used_p), "used;", len(missing), "read from parquet")

# ---------------- sk13: asks (checked measures an MP argued for/against in one speech) ----------------
import importlib.util, html as _html
import xml.etree.ElementTree as ET
RAW = ROOT / "data/raw/hansard"
_spec = importlib.util.spec_from_file_location("clean02", ROOT / "pipeline/02_clean.py"); C02 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C02)
place_re = C02.build_place_regex(members.reset_index())
_xml_cache = {}
def _speeches(f):
    if f not in _xml_cache: _xml_cache[f] = {e.get("id"): e for e in ET.parse(f).getroot().iter("speech")}
    return _xml_cache[f]
def file_for(pid):      # sk10_build_check.file_for
    m = re.match(r"uk\.org\.publicwhip/(debate|westminhall|standing)/(.+)\.[^.]+\.[^.]+#\d+$", pid) or re.match(r"uk\.org\.publicwhip/(debate|westminhall|standing)/(.+?)\.\d+\.\d+#\d+$", pid)
    if not m: return None
    fs = list((RAW / {"debate": "debates", "westminhall": "westminhall", "standing": "standing"}[m.group(1)]).glob(f"*{m.group(2)}.xml"))
    return fs[0] if len(fs) == 1 else None
def clean_sent(sent):
    for pat in C02.ADDRESS: sent = pat.sub(" ", sent)
    sent, _ = place_re.subn(" ", sent)
    return re.sub(r"\s+([,.;:?!])", r"\1", re.sub(r"\s+", " ", sent)).strip(" ,;:")
def recover(pid, stored):      # sk10_build_check.recover: original text (with place names) or None
    f = file_for(pid)
    if f is None: return None
    el = _speeches(f).get(pid.split("#")[0])
    if el is None: return None
    paras = [" ".join("".join(p.itertext()).split()) for p in el.findall("p") if p.get("class") != "italic" and p.get("pwmotiontext") != "yes"]
    pairs = []
    for para in paras:
        para = re.sub(r"\[[^\]]{0,120}\]", " ", para); para = re.sub(r"\bhon\.\s", "hon ", para)
        for s_ in C02.SENTENCE.split(para):
            if C02.COURTESY.search(s_): continue
            c = clean_sent(s_)
            if len(c.split()) >= 3: pairs.append((s_.strip(), c))
    rest = stored; used = []; i = 0
    while i < len(pairs) and not rest.startswith(pairs[i][1]): i += 1
    if i == len(pairs): return None
    while rest and i < len(pairs):
        o, c = pairs[i]
        if rest.startswith(c): used.append(o); rest = rest[len(c):].lstrip()
        i += 1
    if rest: return None
    return re.sub(r"\bhon (?=Friend|Member|Gentleman|Lady)", "hon. ", " ".join(used))
_ABBR = r"(?:Mr|Mrs|Ms|Dr|St|No|Nos|hon|Hon|rt|Rt|Gen|Col|Sir|Prof|Rev|Lt|Sgt|vs|etc|e\.g|i\.e|cf|approx|Co|Ltd|Jr|Sr|Mt|Cllr|Hr|Cmd|para|paras|Vol)"
def split_sents(t):      # sk8_build_mock.split_sents
    t = " ".join(str(t).split()); out, start = [], 0
    for mm in re.finditer(r'[.!?]["”\')]*\s+(?=["“\'(\[]?[A-Z0-9])', t):
        end = mm.end(); last = re.findall(r"(\S+)$", t[start:mm.start()]); w = last[0] if last else ""
        if t[mm.start()] == "." and (re.fullmatch(_ABBR, w.strip('"“\'([')) or re.fullmatch(r"[A-Z]", w)): continue
        out.append(t[start:mm.start() + len(mm.group(0).rstrip())]); start = end
    if start < len(t): out.append(t[start:])
    return [x.strip() for x in out if x.strip()]
def _cut(t, n):
    return t if len(t) <= n else t[:n - 1].rsplit(" ", 1)[0].rstrip(",;:- ") + "…"
_enc = {}
def _encode(texts):
    new = [t for t in dict.fromkeys(texts) if t not in _enc]
    if new:
        V = _bge.encode(new, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
        for t, v in zip(new, V): _enc[t] = v
def best_excerpt(measure, text, n=320):      # sk8_build_mock.best_excerpt (cosine on local bge-base)
    t = " ".join(str(text).split()); ss = split_sents(t)
    if len(ss) <= 1: return _cut(t, n)
    i = int(np.argmax([float(_enc[x] @ _enc[measure]) for x in ss])); lo = i; cur = ss[i]
    if len(cur) < 100:
        if i + 1 < len(ss): cur = cur + " " + ss[i + 1]
        elif i > 0: cur = ss[i - 1] + " " + cur; lo = i - 1
    r = _cut(cur, n)
    return ("… " + r) if lo > 0 else r
ASKS_RAW = json.loads((RES / "sk12_asks_kept.json").read_text())
_ids_set = set(mi)
asks_in = [a for a in ASKS_RAW if int(a["member_id"]) in _ids_set and a["asks"]]
asks_po = sorted([a for a in ASKS_RAW if int(a["member_id"]) not in _ids_set and a["asks"]], key=lambda a: (str(a.get("name") or ""), int(a["member_id"])))     # sk14: profile-only MPs (no map dot)
po_missing = [int(a["member_id"]) for a in asks_po if int(a["member_id"]) not in members.index]
asks_po = [a for a in asks_po if int(a["member_id"]) in members.index]
asks_sel = asks_in + asks_po
po_idx = {int(a["member_id"]): N + k for k, a in enumerate(asks_po)}      # profile-only MPs get ids N, N+1, ... (beyond the map arrays)
# sk15: dedup first (cheap), then compute excerpts only for (passage, measure) pairs missing from the cache
def _profile_pids(ix):
    return {p for v in (fight_claims.get(ix, []), mp_one.get(ix, [])) for fc in v for p, _ in fc[2]}
asks_kept = []; n_before = n_removed = n_lost = n_before_in = 0
for a in asks_sel:
    ix = mi.get(int(a["member_id"]))
    used = _profile_pids(ix) if ix is not None else set()
    keep = [k for k in a["asks"] if not (used & ({k["passage_id"]} | set(k.get("receipts") or [])))]
    n_before += len(a["asks"]); n_removed += len(a["asks"]) - len(keep)
    n_lost += bool(a["asks"]) and not keep
    asks_kept.append((a, keep))
CACHE_F = ROOT / "data/quick/research_2026_11_10/sk15/excerpts.json"; CACHE_F.parent.mkdir(parents=True, exist_ok=True)
CACHE = json.loads(CACHE_F.read_text()) if CACHE_F.exists() else {}      # key "passage_id|measure" -> [excerpt, original_text_flag (None = no passage text)]
_ck = lambda k: k["passage_id"] + "|" + k["measure"]
todo = [k for a, keep in asks_kept for k in keep if _ck(k) not in CACHE]
print("excerpt cache:", len(CACHE), "entries;", len(todo), "to compute")
if todo:
    _need = {k["passage_id"] for k in todo}
    _pp = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "text"])
    _pp = _pp[_pp.passage_id.isin(_need)].drop_duplicates("passage_id").set_index("passage_id")["text"].to_dict()
    _texts = {}
    for pid in _need:
        st = _pp.get(pid)
        if st is None: continue
        try: o = recover(pid, str(st))
        except Exception: o = None
        _texts[pid] = (o, True) if o else (str(st), False)
    for k in todo:
        if k["passage_id"] in _texts: _encode([k["measure"]] + split_sents(_texts[k["passage_id"]][0]))
    for k in todo:
        pid = k["passage_id"]
        CACHE[_ck(k)] = [best_excerpt(k["measure"], _texts[pid][0]), _texts[pid][1]] if pid in _texts else ["", None]
    CACHE_F.write_text(json.dumps(CACHE, ensure_ascii=False))
_uniq = {k["passage_id"]: CACHE[_ck(k)][1] for a, keep in asks_kept for k in keep}
n_orig = sum(1 for v in _uniq.values() if v is True); n_stored = sum(1 for v in _uniq.values() if v is False)
ASKS = {}; n_asks = n_nolink = n_nopass = 0
for a, keep in asks_kept:
    if not keep: continue
    rows = []
    for k in keep:
        pid = k["passage_id"]
        ex = CACHE[_ck(k)][0]; n_nopass += (CACHE[_ck(k)][1] is None)
        u_ = url(pid); n_nolink += (u_ is None)
        rows.append([0 if k["side"] == "FOR" else 1, k["measure"], k.get("debate_title") or "", (k.get("date") or "")[:10], int(k.get("other_mps_sharing") or 0), ex, u_])
    rows.sort(key=lambda r: (r[0], -r[4], "".join(chr(0x10FFFF - ord(c)) for c in r[3])))      # list, then shared-with most first, then newest first
    mid = int(a["member_id"]); ASKS[str(mi[mid] if mid in mi else po_idx[mid])] = rows; n_asks += len(rows)
PO = []
for a in asks_po:
    r = members.loc[int(a["member_id"])]
    PO.append({"id": int(a["member_id"]), "n": r["name"] or a.get("name") or str(a["member_id"]), "p": r.party_abbr or "Ind", "c": r.constituency, "col": "#" + str(r.party_colour)})
n_po_with = sum(1 for a in asks_po if str(po_idx[int(a["member_id"])]) in ASKS)
n_in_with = len(ASKS) - n_po_with
print(f"asks: {len(ASKS)} MPs ({n_in_with} in-app, {n_po_with} profile-only), {n_before} before dedup, {n_removed} removed, {n_asks} embedded, {n_lost} MPs lost whole section; original text {n_orig}, stored {n_stored}, no link {n_nolink}; profile-only {len(PO)}, missing from members {len(po_missing)}")
# 3+ entries (stances + took-a-side [+ asks]) per MP
_e0 = [len(fight_claims.get(i, [])) + len(mp_one.get(i, [])) for i in range(N)]
_e1 = [_e0[i] + len(ASKS.get(str(i), [])) for i in range(N)]
_epo = [len(ASKS.get(str(N + k), [])) for k in range(len(PO))]
(RES / "sk13_build_info.json").write_text(json.dumps({"mps_with_asks": len(ASKS), "asks_embedded": n_asks, "excerpt_original": n_orig, "excerpt_stored": n_stored, "rows_without_link": n_nolink, "rows_without_passage_text": n_nopass,
    "asks_mps_not_selectable": 0, "asks_mps_total": len([a for a in ASKS_RAW if a["asks"]])}, indent=1))
(RES / "sk14_build_info.json").write_text(json.dumps({"mps_on_map": int(on_map.sum()), "mps_in_app": N, "profile_only": len(PO), "profile_only_missing_from_members": po_missing, "findable_total": N + len(PO),
    "asks_mps_in_app": n_in_with, "asks_mps_profile_only": n_po_with, "asks_before_dedup": n_before, "asks_removed": n_removed, "asks_after": n_asks, "mps_lost_whole_asks_section": n_lost,
    "mps_3plus_entries_original_app_in_app": sum(e >= 3 for e in _e0), "mps_3plus_entries_copy_in_app": sum(e >= 3 for e in _e1), "mps_3plus_entries_copy_profile_only": sum(e >= 3 for e in _epo)}, indent=1))
# ---------------- focus (topic attention share from all policy speeches) ----------------
_pm = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["member_id"])
_hm = pd.read_parquet(ROOT / "data/quick/research_2026_11_10/homes.parquet", columns=["home"])
_pm["home"] = _hm["home"].values
_pm = _pm[_pm["home"].isin(subj_idx_map)].copy()
_pm["sk"] = _pm["home"].map(subj_idx_map).astype(int)
_mp_tot = _pm.groupby("member_id").size().to_dict()
_mp_sub = _pm.groupby(["member_id", "sk"]).size()
_parl_tot = len(_pm)
_parl_sub = _pm.groupby("sk").size().to_dict()
for k, s in enumerate(subj_out):
    s["avg_focus"] = round(100.0 * _parl_sub.get(k, 0) / max(1, _parl_tot), 1)

_all_searchable_ids = [int(m["id"]) for m in mp_out] + [int(m["id"]) for m in PO]
FOCUS = {}
_subj_speakers = collections.defaultdict(list)
for idx, mid in enumerate(_all_searchable_ids):
    tot = int(_mp_tot.get(mid, 0))
    if tot <= 0: continue
    rows_f = []
    if mid in _mp_sub.index.get_level_values(0):
        sub_s = _mp_sub.loc[mid]
        for sk_val, cnt_val in sub_s.items():
            sk_i = int(sk_val); cnt_i = int(cnt_val)
            pct = round(100.0 * cnt_i / tot, 1)
            rows_f.append([sk_i, cnt_i, pct])
            if idx < N and cnt_i >= 3:
                _subj_speakers[sk_i].append((pct, cnt_i, idx))
    rows_f.sort(key=lambda r: (-r[2], -r[1]))
    FOCUS[str(idx)] = {"tot": tot, "rows": rows_f}
for k, s in enumerate(subj_out):
    top_sp = sorted(_subj_speakers[k], key=lambda x: (-x[0], -x[1]))[:8]
    s["top_focus"] = [[idx, cnt, pct] for pct, cnt, idx in top_sp]
print(f"focus computed for {len(FOCUS)}/{len(_all_searchable_ids)} MPs across {_parl_tot} policy passages")

# ---------------- axis explainers directly from Z ----------------
axl = []
axes = []
for ax in range(3):
    loads = []
    for k, t in enumerate(SUBJ):
        ok = ~np.isnan(S[:, k]) & on_map
        if ok.sum() >= 12 and np.std(S[ok, k]) > 1e-8 and np.std(Z[ok, ax]) > 1e-8:
            c = float(np.corrcoef(Z[ok, ax], S[ok, k])[0, 1])
            loads.append([t, round(c, 3)])
    loads.sort(key=lambda x: -abs(x[1]))
    axl.append(loads)
    axes.append(", ".join(x[0] for x in loads[:2]) if loads else f"Axis {ax + 1}")

model_out = {"v": [[round(float(x), 6) for x in row] for row in vv],
             "b": [round(float(x), 6) for x in b_dir],
             "ridge": round(float(ridge_direct), 6)}

data = {"axl": axl, "mps": mp_out, "xyz": [[round(float(x), 3) for x in z] for z in Z], "onmap": on_map.tolist(), "subjects": subj_out, "fights": fights_out,
        "groups": groups, "gstab": round(stab_val, 2), "P": Ps, "mpclaims": {str(i): v for i, v in fight_claims.items()}, "mpone": {str(i): v for i, v in mp_one.items()},
        "os": os_out, "axes": axes, "asks": ASKS, "po": PO, "model": model_out, "focus": FOCUS,
        "about": {**{k: _tot[k] for k in ("passages_extracted", "claims", "dense_candidates", "fights_reported")}, "fights_merged": n_after, "fights_two_sided": len(MAPF),
                  "mps": N, "mps_on_map": int(on_map.sum()), "subjects": NT}}
js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
tpl = (HERE / "hierarchy_app_template.html").read_text()
out = HERE / "app_hierarchy.html"
tpl = tpl.replace("/*__HA_QUIZ__*/", (HERE / "ha_quiz.js").read_text()).replace("/*__HA_REL__*/", (HERE / "ha_relationships.js").read_text())
page = tpl.replace("__DATA__", js)
for target in (out, HERE / "sk13_app_asks.html", ROOT / "docs/index.html", ROOT / "docs/stance.html"):
    target.write_text(page)
print(out, round(out.stat().st_size / 1e6, 2), "MB")

lf = [f for f in fights_out if f["u"]]
summary = {"pilot_used": USE_PILOT, "mps_3plus_subjects": int(((~np.isnan(S)).sum(1) >= 3).sum()), "mps_any_position": len({m for f in FALL for k in ("mps_for", "mps_against", "mps_both_sides") for m in [x["member_id"] for x in f[k]]}),
           "n_mps": N, "fights_before_merge": n_before, "fights_after_merge": n_after, "fights_merged_away": n_before - n_after,
           "pairs": {"judged_duplicate_sim_ge_0.85": n_judged, "auto_rule_two_sided": n_auto},
           "merge_groups": [{"label": f["label"], "orig_ids": f["orig"]} for f in MAPF + POOL if f["n_merged"] > 1],
           "two_sided_fights": {"before": len(two), "after": len(MAPF), "with_line": len(lf)},
           "one_sided_pool_fights": len(POOL),
           "mps_with_two_sided_position": len(fight_claims), "mps_with_one_sided_entries": len(mp_one),
           "mps_with_both": len(set(fight_claims) & set(mp_one)), "one_sided_entries_total": sum(len(v) for v in mp_one.values()),
           "mean_r_heldout_fights": float(np.mean([f["r"] for f in lf])),
           "groups": [g["name"] for g in groups], "groups_kept": groups_ok}
(RES / "ha_summary.json").write_text(json.dumps(summary, indent=1, default=float))
print(json.dumps(summary, indent=1, default=float))
