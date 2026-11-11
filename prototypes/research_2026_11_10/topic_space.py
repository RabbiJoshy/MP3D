"""Topic-first space. Free, local, seed 20261110.
1) per-fight free-vote (conscience) detection, 2) one stance scale per topic (rank-1 ALS on MP x fight cells),
3) per-MP/topic focus, 4) overall rank-3 map on topic scores (each topic standardised: counts once),
5) hold-out on topic scores.  Usage: topic_space.py [fights.json]  ->  results/fp_topic_space.json, fp_topic_holdout.json
Generic: nothing here names a topic, fight or MP."""
import json, pathlib, sys, collections
import numpy as np

ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
R = ROOT / "prototypes/research_2026_11_10/results"
SEED = 20261110
CONSC = "Conscience (free votes)"
MIN_PARTY, MIN_MINORITY = 5, 0.20       # free-vote rule
MIN_MPS_MAP = 5                         # a topic scale needs this many scored MPs to enter the overall map
STANCE_STATUS = ("fight", "one_short")


def als(rows, cols, y, M, N, rank, lam, iters=60, seed=SEED, init_ones=False):
    rng = np.random.default_rng(seed)
    U = rng.normal(0, .1, (M, rank)); V = rng.normal(0, .1, (N, rank))
    if init_ones: V[:] = 1.0
    byr = [np.where(rows == a)[0] for a in range(M)]; byc = [np.where(cols == b)[0] for b in range(N)]
    I = lam * np.eye(rank)
    for _ in range(iters):
        for a in range(M):
            ix = byr[a]
            U[a] = np.linalg.solve(V[cols[ix]].T @ V[cols[ix]] + I, V[cols[ix]].T @ y[ix]) if len(ix) else 0
        for b in range(N):
            ix = byc[b]
            V[b] = np.linalg.solve(U[rows[ix]].T @ U[rows[ix]] + I, U[rows[ix]].T @ y[ix]) if len(ix) else 0
    return U, V


def free_vote(f):
    """Returns dict(flag, parties=[{party,n,minority,share,split}]) for the two largest parties on the fight."""
    pos = collections.defaultdict(lambda: [0, 0])      # party -> [for, against]
    for m in f["mps_for"]: pos[m["party"]][0] += 1
    for m in f["mps_against"]: pos[m["party"]][1] += 1
    tot = sorted(((sum(v), p) for p, v in pos.items()), reverse=True)[:2]
    out = []
    for n, p in tot:
        mn = min(pos[p]); sh = mn / n if n else 0
        out.append({"party": p, "n": n, "for": pos[p][0], "against": pos[p][1], "minority": mn, "share": round(sh, 3),
                    "split": bool(n >= MIN_PARTY and sh >= MIN_MINORITY)})
    return {"flag": len(out) == 2 and all(x["split"] for x in out), "parties": out}


def fight_cells(f):
    return [(m["member_id"], m["name"], m["party"], float(m["stance"])) for k in ("mps_for", "mps_against") for m in f[k]]


def build(F, log=print):
    # ---- 1. conscience
    fv = [free_vote(f) for f in F]
    for f, v in zip(F, fv): f["_fv"] = v
    cons = [i for i, v in enumerate(fv) if v["flag"]]
    topic_of = {i: (CONSC if v["flag"] else F[i]["topic"]) for i, v in enumerate(fv)}
    # ---- MP registry
    mp_info = {}
    for f in F:
        for k in ("mps_for", "mps_against", "mps_both_sides"):
            for m in f[k]: mp_info[m["member_id"]] = (m["name"], m["party"])
    # ---- 2/3. topic scales + focus
    by_t = collections.defaultdict(list)
    for i in range(len(F)): by_t[topic_of[i]].append(i)
    scales = {}; focus = collections.defaultdict(dict)       # focus[topic][mp]
    for t, idx in by_t.items():
        for i in idx:
            for k in ("mps_for", "mps_against", "mps_both_sides"):
                for m in F[i][k]:
                    focus[t][m["member_id"]] = focus[t].get(m["member_id"], 0) + int(m.get("focus") or 0)
        use = [i for i in idx if F[i]["status"] in STANCE_STATUS]
        sc = {"topic": t, "fights_all": idx, "fights_used": use, "scale": False}
        two = [i for i in use if F[i]["mps_for"] and F[i]["mps_against"]]
        if not two:
            scales[t] = sc; continue
        cells = {}
        for j, i in enumerate(use):
            for mid, nm, pt, st in fight_cells(F[i]): cells[(mid, j)] = st
        mids = sorted({k[0] for k in cells}); mi = {m: a for a, m in enumerate(mids)}
        rows = np.array([mi[k[0]] for k in cells]); cols = np.array([k[1] for k in cells]); y = np.array(list(cells.values()))
        if len(use) == 1:
            s = np.array([y[rows == a][0] for a in range(len(mids))]); v = np.array([1.0]); expl = None; expl_multi = None; n_multi = 0
        else:
            U, V = als(rows, cols, y, len(mids), len(use), 1, 0.05, iters=100, init_ones=True)
            s = U[:, 0].copy(); v = V[:, 0].copy(); sc_ = np.abs(v).max() or 1.0
            v /= sc_
            # final scores: ridge with the loadings fixed (an MP seen only on a low-loading fight is pulled to 0)
            num = np.zeros(len(mids)); den = np.zeros(len(mids)); np.add.at(num, rows, v[cols] * y); np.add.at(den, rows, v[cols] ** 2)
            s = num / (den + 0.1)
            pred = s[rows] * v[cols]
            expl = float(1 - ((y - pred) ** 2).sum() / (y ** 2).sum())
            cnt = np.bincount(rows, minlength=len(mids)); mm = cnt[rows] >= 2; n_multi = int((cnt >= 2).sum())
            expl_multi = float(1 - ((y[mm] - pred[mm]) ** 2).sum() / (y[mm] ** 2).sum()) if mm.any() else None
        # orientation: Labour mean >= 0 (else the largest party)
        pt_arr = np.array([mp_info[m][1] for m in mids])
        ref = "Lab" if (pt_arr == "Lab").any() else collections.Counter(pt_arr).most_common(1)[0][0]
        if s[pt_arr == ref].mean() < 0: s = -s; v = -v
        sc.update(scale=True, loadings=[{"i": i, "label": F[i]["label"], "v": round(float(v[j]), 3)} for j, i in enumerate(use)],
                  explained=None if expl is None else round(expl, 3), explained_multi=None if expl_multi is None else round(expl_multi, 3),
                  n_multi=n_multi, ref_party=ref, mps={int(m): round(float(s[mi[m]]), 3) for m in mids})
        scales[t] = sc
    # ---- 4. overall map
    tnames = [t for t in sorted(scales) if scales[t]["scale"] and len(scales[t]["mps"]) >= MIN_MPS_MAP]
    excluded = [t for t in sorted(scales) if scales[t]["scale"] and t not in tnames]
    all_mps = sorted({m for t in tnames for m in scales[t]["mps"]})
    ti = {t: k for k, t in enumerate(tnames)}
    raw = {(m, ti[t]): s for t in tnames for m, s in scales[t]["mps"].items()}
    n_by_mp = collections.Counter(m for (m, _) in raw)
    cov = {k: int(sum(1 for m in n_by_mp if n_by_mp[m] >= k)) for k in (1, 2, 3, 4)}
    mps = [m for m in all_mps if n_by_mp[m] >= 2]
    mi = {m: a for a, m in enumerate(mps)}
    cm = np.array([mi[m] for (m, _) in raw if m in mi]); cf = np.array([k for (m, k) in raw if m in mi])
    cy = np.array([s for (m, k), s in raw.items() if m in mi])
    # standardise each topic column (centre, unit variance), using the MPs on the map
    Z = cy.copy(); mu = np.zeros(len(tnames)); sd = np.ones(len(tnames))
    for k in range(len(tnames)):
        ix = cf == k
        if ix.sum() > 1:
            mu[k] = cy[ix].mean(); sd[k] = cy[ix].std() or 1.0
        Z[ix] = (cy[ix] - mu[k]) / sd[k]
    party_of = np.array([mp_info[m][1] for m in mps]); parties = sorted(set(party_of)); pcode = np.array([parties.index(p) for p in party_of])
    U, V = als(cm, cf, Z, len(mps), len(tnames), 3, 1.0, iters=100)

    def finish(U, V, Zc=None):
        Uc = U - U.mean(0); _, _, Vt = np.linalg.svd(Uc, full_matrices=False)
        Rm = Vt.T; Up = Uc @ Rm; Vp = V @ Rm
        for a in range(3):
            if Vp[np.argmax(np.abs(Vp[:, a])), a] < 0: Up[:, a] *= -1; Vp[:, a] *= -1
        spread = (Up.var(0) / Up.var(0).sum())
        return Up, Vp, spread

    def eta2(x, pc):
        tot = ((x - x.mean()) ** 2).sum()
        if tot == 0: return 0.0
        return float(sum(((pc == p).sum()) * (x[pc == p].mean() - x.mean()) ** 2 for p in np.unique(pc)) / tot)

    def axes_of(Up, Vp, spread, pc):
        ax = []
        for a in range(3):
            ld = Vp[:, a] * Up[:, a].std()
            o = np.argsort(-np.abs(ld))
            ax.append({"spread": round(float(spread[a]), 3), "party_eta2": round(eta2(Up[:, a], pc), 3),
                       "loadings": [{"topic": tnames[k], "v": round(float(ld[k]), 3)} for k in o]})
        return ax
    Up, Vp, spread = finish(U, V)
    axes = axes_of(Up, Vp, spread, pcode)
    # party mean of each topic score (raw), for the MP card
    pm = {}
    for t in tnames:
        d_ = collections.defaultdict(list)
        for m, s in scales[t]["mps"].items(): d_[mp_info[m][1]].append(s)
        pm[t] = {p: [round(float(np.mean(v)), 3), len(v)] for p, v in d_.items()}
    mps_pts = [{"id": int(m), "mp": mp_info[m][0], "party": mp_info[m][1], "xyz": [round(float(x), 3) for x in Up[k]],
                "ts": [[tnames[ti_], round(float(raw[(m, ti_)]), 3), focus[tnames[ti_]].get(m, 0)] for ti_ in range(len(tnames)) if (m, ti_) in raw]}
               for k, m in enumerate(mps)]
    # ---- beyond party: residual from the party's (shrunk) mean topic score, MPs with >=3 topic scores
    def party_mean_z(cm_, cf_, z_, k=1.0, loo=False):
        P = len(parties); S = np.zeros((P, len(tnames))); n = np.zeros((P, len(tnames)))
        np.add.at(S, (pcode[cm_], cf_), z_); np.add.at(n, (pcode[cm_], cf_), 1)
        mj = np.where(n.sum(0) > 0, S.sum(0) / np.maximum(n.sum(0), 1), 0.0)
        if not loo: return (S[pcode[cm_], cf_] + k * mj[cf_]) / (n[pcode[cm_], cf_] + k), None
        nn = n[pcode[cm_], cf_] - 1
        return (S[pcode[cm_], cf_] - z_ + k * mj[cf_]) / (nn + k), nn
    ncell = np.bincount(cm, minlength=len(mps))
    keep = ncell >= 3
    beyond = {"n_mps": int(keep.sum()), "ok": bool(keep.sum() >= 30)}
    pmz, _ = party_mean_z(cm, cf, Z)
    res = Z - pmz
    if beyond["ok"]:
        kc = keep[cm]; remap = {a: b for b, a in enumerate(np.where(keep)[0])}
        cm2 = np.array([remap[a] for a in cm[kc]])
        Ub, Vb = als(cm2, cf[kc], res[kc], int(keep.sum()), len(tnames), 3, 1.0, iters=100)
        Ubp, Vbp, bspread = finish(Ub, Vb)
        pc2 = pcode[keep]
        beyond.update(axes=axes_of(Ubp, Vbp, bspread, pc2),
                      mps=[{"id": int(mps[a]), "mp": mp_info[mps[a]][0], "party": mp_info[mps[a]][1], "xyz": [round(float(x), 3) for x in Ubp[remap[a]]],
                            "ts": mps_pts[a]["ts"]} for a in np.where(keep)[0]])
    # ---- distinctive: leave-one-out party mean, MPs with >=3 non-conscience topic scores
    nc = np.array([tnames[k] != CONSC for k in cf])
    cnt_nc = np.bincount(cm[nc], minlength=len(mps))
    lpm, nn = party_mean_z(cm, cf, Z, loo=True)
    dist = []
    for a in range(len(mps)):
        if cnt_nc[a] < 3: continue
        ix = np.where((cm == a) & nc & (nn >= 1))[0]
        if len(ix) < 3: continue
        dv = Z[ix] - lpm[ix]
        o = np.argsort(-np.abs(dv))[:2]
        dist.append({"id": int(mps[a]), "mp": mp_info[mps[a]][0], "party": mp_info[mps[a]][1], "n": int(len(ix)),
                     "dev": round(float(np.sqrt((dv ** 2).mean())), 2),
                     "top": [{"topic": tnames[cf[ix[q]]], "z": round(float(dv[q]), 2), "score": round(float(cy[ix[q]]), 2),
                              "party_mean": round(float(lpm[ix[q]] * sd[cf[ix[q]]] + mu[cf[ix[q]]]), 2)} for q in o]})
    dist.sort(key=lambda x: -x["dev"])
    n_dist_eligible = len(dist)
    # ---- 5. hold-out on topic cells (raw scores)
    ai = {m: a for a, m in enumerate(all_mps)}; allp = sorted({mp_info[m][1] for m in all_mps})
    hm = np.array([ai[m] for (m, _) in raw]); hf = np.array([k for (_, k) in raw]); hy = np.array(list(raw.values()))
    hp = np.array([allp.index(mp_info[m][1]) for m in all_mps])
    ho = holdout(hm, hf, hy, hp, len(all_mps), len(tnames), len(allp), log)
    ho.update(n_mps=len(mps), n_topics=len(tnames), n_cells=int(len(hy)), coverage=cov, n_mps_all=len(all_mps))
    out = {"conscience": [{"i": i, "label": F[i]["label"], "topic": F[i]["topic"], "parties": fv[i]["parties"]} for i in cons],
           "free_vote": [v for v in fv], "topic_of": topic_of, "scales": scales, "focus": {t: {int(k): v for k, v in d_.items()} for t, d_ in focus.items()},
           "tnames": tnames, "excluded_small": excluded, "no_scale": [t for t in sorted(scales) if not scales[t]["scale"]],
           "coverage": cov, "map": {"n_mps": len(mps), "mps": mps_pts, "axes": axes, "party_mean": pm}, "beyond": beyond,
           "distinct": {"eligible": n_dist_eligible, "list": dist[:15]}, "holdout": ho}
    for f in F: f.pop("_fv", None)
    return out


def holdout(cm, cf, y, pcode_mp, M, T, P, log=print, seed=SEED):
    pcode = pcode_mp   # party code per MP row
    sg = lambda x: np.where(x >= 0, 1, -1)

    def party_mean(cm_, cf_, y_, k=1.0):
        S = np.zeros((P, T)); n = np.zeros((P, T)); np.add.at(S, (pcode[cm_], cf_), y_); np.add.at(n, (pcode[cm_], cf_), 1)
        tn = np.zeros(T); tc = n.sum(0); tn = np.where(tc > 0, S.sum(0) / np.maximum(tc, 1), 0.0)
        return (S + k * tn[None, :]) / (n + k), tn

    CFG = [(0, 1.0)] + [(r, l) for r in (1, 2) for l in (1.0, 3.0, 10.0)]

    def fit_pred(tr, te, R, lam):
        B, _ = party_mean(cm[tr], cf[tr], y[tr]); b_te = B[pcode[cm[te]], cf[te]]
        if R == 0: return b_te
        r = y[tr] - B[pcode[cm[tr]], cf[tr]]
        U, V = als(cm[tr], cf[tr], r, M, T, R, lam, iters=40)
        return b_te + (U[cm[te]] * V[cf[te]]).sum(1)

    def choose(tr_idx, sd):
        rng = np.random.default_rng(sd); fold = rng.permutation(len(tr_idx)) % 5
        err = collections.defaultdict(float)
        for k in range(5):
            a = tr_idx[fold != k]; b = tr_idx[fold == k]
            ma = np.zeros(len(y), bool); ma[a] = True; mb = np.zeros(len(y), bool); mb[b] = True
            for R, lam in CFG: err[(R, lam)] += np.abs(fit_pred(ma, mb, R, lam) - y[mb]).sum()
        best = CFG[0]
        for c in CFG:
            if err[c] < err[best] - 1e-9: best = c
        return best
    rng = np.random.default_rng(seed); fold = rng.permutation(len(y)) % 5
    pa = np.zeros(len(y)); pb = np.zeros(len(y)); pc_ = np.zeros(len(y)); chosen = []
    for k in range(5):
        te = fold == k; tr = ~te
        B, tn = party_mean(cm[tr], cf[tr], y[tr]); pb[te] = B[pcode[cm[te]], cf[te]]; pa[te] = tn[cf[te]]
        R, lam = choose(np.where(tr)[0], seed + 1 + k); chosen.append((R, lam)); log(f"outer fold {k}: R={R} lam={lam}")
        pc_[te] = fit_pred(tr, te, R, lam)
    ys = sg(y); ag = sg(pb) != ys
    ncell = np.bincount(cm, minlength=M); multi = ncell[cm] >= 2
    def stat(p, m):
        if not m.any(): return {"sign": None, "mae": None, "n": 0}
        return {"sign": round(float((sg(p[m]) == ys[m]).mean()), 3), "mae": round(float(np.abs(p[m] - y[m]).mean()), 3), "n": int(m.sum())}
    allm = np.ones(len(y), bool)
    out = {"chosen": [list(c) for c in chosen], "R_counts": dict(collections.Counter(str(c[0]) for c in chosen))}
    for nm, p in (("topic_mean", pa), ("party_mean", pb), ("party_plus_rank", pc_)):
        out[nm] = {"overall": stat(p, allm), "against": stat(p, ag), "multi_overall": stat(p, multi), "multi_against": stat(p, ag & multi)}
    out["against_n"] = int(ag.sum()); out["multi_cells"] = int(multi.sum())
    return out


def main():
    inp = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else R / "fp_fights_final3.json"
    d = json.load(open(inp)); out = build(d["fights"])
    json.dump(out["holdout"], open(R / "fp_topic_holdout.json", "w"), indent=1)
    json.dump({k: v for k, v in out.items() if k != "holdout"}, open(R / "fp_topic_space.json", "w"), indent=1)
    print("conscience:", [(c["label"], c["topic"]) for c in out["conscience"]])
    for t, s in out["scales"].items():
        print(f"{t[:26]:26s} fights_used={len(s['fights_used'])} all={len(s['fights_all'])}", (f"mps={len(s['mps'])} expl={s['explained']} multi={s['explained_multi']}({s['n_multi']})" if s["scale"] else "no scale"))
    print("coverage", out["coverage"], "map mps", out["map"]["n_mps"], "beyond", out["beyond"]["n_mps"], out["beyond"]["ok"])
    for a in out["map"]["axes"]: print(a["spread"], a["party_eta2"], [(x["topic"][:16], x["v"]) for x in a["loadings"][:4]])
    print(json.dumps(out["holdout"], indent=1))
    for x in out["distinct"]["list"][:5]: print(x["mp"], x["party"], x["dev"], x["top"][0])


if __name__ == "__main__":
    main()
