"""Party line + individual deviation model; hold-out evaluation. Free, local, seed 20261110."""
import numpy as np, collections

def als(rows, cols, y, M, N, rank, lam, iters=40, seed=20261110):
    rng = np.random.default_rng(seed)
    U = rng.normal(0, .1, (M, rank)); V = rng.normal(0, .1, (N, rank))
    byr = [np.where(rows == a)[0] for a in range(M)]; byc = [np.where(cols == b)[0] for b in range(N)]
    I = lam * np.eye(rank)
    for _ in range(iters):
        for a in range(M):
            ix = byr[a]
            if len(ix): Vx = V[cols[ix]]; U[a] = np.linalg.solve(Vx.T @ Vx + I, Vx.T @ y[ix])
            else: U[a] = 0
        for b in range(N):
            ix = byc[b]
            if len(ix): Ux = U[rows[ix]]; V[b] = np.linalg.solve(Ux.T @ Ux + I, Ux.T @ y[ix])
            else: V[b] = 0
    return U, V

def party_line(pcode, cm, cf, cy, P, N, k=1.0):
    """Shrunk party mean per (party, fight) from the given cells. Returns B[P,N], plus raw sums S[P,N], fight sums."""
    S = np.zeros((P, N)); n = np.zeros((P, N))
    np.add.at(S, (pcode[cm], cf), cy); np.add.at(n, (pcode[cm], cf), 1)
    Sf = S.sum(0); nf = n.sum(0)
    mj = np.where(nf > 0, Sf / np.maximum(nf, 1), 0.0)
    B = (S + k * mj[None, :]) / (n + k)
    return B, S, Sf

def majority_sign(S, Sf, p, f):
    s = S[p, f]; s = np.where(s != 0, s, Sf[f])
    return np.where(s >= 0, 1, -1)

RS = (0, 1, 2, 3); LAMS = (0.3, 1.0, 3.0, 10.0)
CONFIGS = [(0, 1.0)] + [(r, l) for r in (1, 2, 3) for l in LAMS]

def choose_config(pcode, cm, cf, cy, P, M, N, seed):
    """Inner 5-fold CV on training cells only. Returns (R, lam, table)."""
    rng = np.random.default_rng(seed); fold = rng.permutation(len(cy)) % 5
    hits = collections.defaultdict(int)
    for k in range(5):
        tr, te = fold != k, fold == k
        B, _, _ = party_line(pcode, cm[tr], cf[tr], cy[tr], P, N)
        btr = B[pcode[cm[tr]], cf[tr]]; r = cy[tr] - btr
        bte = B[pcode[cm[te]], cf[te]]; yt = np.where(cy[te] > 0, 1, -1)
        for (R, lam) in CONFIGS:
            if R == 0: p = bte
            else:
                U, V = als(cm[tr], cf[tr], r, M, N, R, lam, iters=25)
                p = bte + (U[cm[te]] * V[cf[te]]).sum(1)
            hits[(R, lam)] += int((np.where(p >= 0, 1, -1) == yt).sum())
    best = None
    for c in CONFIGS:   # ascending R: ties go to the simpler model
        if best is None or hits[c] > hits[best]: best = c
    return best[0], best[1], {f"{c[0]}|{c[1]}": hits[c] / len(cy) for c in CONFIGS}

def holdout(cm, cf, cy, cd, party_of, M, N, seed=20261110, n_perm=3, log=print):
    parties = sorted(set(party_of)); pi = {p: i for i, p in enumerate(parties)}
    pcode = np.array([pi[p] for p in party_of]); P = len(parties)
    csg = np.where(cy > 0, 1, -1)
    rng = np.random.default_rng(seed); fold = rng.permutation(len(cy)) % 5
    ncell_mp = np.bincount(cm, minlength=M)
    pred = {k: np.zeros(len(cy), int) for k in "abcd"}; perm = np.zeros((n_perm, len(cy)), int)
    againstflag = np.zeros(len(cy), bool); chosen = []
    for k in range(5):
        tr, te = fold != k, fold == k
        # (a) old party-blind map
        U, V = als(cm[tr], cf[tr], cy[tr], M, N, 3, 1.0, iters=60)
        pred["a"][te] = np.where((U[cm[te]] * V[cf[te]]).sum(1) >= 0, 1, -1)
        # (b),(c)
        B, S, Sf = party_line(pcode, cm[tr], cf[tr], cy[tr], P, N)
        maj = majority_sign(S, Sf, pcode[cm[te]], cf[te]); pred["b"][te] = maj
        bte = B[pcode[cm[te]], cf[te]]; pred["c"][te] = np.where(bte >= 0, 1, -1)
        againstflag[te] = maj != csg[te]
        # (d) inner CV
        R, lam, tab = choose_config(pcode, cm[tr], cf[tr], cy[tr], P, M, N, seed + 1 + k)
        chosen.append((R, lam)); log(f"outer fold {k}: R={R} lam={lam}")
        r = cy[tr] - B[pcode[cm[tr]], cf[tr]]
        if R == 0: pred["d"][te] = pred["c"][te]
        else:
            U, V = als(cm[tr], cf[tr], r, M, N, R, lam, iters=60)
            pred["d"][te] = np.where(bte + (U[cm[te]] * V[cf[te]]).sum(1) >= 0, 1, -1)
        # permutation: shuffle residuals within fight, training only
        cft = cf[tr]
        for q in range(n_perm):
            prng = np.random.default_rng(seed + 100 + 10 * k + q); rp = r.copy()
            for j in np.unique(cft):
                ix = np.where(cft == j)[0]; rp[ix] = r[prng.permutation(ix)]
            if R == 0: perm[q, te] = pred["c"][te]
            else:
                U, V = als(cm[tr], cf[tr], rp, M, N, R, lam, iters=60)
                perm[q, te] = np.where(bte + (U[cm[te]] * V[cf[te]]).sum(1) >= 0, 1, -1)
    def acc(p, mask=None):
        m = np.ones(len(cy), bool) if mask is None else mask
        return float((p[m] == csg[m]).mean()) if m.sum() else None
    ag = againstflag; big = ncell_mp[cm] >= 5
    out = {"n": int(len(cy)), "against_n": int(ag.sum()), "n_mps_ge5_cells": int((ncell_mp >= 5).sum()), "cells_ge5": int(big.sum()), "against_ge5_n": int((ag & big).sum()),
           "chosen_R_counts": dict(collections.Counter(str(c[0]) for c in chosen)), "chosen": [list(c) for c in chosen]}
    for k, nm in zip("abcd", ("old_map", "party_majority", "party_line_shrunk", "party_line_plus_deviation")):
        out[nm] = {"overall": acc(pred[k]), "against": acc(pred[k], ag), "ge5_cells": acc(pred[k], big), "against_ge5": acc(pred[k], ag & big)}
    pm = [acc(perm[q]) for q in range(n_perm)]; pa = [acc(perm[q], ag) for q in range(n_perm)]
    out["permutation"] = {"overall_mean": float(np.mean(pm)), "against_mean": float(np.mean(pa)), "repeats": n_perm,
                          "gain_real_vs_c_overall": out["party_line_plus_deviation"]["overall"] - out["party_line_shrunk"]["overall"],
                          "gain_perm_vs_c_overall": float(np.mean(pm)) - out["party_line_shrunk"]["overall"],
                          "gain_real_vs_c_against": out["party_line_plus_deviation"]["against"] - out["party_line_shrunk"]["against"],
                          "gain_perm_vs_c_against": float(np.mean(pa)) - out["party_line_shrunk"]["against"]}
    dn = cd
    out["dense_only"] = {nm: acc(pred[k], dn) for k, nm in zip("abcd", ("old_map", "party_majority", "party_line_shrunk", "party_line_plus_deviation"))}
    out["dense_n"] = int(dn.sum())
    return out, pcode, parties
