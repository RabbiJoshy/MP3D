"""Build prototypes/unified.html from:
1. data/quick/claim_topics.json (20 unsupervised topics scored via 1-Line Claim -> BGE Centroids -> Real Anchor Passages)
2. data/quick/topic_space.npz (all 72 non-chatter speech topics across 260k passages for full MP Focus profiles)
3. data/quick/topic_axes.json (fallback internal axes for the remaining focus topics)
"""
import json
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
HERE = ROOT / "prototypes"

claim_data = json.loads((ROOT / "data/quick/claim_topics.json").read_text())
ts = np.load(ROOT / "data/quick/topic_space.npz", allow_pickle=True)
members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
members[["party", "party_abbr", "constituency"]] = members[["party", "party_abbr", "constituency"]].fillna("")
members["party_colour"] = members.party_colour.fillna("888888")

passages = pd.read_parquet(
    ROOT / "data/clean/passages.parquet",
    columns=["passage_id", "member_id", "role", "date", "major", "minor", "text", "words"],
)
passages["topic"] = ts["topic"]
passages["sim"] = ts["sim"]

is_chatter = ts["is_chatter"]
top_terms = ts["top_terms"]
ts_ids = ts["ids"]
ts_counts = ts["counts"]  # shape (len(ts_ids), 90) word counts per MP per topic

real_k = [int(k) for k in np.where(~is_chatter)[0]]
# Put the 20 claim-scored topics first, then the remaining 52 focus topics
claim_ks = [int(k) for k in claim_data.keys()]
other_ks = [k for k in real_k if k not in claim_ks]
ordered_ks = claim_ks + other_ks
k_to_col = {k: idx for idx, k in enumerate(ordered_ks)}

# Filter to MPs in members.parquet with at least 800 words spoken overall
valid_mask = np.array([int(m) in members.index and ts_counts[i, real_k].sum() >= 600 for i, m in enumerate(ts_ids)])
all_mp_ids = [int(m) for m in ts_ids[valid_mask]]
mp_idx = {m: i for i, m in enumerate(all_mp_ids)}
N_MPS = len(all_mp_ids)
N_TOP = len(ordered_ks)
print(f"Building unified space for {N_MPS} MPs across {N_TOP} topics ({len(claim_ks)} with Claim-Anchor stance axes)")

# 1. Build Focus matrix F_focus[i, col] from ts_counts across all 72 non-chatter topics
W_counts = ts_counts[valid_mask][:, ordered_ks].astype(float)
SH = 2500.0  # word-count shrinkage (~15 speeches)
base_rate = (W_counts.sum(0) + 100.0) / (W_counts.sum() + 100.0 * N_TOP)
emp_rate = (W_counts + SH * base_rate) / (W_counts.sum(1, keepdims=True) + SH)
F_focus = np.sqrt(emp_rate) - np.sqrt(base_rate)
F_focus = (F_focus - F_focus.mean(0)) / (F_focus.std(0) + 1e-9)

# Also compute passage counts per MP per topic
P_counts = passages.groupby(["member_id", "topic"]).size().unstack(fill_value=0)

# 2. Build Stance matrix S_stance[i, c] for the 20 Claim-Anchor topics (2024-26 Parliament)
#    and S_stance_all[i, c] (Combined 2022-26 All History)
hist_path = ROOT / "data/quick/claim_topics_hist.json"
hist_data = json.loads(hist_path.read_text()) if hist_path.exists() else {}

N_ST = len(claim_ks)
S_stance = np.full((N_MPS, N_ST), np.nan)
S_stance_all = np.full((N_MPS, N_ST), np.nan)
for c_idx, tk in enumerate(claim_ks):
    tk_info = claim_data[str(tk)]
    for pt in tk_info["pts"]:
        mid, sc = int(pt[0]), float(pt[1])
        if mid in mp_idx:
            S_stance[mp_idx[mid], c_idx] = sc
            S_stance_all[mp_idx[mid], c_idx] = sc
    hk_info = hist_data.get(str(tk), {})
    for pt in hk_info.get("pts", []):
        mid, sc_h = int(pt[0]), float(pt[1])
        if mid in mp_idx:
            i = mp_idx[mid]
            if np.isnan(S_stance_all[i, c_idx]):
                S_stance_all[i, c_idx] = sc_h
            else:
                S_stance_all[i, c_idx] = round(0.65 * S_stance_all[i, c_idx] + 0.35 * sc_h, 3)

parties = np.array([members.loc[m, "party_abbr"] or "Ind" for m in all_mp_ids])
major_parties = [pa for pa in set(parties) if (parties == pa).sum() >= 5 and pa != "Ind"]


def demean_by_party(X):
    Out = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        obs = ~np.isnan(col)
        if obs.sum() < 5:
            continue
        overall_mu = np.nanmean(col)
        for pa in set(parties):
            mask = (parties == pa) & obs
            if mask.sum() >= 3:
                Out[mask, j] = col[mask] - np.mean(col[mask])
            elif mask.sum() > 0:
                Out[mask, j] = col[mask] - overall_mu
    return Out


def center_party_balanced(X):
    Out = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        obs = ~np.isnan(col)
        if obs.sum() < 5:
            continue
        p_means = [np.mean(col[(parties == pa) & obs]) for pa in major_parties if ((parties == pa) & obs).sum() >= 3]
        mu = np.mean(p_means) if p_means else np.nanmean(col)
        sd = np.nanstd(col) + 1e-9
        Out[:, j] = (col - mu) / sd
    return Out


def fit_whitened_3d(S_mat, F_mat, party_equal_whiten=True, rank=3):
    """Build a clean, non-hallucinating 3D space from observed stances S_mat + focus F_mat:
    1. Unobserved stances stay at 0 (neutral/unobserved) rather than iteratively hallucinating extreme values.
    2. Each MP's stance rows are weighted by coverage so MPs with many scored stances anchor the principal axes,
       while MPs with fewer stances are smoothly placed by their observed stances + focus profile.
    3. Spherically whiten (Cov = I) so every topic is a true diameter through [0, 0, 0].
    """
    F_capped = np.clip(F_mat, -1.5, 1.6)
    n_obs = (~np.isnan(S_mat)).sum(1)
    # Per-MP stance scaling: normalize by sqrt(max(n_obs, 2.5)) so MPs with 2-10 stances have comparable scale
    scale_s = np.sqrt(4.0 / np.maximum(n_obs, 2.5))[:, None]
    S_clean = np.nan_to_num(S_mat, nan=0.0) * scale_s

    w_s = 2.6 / np.sqrt(max(S_mat.shape[1], 1))
    w_f = 0.42 / np.sqrt(F_mat.shape[1])
    X_comb = np.hstack([S_clean * w_s, F_capped * w_f])

    # Fit principal subspace weighting MPs who have at least 2 scored stances higher so the axes are purely substantive
    row_w = np.sqrt(np.where(n_obs >= 2, 1.0 + 0.25 * np.minimum(n_obs, 8), 0.35))[:, None]
    X_weighted = (X_comb - X_comb.mean(0)) * row_w
    _, _, Vt = np.linalg.svd(X_weighted, full_matrices=False)
    Z = (X_comb - X_comb.mean(0)) @ Vt[:rank].T

    if party_equal_whiten:
        p_means = np.array([Z[parties == pa].mean(0) for pa in major_parties if (parties == pa).sum() >= 5])
        Z = Z - p_means.mean(0)
        mp_w = np.ones(len(Z))
        for pa in major_parties:
            m_pa = parties == pa
            mp_w[m_pa] = 1.0 / max(m_pa.sum(), 1)
        mp_w /= mp_w.sum()
        Cov = (Z * mp_w[:, None]).T @ Z + 0.30 * np.cov(Z.T)
    else:
        Z = Z - Z.mean(0)
        Cov = np.cov(Z.T)
    eigvals, eigvecs = np.linalg.eigh(Cov)
    idx = np.argsort(-eigvals)
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    Z_white = Z @ eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-9)))
    rad = np.linalg.norm(Z_white, axis=1)
    Z_white = Z_white / np.percentile(rad, 96)
    return Z_white


def fit_raw_vec(Z3, y):
    obs = ~np.isnan(y)
    if obs.sum() < 10 or np.std(y[obs]) < 1e-8:
        return np.zeros(3)
    Zo = Z3[obs] - Z3[obs].mean(0)
    yo = (y[obs] - np.mean(y[obs])) / np.std(y[obs])
    beta, *_ = np.linalg.lstsq(Zo, yo, rcond=None)
    return beta


def topic_vec(Z3, S_mat, tk, positive_kw):
    """Return regression vector in Z3 pointing toward the pole matching positive_kw."""
    if tk not in claim_ks:
        return np.zeros(3)
    c = claim_ks.index(tk)
    v = fit_raw_vec(Z3, S_mat[:, c])
    v = v / (np.linalg.norm(v) + 1e-9)
    lo_l = claim_data[str(tk)]["pole_lo"].lower()
    hi_l = claim_data[str(tk)]["pole_hi"].lower()
    if any(k in lo_l for k in positive_kw) and not any(k in hi_l for k in positive_kw):
        return -v
    return v


def orient_orthogonally(Z3, v_x, v_y):
    """Apply an exact 3x3 orthogonal rotation Q (Q^T Q = I) so X aligns with v_x and Y aligns with v_y
    without changing spherical whitening or any diameter geometry."""
    e0 = v_x / (np.linalg.norm(v_x) + 1e-9)
    v_y_orth = v_y - np.dot(v_y, e0) * e0
    if np.linalg.norm(v_y_orth) < 1e-8:
        v_y_orth = np.array([0.0, 1.0, 0.0]) - e0[1] * e0
    e1 = v_y_orth / (np.linalg.norm(v_y_orth) + 1e-9)
    e2 = np.cross(e0, e1)
    Q = np.column_stack([e0, e1, e2])
    return Z3 @ Q


S_bal = center_party_balanced(S_stance)
F_bal = center_party_balanced(F_focus)
S_win = center_party_balanced(demean_by_party(S_stance))
F_win = center_party_balanced(demean_by_party(F_focus))

Z3_bal = fit_whitened_3d(S_bal, F_bal, party_equal_whiten=True)
Z3_win = fit_whitened_3d(S_win, F_win, party_equal_whiten=False)

# Orient Z3_bal:
# X = Government / Public Ownership / Green Transition (-) <-> Conservative / Deregulation / Deterrence (+)
# Y = Establishment / Frontbench Caution (-) <-> Progressive / Humanitarian / Conscience Reform (+: Scrap 2-Child Cap, Sanction Israel, Assisted Dying, PR, Public Water)
vx_bal = (
    1.5 * (Z3_bal[np.isin(parties, ["Con", "RUK", "DUP"])].mean(0) - Z3_bal[np.isin(parties, ["Lab", "Green", "PC", "SNP"])].mean(0))
    + topic_vec(Z3_bal, S_bal, 67, ["opposes public"])
    + topic_vec(Z3_bal, S_bal, 33, ["oppose agricultural"])
    + topic_vec(Z3_bal, S_bal, 28, ["stricter", "detain"])
)
vy_bal = (
    1.4 * topic_vec(Z3_bal, S_bal, 61, ["remove", "scrap"])
    + 1.4 * topic_vec(Z3_bal, S_bal, 34, ["sanction", "punitive"])
    + 1.2 * topic_vec(Z3_bal, S_bal, 48, ["legalize", "support"])
    + 1.0 * topic_vec(Z3_bal, S_bal, 46, ["public ownership"])
    + 1.0 * topic_vec(Z3_bal, S_bal, 41, ["proportional", "replace"])
)
Z3_bal = orient_orthogonally(Z3_bal, vx_bal, vy_bal)

# Orient Z3_win (Within-Party / Party Removed):
# X = Backbench Rebels / Welfare & Gaza Critics (-) <-> Frontbench / Government Line (+)
# Y = Social Conservative / Caution (-) <-> Free-Vote Liberal / Assisted Dying & Constitutional Reform (+)
vx_win = (
    1.5 * topic_vec(Z3_win, S_win, 61, ["rely on", "broader", "back government"])
    + 1.5 * topic_vec(Z3_win, S_win, 34, ["ceasefire", "diplomacy", "limited"])
    + 1.3 * topic_vec(Z3_win, S_win, 73, ["support means-testing", "means-test"])
    + 1.1 * topic_vec(Z3_win, S_win, 46, ["regulation"])
)
vy_win = (
    1.6 * topic_vec(Z3_win, S_win, 48, ["legalize", "support"])
    + 1.0 * topic_vec(Z3_win, S_win, 41, ["proportional", "replace"])
    + 0.8 * topic_vec(Z3_win, S_win, 43, ["preserve jury"])
)
Z3_win = orient_orthogonally(Z3_win, vx_win, vy_win)

# Fit All-History (2022-26) 3D spaces and align orthogonally to Z3_bal / Z3_win so toggling is seamless
S_bal_all = center_party_balanced(S_stance_all)
S_win_all = center_party_balanced(demean_by_party(S_stance_all))
Z3_bal_all = fit_whitened_3d(S_bal_all, F_bal, party_equal_whiten=True)
Z3_win_all = fit_whitened_3d(S_win_all, F_win, party_equal_whiten=False)


def procrustes_align(Z_src, Z_ref):
    U, _, Vt = np.linalg.svd(Z_src.T @ Z_ref)
    d = np.linalg.det(U @ Vt)
    D_mat = np.diag([1.0, 1.0, 1.0 if d > 0 else -1.0])
    return Z_src @ (U @ D_mat @ Vt)


Z3_bal_all = procrustes_align(Z3_bal_all, Z3_bal)
Z3_win_all = procrustes_align(Z3_win_all, Z3_win)


def fit_diameter(Z3, y):
    obs = ~np.isnan(y)
    if obs.sum() < 12 or np.std(y[obs]) < 1e-8:
        return [1.0, 0.0, 0.0], 0.0
    Zo = Z3[obs] - Z3[obs].mean(0)
    yo = (y[obs] - np.mean(y[obs])) / np.std(y[obs])
    beta, *_ = np.linalg.lstsq(Zo, yo, rcond=None)
    norm = float(np.linalg.norm(beta))
    if norm < 1e-8:
        return [1.0, 0.0, 0.0], 0.0
    u = (beta / norm).tolist()
    pred = Zo @ beta
    r = float(np.corrcoef(pred, yo)[0, 1]) if np.std(pred) > 1e-8 else 0.0
    return [round(x, 4) for x in u], round(abs(r), 3)


def shrunk_corr(x, y, k_shrink=25):
    obs = ~np.isnan(x) & ~np.isnan(y)
    n = int(obs.sum())
    if n < 15:
        return 0.0, 0.0, n
    sx, sy = np.std(x[obs]), np.std(y[obs])
    if sx < 1e-8 or sy < 1e-8:
        return 0.0, 0.0, n
    r = float(np.corrcoef(x[obs], y[obs])[0, 1])
    return r * (n / (n + k_shrink)), r, n


def short_words(wlist, n=3):
    out = []
    for w in wlist:
        w = str(w)
        if not any(w in x or x in w for x in out):
            out.append(w)
        if len(out) == n:
            break
    return ", ".join(out)


topics_out = []
for col_i, tk in enumerate(ordered_ks):
    t_words = [str(w) for w in top_terms[tk][:8]]
    has_claim = str(tk) in claim_data
    u_foc_bal, r_foc_bal = fit_diameter(Z3_bal, F_bal[:, col_i])
    u_foc_win, r_foc_win = fit_diameter(Z3_win, F_win[:, col_i])

    if has_claim:
        cd = claim_data[str(tk)]
        hk = hist_data.get(str(tk), {})
        c_idx = claim_ks.index(tk)
        u_div_bal, r_div_bal = fit_diameter(Z3_bal, S_bal[:, c_idx])
        u_div_win, r_div_win = fit_diameter(Z3_win, S_win[:, c_idx])
        u_div_bal_all, r_div_bal_all = fit_diameter(Z3_bal_all, S_bal_all[:, c_idx])
        u_div_win_all, r_div_win_all = fit_diameter(Z3_win_all, S_win_all[:, c_idx])
        v_div_win = S_win[:, c_idx]
        title = cd["title"]
        question = cd["question"]
        pole_lo = cd["pole_lo"]
        pole_hi = cd["pole_hi"]
        kind = cd["kind"]
        pts = [
            [mp_idx[int(p[0])], float(p[1]), int(p[2]), int(p[3]), int(p[4])]
            for p in cd["pts"]
            if int(p[0]) in mp_idx
        ]
        quotes = {mp_idx[int(m)]: q for m, q in cd["quotes"].items() if int(m) in mp_idx}

        # Build All-History (2022-26) pts_all and quotes_all
        pts_map_cur = {int(p[0]): p for p in cd["pts"] if int(p[0]) in mp_idx}
        pts_map_hist = {int(p[0]): p for p in hk.get("pts", []) if int(p[0]) in mp_idx}
        quotes_hist_raw = hk.get("quotes", {})
        pts_all = []
        quotes_all = {}
        for mid in sorted(set(pts_map_cur.keys()) | set(pts_map_hist.keys())):
            i = mp_idx[mid]
            p_cur = pts_map_cur.get(mid)
            p_hist = pts_map_hist.get(mid)
            q_cur = dict(cd["quotes"].get(str(mid), {})) if str(mid) in cd["quotes"] else None
            q_hist = dict(quotes_hist_raw.get(str(mid), {})) if str(mid) in quotes_hist_raw else None
            if p_cur and p_hist:
                sc_cur, sc_hist = float(p_cur[1]), float(p_hist[1])
                sc_comb = round(0.65 * sc_cur + 0.35 * sc_hist, 3)
                nsp = int(p_cur[3]) + int(p_hist[3])
                nwords = int(p_cur[4]) + int(p_hist[4])
                # If MP genuinely flipped between 2022-24 and 2024-26, show their current position + cross-parliament shift arrow
                if abs(sc_cur) >= 0.35 and abs(sc_hist) >= 0.35 and sc_cur * sc_hist < 0 and abs(sc_cur - sc_hist) >= 0.85:
                    sc_comb = sc_cur
                pts_all.append([i, sc_comb, 1 if sc_comb > 0 else -1, nsp, nwords])
                if q_cur:
                    q_comb = dict(q_cur)
                    ex_cur = list(q_cur.get("ex", [{"t": q_cur.get("t", ""), "d": q_cur.get("d", ""), "h": q_cur.get("h", "")}]))
                    if q_hist:
                        ex_hist = list(q_hist.get("ex", [{"t": q_hist.get("t", ""), "d": q_hist.get("d", ""), "h": q_hist.get("h", "")}]))
                        q_comb["ex"] = (ex_cur[:2] + ex_hist[:1])
                        if abs(sc_cur) >= 0.35 and abs(sc_hist) >= 0.35 and sc_cur * sc_hist < 0 and abs(sc_cur - sc_hist) >= 0.85:
                            q_comb["shift"] = {
                                "s0": round(sc_hist, 2),
                                "s1": round(sc_cur, 2),
                                "d0": q_hist.get("d", "2023"),
                                "d1": q_cur.get("d", "2025"),
                                "note": f"2022–24: {q_hist.get('c', '')} → 2024–26: {q_cur.get('c', '')}",
                            }
                    quotes_all[i] = q_comb
            elif p_cur:
                pts_all.append([i, float(p_cur[1]), int(p_cur[2]), int(p_cur[3]), int(p_cur[4])])
                if q_cur:
                    quotes_all[i] = q_cur
            elif p_hist:
                pts_all.append([i, float(p_hist[1]), int(p_hist[2]), int(p_hist[3]), int(p_hist[4])])
                if q_hist:
                    q_h = dict(q_hist)
                    q_h["c"] = f"[2022–24 Parliament] {q_h.get('c', '')}"
                    quotes_all[i] = q_h

        n_lo_all = sum(1 for p in pts_all if p[1] < -0.15)
        n_hi_all = sum(1 for p in pts_all if p[1] > 0.15)
        stats = {
            "n_lo": cd["n_lo"],
            "n_hi": cd["n_hi"],
            "n_neu": cd["n_neu"],
            "split_balance": cd["split_balance"],
            "party_r2": cd["party_r2"],
            "bb_party_r2": cd["bb_party_r2"],
            "jev_r_all": cd["jev_r_all"],
            "jev_r_lab": cd["jev_r_lab"],
            "vote_auc": cd["vote_auc"],
            "vote_auc_lab": cd["vote_auc_lab"],
            "wiki_n": cd.get("wiki_n", 0),
            "wiki_raw_pct": cd.get("wiki_raw_pct"),
        }
        ex_lo = cd["ex_lo"]
        ex_hi = cd["ex_hi"]
    else:
        u_div_bal, r_div_bal = u_foc_bal, r_foc_bal
        u_div_win, r_div_win = u_foc_win, r_foc_win
        u_div_bal_all, r_div_bal_all = u_foc_bal, r_foc_bal
        u_div_win_all, r_div_win_all = u_foc_win, r_foc_win
        v_div_win = F_win[:, col_i]
        title = short_words(t_words, 2).title()
        question = f"How often MPs focus on {short_words(t_words, 3)}"
        pole_lo = f"Rarely mentions {t_words[0]}"
        pole_hi = f"Champions {short_words(t_words, 2)}"
        kind = "focus_only"
        # Build pts for top MPs talking about this topic
        p_col = P_counts[tk] if tk in P_counts.columns else pd.Series(0, index=all_mp_ids)
        pts = []
        for m in all_mp_ids:
            nsp = int(p_col.get(m, 0))
            if nsp >= 2:
                i = mp_idx[m]
                sc = round(float(np.clip(F_bal[i, col_i] * 0.45, -1.1, 1.1)), 3)
                pts.append([i, sc, 1 if sc > 0.2 else (-1 if sc < -0.2 else 0), nsp, 0])
        quotes = {}
        pts_all = pts
        quotes_all = {}
        n_lo_all = 0
        n_hi_all = len(pts)
        stats = {
            "n_lo": 0,
            "n_hi": len(pts),
            "n_neu": 0,
            "split_balance": 0.0,
            "party_r2": 0.05,
            "bb_party_r2": 0.05,
            "jev_r_all": None,
            "jev_r_lab": None,
            "vote_auc": None,
            "vote_auc_lab": None,
        }
        ex_lo, ex_hi = [], []

    # Compute Within-Party correlations for both Stance/Divide and Focus
    div_corrs, foc_corrs = [], []
    v_foc_win = F_win[:, col_i]
    for col_2, tk2 in enumerate(ordered_ks):
        if col_2 == col_i:
            continue
        t2_words = [str(w) for w in top_terms[tk2][:4]]
        nm2 = claim_data[str(tk2)]["title"] if str(tk2) in claim_data else short_words(t2_words, 2).title()

        # Correlate with other topic's Claim-Stance if available
        if str(tk2) in claim_data:
            c2 = claim_ks.index(tk2)
            cd2 = claim_data[str(tk2)]
            rs_d, r_d, nd = shrunk_corr(v_div_win, S_win[:, c2], k_shrink=18)
            if nd >= 14:
                pole2 = cd2["pole_hi"] if r_d > 0 else cd2["pole_lo"]
                div_corrs.append({"k": col_2, "kind": "stance", "r": round(r_d, 2), "n": nd, "name": nm2, "pole": pole2, "s": abs(rs_d)})
            rs_fd, r_fd, nfd = shrunk_corr(v_foc_win, S_win[:, c2], k_shrink=18)
            if nfd >= 18:
                pole2 = cd2["pole_hi"] if r_fd > 0 else cd2["pole_lo"]
                foc_corrs.append({"k": col_2, "kind": "stance", "r": round(r_fd, 2), "n": nfd, "name": nm2, "pole": pole2, "s": abs(rs_fd)})

        # Correlate with other topic's Focus
        rs_df, r_df, ndf = shrunk_corr(v_div_win, F_win[:, col_2], k_shrink=25)
        if ndf >= 20:
            div_corrs.append({"k": col_2, "kind": "focus", "r": round(r_df, 2), "n": ndf, "name": nm2, "pole": f"Talks heavily about {nm2}" if r_df > 0 else f"Rarely mentions {nm2}", "s": abs(rs_df)})

        rs_ff, r_ff, nff = shrunk_corr(v_foc_win, F_win[:, col_2], k_shrink=25)
        if nff >= 25:
            foc_corrs.append({"k": col_2, "kind": "focus", "r": round(r_ff, 2), "n": nff, "name": nm2, "pole": f"Talks heavily about {nm2}" if r_ff > 0 else f"Rarely mentions {nm2}", "s": abs(rs_ff)})

    div_corrs.sort(key=lambda x: -x["s"])
    foc_corrs.sort(key=lambda x: -x["s"])
    for lst in (div_corrs, foc_corrs):
        for item in lst:
            item.pop("s", None)

    topics_out.append({
        "k": col_i,
        "tk": tk,
        "has_claim": has_claim,
        "title": title,
        "words": t_words[:6],
        "kind": kind,
        "question": question,
        "pole_lo": pole_lo,
        "pole_hi": pole_hi,
        "stats": stats,
        "ex_lo": ex_lo,
        "ex_hi": ex_hi,
        "diam_bal": {"div": u_div_bal, "r_div": r_div_bal, "foc": u_foc_bal, "r_foc": r_foc_bal},
        "diam_win": {"div": u_div_win, "r_div": r_div_win, "foc": u_foc_win, "r_foc": r_foc_win},
        "diam_bal_all": {"div": u_div_bal_all, "r_div": r_div_bal_all, "foc": u_foc_bal, "r_foc": r_foc_bal},
        "diam_win_all": {"div": u_div_win_all, "r_div": r_div_win_all, "foc": u_foc_win, "r_foc": r_foc_win},
        "pts": pts,
        "quotes": quotes,
        "pts_all": pts_all,
        "quotes_all": quotes_all,
        "n_lo_all": n_lo_all,
        "n_hi_all": n_hi_all,
        "div_corrs": div_corrs[:8],
        "foc_corrs": foc_corrs[:8],
    })

mps_out = []
for i, mid in enumerate(all_mp_ids):
    m = members.loc[mid]
    st_list = []
    st_list_all = []
    for c_idx in range(N_ST):
        sc = S_stance[i, c_idx]
        if not np.isnan(sc):
            st_list.append([c_idx, round(float(sc), 2)])
        sc_a = S_stance_all[i, c_idx]
        if not np.isnan(sc_a):
            st_list_all.append([c_idx, round(float(sc_a), 2)])
    mps_out.append({
        "i": i,
        "id": int(mid),
        "n": m["name"],
        "p": m["party_abbr"] or "Ind",
        "pn": m["party"],
        "c": "#" + m["party_colour"],
        "k": m["constituency"],
        "b": [round(float(v), 3) for v in Z3_bal[i]],
        "w": [round(float(v), 3) for v in Z3_win[i]],
        "b_all": [round(float(v), 3) for v in Z3_bal_all[i]],
        "w_all": [round(float(v), 3) for v in Z3_win_all[i]],
        "f": [round(float(v), 2) for v in F_focus[i]],
        "st": st_list,
        "st_all": st_list_all,
    })

party_stats = []
for pa in sorted(major_parties, key=lambda p: -(parties == p).sum()):
    mask = parties == pa
    party_stats.append({
        "p": pa,
        "n": int(mask.sum()),
        "c": "#" + members.loc[all_mp_ids[np.where(mask)[0][0]], "party_colour"],
        "b": [round(float(v), 3) for v in Z3_bal[mask].mean(0)],
        "w": [round(float(v), 3) for v in Z3_win[mask].mean(0)],
        "b_all": [round(float(v), 3) for v in Z3_bal_all[mask].mean(0)],
        "w_all": [round(float(v), 3) for v in Z3_win_all[mask].mean(0)],
    })


# Presentation-only thumbnails: read cached portraits without changing source data.
# Embed these in the final bundle so local previews and static docs work offline.
import base64
import io
from PIL import Image, ImageOps

for mp in mps_out:
    photo_path = ROOT / "data/raw/photos" / f"{mp['id']}.jpg"
    if photo_path.exists():
        try:
            with Image.open(photo_path) as photo:
                thumb = ImageOps.fit(photo.convert("RGB"), (96, 112), centering=(0.5, 0.3))
                photo_bytes = io.BytesIO()
                thumb.save(photo_bytes, format="JPEG", quality=75, optimize=True)
                mp["photo"] = "data:image/jpeg;base64," + base64.b64encode(photo_bytes.getvalue()).decode("ascii")
        except (OSError, ValueError):
            pass  # The UI falls back to the Parliament portrait URL.

# Presentation review only: retain all original scores, axes and map topics.
opening_review = json.loads((HERE / "opening_question_review.json").read_text())
for topic in topics_out:
    if topic["title"] in opening_review:
        topic["quiz_review"] = opening_review[topic["title"]]

evidence_dir = HERE / "research_2026_11_10" / "results"
evidence_path = next(
    (evidence_dir / name for name in ("fp2_mp_evidence.json", "fp_mp_evidence.json")
     if (evidence_dir / name).is_file()),
    None,
)
debate_evidence = json.loads(evidence_path.read_text()) if evidence_path else {}
if evidence_path:
    # Embed only sources reachable from accepted positions, never dropped claims.
    used_passages = {c["p"] for f in debate_evidence["fights"]
                     for side in ("for", "against", "both")
                     for m in f[side] for c in m["claims"]}
    missing = used_passages - debate_evidence["P"].keys()
    if missing:
        raise ValueError(f"Debate evidence is missing {len(missing)} source passages")
    debate_evidence["P"] = {p: debate_evidence["P"][p] for p in sorted(used_passages)}
    dates = sorted(p["date"] for p in debate_evidence["P"].values() if p.get("date"))
    debate_evidence["period"] = f"{dates[0]} to {dates[-1]}" if dates else ""
    debate_evidence["source_file"] = evidence_path.name
    print(f"Debate positions from {evidence_path.name}")

bundle = {"mps": mps_out, "parties": party_stats, "topics": topics_out,
          "debate_evidence": debate_evidence, "default_mode": "topics"}
tpl = (HERE / "unified_template.html").read_text()
for marker, module in (
    ("__TOPIC_SELECTION__", "topic_selection.js"),
    ("__VISITOR_PLACEMENT__", "visitor_placement.js"),
    ("__TOPIC_RELATIONSHIPS__", "topic_relationships.js"),
    ("__MAP_EVIDENCE__", "map_evidence.js"),
):
    tpl = tpl.replace(marker, (HERE / module).read_text())
page_unified = tpl.replace("__DATA__", json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
(HERE / "unified.html").write_text(page_unified)
print(f"Wrote prototypes/unified.html ({len(page_unified.encode()) // 1024} KB)")

# Also build the production visitor app (prototypes/app.html, docs/index.html, docs/stance.html) with default_mode="quiz"
bundle_app = {**bundle, "default_mode": "quiz"}
page_app = tpl.replace("__DATA__", json.dumps(bundle_app, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
for out_path in [HERE / "app.html", ROOT / "docs/index.html", ROOT / "docs/stance.html"]:
    out_path.write_text(page_app)
print(f"Wrote prototypes/app.html, docs/index.html, docs/stance.html ({len(page_app.encode()) // 1024} KB)")

# Build the streamlined 3-tab prototypes/lab.html with a lightweight validation summary bundle
lab_topics = []
for t in topics_out:
    if not t["has_claim"]:
        continue
    shift_quotes = {k: {"shift": q["shift"]} for k, q in t["quotes"].items() if "shift" in q}
    lab_topics.append({
        "k": t["k"],
        "has_claim": True,
        "title": t["title"],
        "pole_lo": t["pole_lo"],
        "pole_hi": t["pole_hi"],
        "stats": t["stats"],
        "quotes": shift_quotes,
    })
bundle_lab = {"mps": [{"n": m["n"], "p": m["p"]} for m in mps_out], "topics": lab_topics}
lab_tpl = (HERE / "lab_template.html").read_text()
page_lab = lab_tpl.replace("__DATA__", json.dumps(bundle_lab, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
(HERE / "lab.html").write_text(page_lab)
print(f"Wrote prototypes/lab.html ({len(page_lab.encode()) // 1024} KB)")

