"""Zero-LLM-cost benchmark of the live 3D speech space against within-party votes & rebellions."""
import json
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


def load_app_data(path="docs/index.html"):
    html = open(path, encoding="utf-8").read()
    m = re.search(r'<script id="d" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(m.group(1))


def cv_auc(X, y, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    clf = LogisticRegression(C=1.0, max_iter=1000)
    probs = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, probs))


def main():
    D = load_app_data()
    mps = D["mps"]
    xyz = np.asarray(D["xyz"], dtype=float)
    onmap = np.asarray(D["onmap"], dtype=bool)
    nsub = np.array([sum(1 for s in D["subjects"] if s["sc"][i] is not None) for i in range(len(mps))])
    id_to_idx = {m["id"]: i for i, m in enumerate(mps)}

    votes = pd.read_parquet("data/clean/votes_long.parquet")
    votes = votes[votes["member_id"].isin(id_to_idx)].copy()
    votes["idx"] = votes["member_id"].map(id_to_idx)

    print(f"Live app: {len(mps)} placed MPs ({onmap.sum()} onmap, {(nsub >= 3).sum()} with >=3 subjects)")
    print(f"Axes: {D['axes']}\n")

    # 1. Within-party split / free-vote divisions (>= 15 Aye and >= 15 No in same party)
    print("=== 1. WITHIN-PARTY SPLIT & FREE-VOTE DIVISIONS (>=15 on each side in-party) ===")
    split_rows = []
    for (div_id, party), grp in votes.groupby(["division_id", "party"]):
        n_aye = int((grp["vote"] == 1).sum())
        n_no = int((grp["vote"] == -1).sum())
        if min(n_aye, n_no) >= 15:
            idx = grp["idx"].to_numpy()
            y = (grp["vote"].to_numpy() == 1).astype(int)
            auc_all = cv_auc(xyz[idx], y)
            mask3 = nsub[idx] >= 3
            auc_3p = cv_auc(xyz[idx[mask3]], y[mask3]) if min(int(y[mask3].sum()), int((1 - y[mask3]).sum())) >= 10 else None
            split_rows.append({
                "div_id": int(div_id),
                "date": grp["date"].iloc[0],
                "party": party,
                "title": grp["title"].iloc[0],
                "n_aye": n_aye,
                "n_no": n_no,
                "auc_all": auc_all,
                "auc_3plus": auc_3p,
                "n_3plus": int(mask3.sum()),
            })

    split_df = pd.DataFrame(split_rows).sort_values("auc_all", ascending=False)
    for _, r in split_df.iterrows():
        s3 = f"{r['auc_3plus']:.3f} (n={r['n_3plus']})" if pd.notna(r["auc_3plus"]) else "n/a"
        print(f"  [{r['party'][:3]}] AUC={r['auc_all']:.3f} | >=3subj={s3:>14} | {r['n_aye']:>3}Y/{r['n_no']:<3}N | {r['date']} | {r['title'][:72]}")
    if len(split_df):
        print(f"  Mean across {len(split_df)} split divisions: AUC_all={split_df['auc_all'].mean():.3f}, "
              f"AUC_3plus={split_df['auc_3plus'].dropna().mean():.3f}\n")

    # 2. Within-party rebellion propensity (total rebels per MP)
    print("=== 2. WITHIN-PARTY REBELLION RATE (frequent rebels >=3 vs loyalists) ===")
    for party in ["Labour", "Conservative", "Liberal Democrat"]:
        pv = votes[votes["party"] == party]
        if pv.empty:
            continue
        mp_reb = pv.groupby("idx").agg(n_votes=("rebel", "size"), n_reb=("rebel", "sum")).reset_index()
        mp_reb["rate"] = mp_reb["n_reb"] / mp_reb["n_votes"]
        idx = mp_reb["idx"].to_numpy()
        y_reb = (mp_reb["n_reb"].to_numpy() >= 3).astype(int)
        if min(int(y_reb.sum()), int((1 - y_reb).sum())) < 10:
            print(f"  {party}: too few >=3-time rebels ({int(y_reb.sum())}/{len(y_reb)})")
            continue
        auc_all = cv_auc(xyz[idx], y_reb)
        mask3 = nsub[idx] >= 3
        auc_3p = cv_auc(xyz[idx[mask3]], y_reb[mask3]) if min(int(y_reb[mask3].sum()), int((1 - y_reb[mask3]).sum())) >= 10 else float("nan")
        ax_corrs = [spearmanr(xyz[idx, k], mp_reb["rate"])[0] for k in range(3)]
        print(f"  {party:16s} (n={len(idx)}, rebels>=3: {int(y_reb.sum())}): "
              f"CV AUC={auc_all:.3f} | >=3subj AUC={auc_3p:.3f} (n={int(mask3.sum())}) | "
              f"axis spearman r={[round(float(c), 3) for c in ax_corrs]}")

    # 3. All smaller within-party rebellions (5..14 rebels)
    print("\n=== 3. ALL MODERATE WITHIN-PARTY REBELLIONS (5-14 minority votes) ===")
    mod_aucs, mod_aucs_3p = [], []
    for (div_id, party), grp in votes.groupby(["division_id", "party"]):
        n_aye = int((grp["vote"] == 1).sum())
        n_no = int((grp["vote"] == -1).sum())
        if 5 <= min(n_aye, n_no) < 15:
            idx = grp["idx"].to_numpy()
            y = (grp["vote"].to_numpy() == 1).astype(int)
            mod_aucs.append(cv_auc(xyz[idx], y, folds=5))
            mask3 = nsub[idx] >= 3
            if min(int(y[mask3].sum()), int((1 - y[mask3]).sum())) >= 5:
                mod_aucs_3p.append(cv_auc(xyz[idx[mask3]], y[mask3], folds=5))
    print(f"  {len(mod_aucs)} divisions with 5-14 minority votes: mean AUC_all={np.mean(mod_aucs):.3f} | "
          f"mean AUC_3plus={np.mean(mod_aucs_3p):.3f} (n_divs={len(mod_aucs_3p)})")


if __name__ == "__main__":
    main()
