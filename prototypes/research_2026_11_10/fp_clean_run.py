import sys, json, pandas as pd
sys.path.insert(0, ".")
import fp_dense2_run as R2
from concurrent.futures import ThreadPoolExecutor
ROOT = R2.ROOT; D = R2.D; CAP_USD = 0.092   # judge budget (12c cap minus ~1.2c already spent, margin)
c = pd.read_json(D / "fp/dense3_cands.jsonl", lines=True)
p = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "text"]); c = c.merge(p, on="passage_id")
order = c.groupby("unit").size().sort_values(ascending=False).index.tolist(); done = []; skipped = []; out = []
for u in order:
    if R2.used[0] * 0.042 / 1e6 > CAP_USD * 0.93: skipped.append(int(u)); continue
    rows = c[c.unit == u].to_dict("records")
    with ThreadPoolExecutor(8) as ex: res = list(ex.map(R2.ask, rows))
    for r, (f, a) in zip(rows, res): r["p_for"], r["p_against"], r["verdict"] = f, a, R2.verdict(f, a); r.pop("text")
    out += rows; done.append(int(u)); print("unit", u, len(rows), "cum usd", round(R2.used[0] * 0.042 / 1e6, 4), flush=True)
pd.DataFrame(out).to_json(D / "fp/dense3_sides3.tmp", orient="records", lines=True)
pd.DataFrame(out).to_json(D / "fp/dense_sides3.jsonl", orient="records", lines=True)
json.dump({"tokens": R2.used[0], "usd": R2.used[0] * 0.042 / 1e6, "done": done, "skipped": skipped, "passages": len(out), "err": int((pd.DataFrame(out).verdict == "ERR").sum())}, open(HERE := str(ROOT / "prototypes/research_2026_11_10/results/fp_clean_judge.json"), "w"))
print("passages", len(out), "usd", R2.used[0] * 0.042 / 1e6, "skipped", skipped, pd.DataFrame(out).verdict.value_counts().to_dict())
