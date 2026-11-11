import sys, json, pandas as pd
sys.path.insert(0, ".")
import fp_dense2_run as R2
from concurrent.futures import ThreadPoolExecutor
ROOT, D = R2.ROOT, R2.D
c = pd.read_json(D / "fp/dense3_cands.jsonl", lines=True); c = c[c.unit.isin([0, 27])]
c = c.merge(pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "text"]), on="passage_id")
rows = c.to_dict("records")
with ThreadPoolExecutor(8) as ex: res = list(ex.map(R2.ask, rows))
for r, (f, a) in zip(rows, res): r["p_for"], r["p_against"], r["verdict"] = f, a, R2.verdict(f, a); r.pop("text")
old = pd.read_json(D / "fp/dense_sides3.jsonl", lines=True)
pd.concat([old, pd.DataFrame(rows)]).to_json(D / "fp/dense_sides3.jsonl", orient="records", lines=True)
jp = ROOT / "prototypes/research_2026_11_10/results/fp_clean_judge.json"; j = json.load(open(jp))
u2 = R2.used[0] * 0.042 / 1e6
j.update(tokens=j["tokens"] + R2.used[0], usd=j["usd"] + u2, done=j["done"] + [0, 27], skipped=[], passages=j["passages"] + len(rows)); json.dump(j, open(jp, "w"))
print(len(rows), u2, j["usd"], pd.DataFrame(rows).verdict.value_counts().to_dict())
