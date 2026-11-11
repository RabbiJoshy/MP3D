"""Wrapper: strict judge (identical prompts/rule via jev_sides2_run) over claims in edu2 contested groups; reuse old verdicts only where label unchanged."""
import sys, json, pandas as pd
sys.path.insert(0, "."); 
from jev_sides2_run import ask, verdict, used, D, HERE
from concurrent.futures import ThreadPoolExecutor
m = pd.read_json(D / "edu2_measures.jsonl", lines=True); con = pd.read_json(D / "edu2_contested.json").index
m = m[m.g.isin(con)].reset_index(drop=True)
o = pd.read_json(D / "edu_jev2_sides.jsonl", lines=True)[["passage_id", "label", "p_for", "p_against", "new_verdict"]].rename(columns={"label": "old_label"})
m = m.merge(o, on="passage_id", how="left")
m["reused"] = m.label == m.old_label
todo = m[~m.reused]
rows = todo.to_dict("records")
with ThreadPoolExecutor(8) as ex: res = list(ex.map(ask, rows))
for r, (f, a) in zip(rows, res): r["p_for"], r["p_against"], r["new_verdict"] = f, a, verdict(f, a)
out = pd.concat([m[m.reused], pd.DataFrame(rows)]).drop(columns=["old_label"])
out.to_json(D / "edu2_jev2_sides.jsonl", orient="records", lines=True)
print("judged", len(rows), "reused", int(m.reused.sum()), "tokens", used[0], "cost $", used[0] * 0.042 / 1e6)
json.dump({"tokens": used[0]}, open(HERE / "results/edu2_jev_tokens.json", "w"))
