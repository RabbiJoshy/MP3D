import re, json, pathlib, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
key = next(m[1].strip().strip("\"'") for l in (ROOT / ".env").read_text().splitlines() if (m := re.match(r"\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)", l)))
S = requests.Session(); S.headers.update({"Authorization": f"Bearer {key}"}); used = [0]
HERE = ROOT / "prototypes/research_2026_11_10"; D = ROOT / "data/quick/research_2026_11_10"
test = json.load(open(HERE / "results/edu_jev2_test.json"))
PROMPT_FOR, PROMPT_AGAINST = test["prompt_for"], test["prompt_against"]
def ask(r):
    q = {"for": {"type": "noul", "instructions": PROMPT_FOR.format(m=r["label"])},
         "against": {"type": "noul", "instructions": PROMPT_AGAINST.format(m=r["label"])}}
    for _ in range(6):
        try:
            x = S.post("https://api.typesafe.ai/v1/systemone", json={"model": "jev-1.14.0", "state": r["text"][:3000], "questions": q}, timeout=60)
            if x.status_code == 200:
                d = x.json(); used[0] += d["usage"]["input_tokens"]; return d["answers"]["for"]["noul"], d["answers"]["against"]["noul"]
        except requests.RequestException: pass
    return None, None
def verdict(f, a):
    if f is None: return "ERR"
    if f >= .5 and f > a: return "FOR"
    if a >= .5 and a > f: return "AGAINST"
    return "NEITHER"
if __name__ == "__main__":
    c = pd.read_json(D / "fp/dense2_cands.jsonl", lines=True)
    j = pd.read_json(D / "fp/jev_sides.jsonl", lines=True)[["passage_id", "label", "p_for", "p_against", "new_verdict"]]
    d1 = pd.read_json(D / "fp/dense_sides.jsonl", lines=True).rename(columns={"verdict": "new_verdict"})[["passage_id", "label", "p_for", "p_against", "new_verdict"]]
    j = pd.concat([j, d1]).dropna(subset=["new_verdict"]).drop_duplicates(["passage_id", "label"])
    m = c.merge(j, on=["passage_id", "label"], how="left")
    done = m[m.new_verdict.notna()].copy(); done["reused"] = True
    todo = m[m.new_verdict.isna()].drop(columns=["p_for", "p_against", "new_verdict"])
    p = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "text"])
    todo = todo.merge(p, on="passage_id")
    est = (todo.text.str.len().clip(upper=3000).sum() / 4 * 2 + len(todo) * 240) * 0.042 / 1e6
    print("reused", len(done), "to judge", len(todo), "est $", round(est, 4)); assert est < 0.28
    rows = todo.to_dict("records")
    with ThreadPoolExecutor(8) as ex: res = list(ex.map(ask, rows))
    for r, (f, a) in zip(rows, res): r["p_for"], r["p_against"], r["new_verdict"] = f, a, verdict(f, a)
    new = pd.DataFrame(rows).drop(columns=["text"]); new["reused"] = False
    out = pd.concat([done, new]).rename(columns={"new_verdict": "verdict"})
    out.to_json(D / "fp/dense_sides2.jsonl", orient="records", lines=True)
    print("tokens", used[0], "cost $", used[0] * 0.042 / 1e6, out.verdict.value_counts().to_dict())
    json.dump({"tokens": used[0]}, open(HERE / "results/fp_dense2_tokens.json", "w"))
