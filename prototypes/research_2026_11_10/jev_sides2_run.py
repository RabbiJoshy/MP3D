"""Run jev_sides2's identical prompts + decision rule over all claims in originally-contested edu groups."""
import re, json, pathlib, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
key = next(m[1].strip().strip("\"'") for l in (ROOT / ".env").read_text().splitlines() if (m := re.match(r"\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)", l)))
S = requests.Session(); S.headers.update({"Authorization": f"Bearer {key}"}); used = [0]
HERE = ROOT / "prototypes/research_2026_11_10"; D = ROOT / "data/quick/research_2026_11_10"
test = json.load(open(HERE / "results/edu_jev2_test.json"))
PROMPT_FOR, PROMPT_AGAINST = test["prompt_for"], test["prompt_against"]  # identical saved prompts
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
    import sys
    m = pd.read_json(D / "edu_measures.jsonl", lines=True)
    con = {int(k) for k in json.load(open(HERE / "results/edu_contested.json"))["label"]}
    m = m[m.g.isin(con)].reset_index(drop=True)
    est = sum(min(len(t), 3000) for t in m.text) / 4 * 2 + len(m) * 2 * 120
    print("claims", len(m), "groups", m.g.nunique(), "est tokens", int(est), "est cost $", est * 0.042 / 1e6)
    assert est * 0.042 / 1e6 < 0.10
    if "--dry" in sys.argv: sys.exit()
    rows = m.to_dict("records")
    with ThreadPoolExecutor(8) as ex: res = list(ex.map(ask, rows))
    for r, (f, a) in zip(rows, res): r["p_for"], r["p_against"], r["new_verdict"] = f, a, verdict(f, a)
    pd.DataFrame(rows).to_json(D / "edu_jev2_sides.jsonl", orient="records", lines=True)
    print("tokens", used[0], "cost $", used[0] * 0.042 / 1e6)
    json.dump({"tokens": used[0]}, open(HERE / "results/_jev2_full_tokens.json", "w"))
