"""Generic: Jev picks the most on-topic candidate per issue (10-claim probe), then labels every claim (summary as state)."""
import re, sys, json, pathlib, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
tag = "all"
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
key = next(m[1].strip().strip("\"'") for l in (ROOT / ".env").read_text().splitlines() if (m := re.match(r"\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)", l)))
S = requests.Session(); S.headers.update({"Authorization": f"Bearer {key}"})
c = pd.read_json("all_issues.jsonl", lines=True); cands = json.load(open("all_cands.json"))
used = [0]
def jev(pos, prop):
    qs = {"agree": {"type": "noul", "instructions": f"Would the speaker agree with this statement: \"{prop}\"? Judge what the speaker argues; criticising a policy means disagreeing with it."},
          "relevant": {"type": "noul", "instructions": f"Does the speaker take any position, for or against, on this statement: \"{prop}\"?"}}
    for _ in range(6):
        try:
            x = S.post("https://api.typesafe.ai/v1/systemone", json={"model": "jev-1.14.0", "state": "The speaker: " + pos, "questions": qs}, timeout=60)
            if x.status_code == 200:
                d = x.json(); used[0] += d["usage"]["input_tokens"]; return d["answers"]["agree"]["noul"], d["answers"]["relevant"]["noul"]
        except requests.RequestException: pass
    raise RuntimeError("jev failed")
chosen = {}
with ThreadPoolExecutor(16) as ex:
    for iss, g in c.groupby("issue"):
        probe = g.sample(min(5, len(g)), random_state=1)
        best = max(cands[iss] or ["(none)"], key=lambda p: sum(r >= .5 for _, r in ex.map(lambda pos: jev(pos, p), probe.position)))
        chosen[iss] = best
    c["proposition"] = c.issue.map(chosen)
    res = list(ex.map(lambda r: jev(r.position, r.proposition), c.itertuples()))
c["p_agree"] = [a for a, _ in res]; c["p_rel"] = [r for _, r in res]
c["jev"] = ["UNRELATED" if r < .5 else "AGREE" if a >= .5 else "DISAGREE" for a, r in res]
c.to_json("all_jev.jsonl", orient="records", lines=True)
print(c.jev.value_counts().to_dict())
print(f"tokens {used[0]}  ~${used[0]*0.042/1e6:.4f}")
