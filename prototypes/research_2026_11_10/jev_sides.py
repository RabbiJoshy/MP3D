"""Jev double-check of sides on contested measures: state = passage; question = does the speaker want <group label>?"""
import re, sys, json, pathlib, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
tag = sys.argv[1]
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
key = next(m[1].strip().strip("\"'") for l in (ROOT / ".env").read_text().splitlines() if (m := re.match(r"\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)", l)))
S = requests.Session(); S.headers.update({"Authorization": f"Bearer {key}"}); used = [0]
c = pd.read_json(f"{tag}_measures.jsonl", lines=True); con = pd.read_json(f"{tag}_contested.json")
c = c[c.g.isin(con.index)].copy()
def ask(r):
    q = {"want": {"type": "noul", "instructions": f"Does the speaker want this to happen: \"{r.label}\"? Judge what the speaker argues, not what they describe; criticising it means no."},
         "rel": {"type": "noul", "instructions": f"Does the speaker express a view, for or against, on: \"{r.label}\"?"}}
    for _ in range(6):
        try:
            x = S.post("https://api.typesafe.ai/v1/systemone", json={"model": "jev-1.14.0", "state": r.text[:3000], "questions": q}, timeout=60)
            if x.status_code == 200:
                d = x.json(); used[0] += d["usage"]["input_tokens"]; return d["answers"]["want"]["noul"], d["answers"]["rel"]["noul"]
        except requests.RequestException: pass
    return None, None
with ThreadPoolExecutor(16) as ex: res = list(ex.map(ask, c.itertuples()))
c["p_want"] = [a for a, _ in res]; c["p_rel"] = [b for _, b in res]
c["jev"] = ["?" if a is None else ("OFF" if b < .5 else "FOR" if a >= .5 else "AGAINST") for a, b in res]
on = c[c.jev.isin(["FOR", "AGAINST"])]
agree = (on.jev == on.stance).mean()
print(f"{len(c)} contested claims; Jev says off-topic for {(c.jev=='OFF').mean():.0%}; on the rest, Jev agrees with the extractor {agree:.0%} ({len(on)} claims)")
print(f"Jev tokens {used[0]}  ~${used[0]*0.042/1e6:.4f}")
c.to_json(f"{tag}_jev_sides.jsonl", orient="records", lines=True)
for lab, g in on.groupby("label"):
    print(f"  {lab[:70]:70}  agree {(g.jev==g.stance).mean():.0%} of {len(g)}")
