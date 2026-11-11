"""Jev substance filter on 200 random passages (100 current Parliament, 100 previous). Output: substance_test.jsonl"""
import json, pathlib, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests

ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
OUT = pathlib.Path(__file__).with_name("substance_test.jsonl")
URL, MODEL = "https://api.typesafe.ai/v1/systemone", "jev-1.14.0"
QS = {
    "substance": {"type": "noul", "instructions": "Does the speaker argue for or against a policy, or argue what the government or Parliament should do about a policy issue? Answer no for procedure, thanks, tributes, points of order, constituency casework without a policy argument, or questions that only ask for information."},
    "procedure": {"type": "noul", "instructions": "Is this passage mainly parliamentary procedure, chairing, thanks, tributes or courtesies?"},
}

def key():
    for line in (ROOT / ".env").read_text().splitlines():
        m = re.match(r"\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)", line)
        if m:
            return m[1].strip().strip("\"'")
    sys.exit("no key")

cur = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "date", "major", "minor", "role", "text", "words"]).sample(100, random_state=11)
hist = pd.read_parquet(ROOT / "data/quick/passages_hist_2022_2024.parquet").sample(100, random_state=11)
cur["parliament"], hist["parliament"] = "2024-", "2022-24"
df = pd.concat([cur, hist], ignore_index=True)

s = requests.Session()
s.headers.update({"Authorization": f"Bearer {key()}", "Content-Type": "application/json"})

def run(r):
    for a in range(6):
        try:
            resp = s.post(URL, json={"model": MODEL, "state": r.text, "questions": QS}, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** a); continue
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** a); continue
        resp.raise_for_status()
        d = resp.json()["answers"]
        return {"id": r.passage_id, "parliament": r.parliament, "major": r.major, "minor": r.minor, "role": getattr(r, "role", None),
                "words": int(r.words), "text": r.text, "substance": d["substance"]["noul"], "procedure": d["procedure"]["noul"]}
    return None

with ThreadPoolExecutor(8) as ex:
    rows = [x for x in ex.map(run, df.itertuples()) if x]
OUT.write_text("\n".join(json.dumps(x) for x in rows))
r = pd.DataFrame(rows)
print(len(r), "scored")
print(r.groupby("parliament")[["substance", "procedure"]].apply(lambda g: (g >= .5).mean()).round(2))
