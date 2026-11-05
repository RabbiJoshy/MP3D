"""Score asylum passages for stance with TypeSafe's Jev. Resumable; writes data/quick/stance_asylum_jev.jsonl.

    python prototypes/stance_jev.py MODEL [limit]

Jev sees only the passage, no MP name, party or date. The scale is asked forwards and backwards, because the
docs warn about order effects, and the two are averaged. Reads TYPESAFE_API_KEY from .env.
"""
import json
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data/quick/stance_asylum_jev.jsonl"
URL = "https://api.typesafe.ai/v1/systemone"
ASYLUM_TOPIC, WORKERS = 28, 4

SCALE = [
    "Strongly favours protecting asylum seekers and migrants: safe routes, refugee rights, welcome; opposes tougher rules",
    "Leans towards protecting asylum seekers and migrants",
    "Takes a balanced or mixed position between protection and control",
    "Leans towards tougher immigration control",
    "Strongly favours tougher immigration control: deterrence, removals, lower numbers, stricter rules",
]
INSTRUCT = "What policy direction on asylum and immigration does the speaker themselves argue for? Judge what they argue, not the words they use. A speaker who says deterrents do not work is arguing against deterrence."
QUESTIONS = {
    "relevant": {"type": "noul", "instructions": "Is this passage mainly about asylum, refugees, or immigration?"},
    "position": {"type": "noul", "instructions": "Does the speaker argue for a particular direction in asylum or immigration policy?"},
    "attack": {"type": "noul", "instructions": "Is a main purpose of this passage to criticise the record or competence of a government or political party?"},
    "stance_forward": {"type": "score", "instructions": INSTRUCT, "criteria": SCALE},
    "stance_backward": {"type": "score", "instructions": INSTRUCT, "criteria": SCALE[::-1]},
}


def api_key():
    for line in (ROOT / ".env").read_text().splitlines():
        m = re.match(r"\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)", line)
        if m:
            return m[1].strip().strip("\"'")
    raise SystemExit("TYPESAFE_API_KEY not in .env")


def main(model, limit):
    key = api_key()
    q = np.load(ROOT / "data/quick/topic_space.npz", allow_pickle=True)
    df = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "text", "role"])
    df["topic"] = q["topic"]
    df = df[(df.topic == ASYLUM_TOPIC) & (df.role != "minister")]
    done = {json.loads(l)["id"] for l in OUT.read_text().splitlines() if l.strip()} if OUT.exists() else set()
    todo = df[~df.passage_id.isin(done)]
    if limit:
        todo = todo.sample(min(limit, len(todo)), random_state=7)
    print(f"{len(df)} passages, {len(done)} done, running {len(todo)} on {model}", flush=True)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    used = {"input": 0, "failed": 0, "models": set()}

    def run(row):
        for attempt in range(7):
            try:
                r = session.post(URL, json={"model": model, "state": row.text, "questions": QUESTIONS}, timeout=90)
            except requests.RequestException:
                time.sleep(min(60, 2 ** attempt))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(float(r.headers.get("retry-after") or min(60, 2 ** attempt)))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
            d = r.json()
            a = d["answers"]
            used["input"] += d.get("usage", {}).get("input_tokens", 0)
            used["models"].add(d.get("model"))
            forward, backward = a["stance_forward"]["score"], 4 - a["stance_backward"]["score"]
            takes_position = a["position"]["noul"] >= 0.5
            mean = (forward + backward) / 2 - 2
            return {"id": row.passage_id, "stance": round(mean, 3) if takes_position else None, "stance_any": round(mean, 3),
                    "forward": round(forward - 2, 3), "backward": round(backward - 2, 3), "position_p": a["position"]["noul"],
                    "relevant_p": a["relevant"]["noul"], "attack_p": a["attack"]["noul"], "attack": a["attack"]["noul"] >= 0.5,
                    "conf_forward": a["stance_forward"].get("confidence"), "conf_backward": a["stance_backward"].get("confidence"),
                    "confidence": "high" if min(a["stance_forward"]["confidence"], a["stance_backward"]["confidence"]) >= .5 else "low",
                    "claim": "", "evidence": ""}
        used["failed"] += 1
        return None

    n = 0
    with ThreadPoolExecutor(WORKERS) as ex, OUT.open("a") as f:
        futures = [ex.submit(run, r) for r in todo.itertuples()]
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                f.write(json.dumps(row) + "\n")
                n += 1
            if n % 200 == 0:
                f.flush()
                print(f"{n}/{len(todo)} input tokens {used['input']} failed {used['failed']}", flush=True)
    print("DONE", n, "of", len(todo), "| input tokens", used["input"], "| est cost $%.4f" % (used["input"] * 0.042 / 1e6), "| failed", used["failed"], "| model", used["models"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
