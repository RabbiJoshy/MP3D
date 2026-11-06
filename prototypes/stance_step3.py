"""Step 3: depth on the contested issues from step 2.

Built on step 2's setup: data-made phrases, short snippets.

Candidates: the 100 most contested in the earlier screen, which is already paid for. For each, the phrase is the
target word plus the content word most often within five words of it, written as MPs most often write it. Snippets are
the sentences mentioning both, plus one either side. Jev reads up to 10, stopping after 6 if they all fall on one side.
The only human input is the wording of the two general questions and these thresholds.
"""
import collections
import json
import pathlib
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
N_CANDIDATES, MAX_SNIPPETS, EARLY_STOP_AT, MIN_DEBATES_BEFORE_STOP, WORKERS, SEED = 100, 12, 8, 6, 8, 7
WORD = re.compile(r"[a-zA-Z][a-zA-Z\-']*")
SPLIT = re.compile(r"(?<=[.?!])\s+(?=[\"“‘A-Z(])")
OUT = ROOT / "data/quick/step3_answers.jsonl"

screen = pd.DataFrame(json.loads(l) for l in (ROOT / "data/quick/contest_answers.jsonl").read_text().splitlines() if l.strip())
cand_meta = pq.read_table(ROOT / "data/quick/contest_candidates.parquet", columns=["target", "pid", "member"]).to_pandas().drop_duplicates(["pid", "target"])
sc = screen.merge(cand_meta, on=["pid", "target"])
sc["side"] = np.where(sc.support >= 0.5, 1, np.where(sc.oppose >= 0.5, -1, 0))
g = sc[sc.side != 0].groupby("target")
stats = pd.DataFrame({"sup": g.apply(lambda x: x[x.side == 1].member.nunique()), "opp": g.apply(lambda x: x[x.side == -1].member.nunique())}).fillna(0)
stats["contest"] = stats.min(1) / stats.sum(1).clip(1)
chosen = stats[(stats.sup >= 2) & (stats.opp >= 2)].sort_values(["contest", "sup"], ascending=False).head(N_CANDIDATES).index.tolist()
print(f"chosen {len(chosen)} candidates from {len(stats)} screened", flush=True)

cand = pq.read_table(ROOT / "data/quick/contest_candidates.parquet", columns=["target", "pid", "member", "debate", "text"], filters=[("target", "in", chosen)]).to_pandas().drop_duplicates(["pid", "target"])
freq = collections.Counter(w.lower() for t in cand.text.sample(min(20000, len(cand)), random_state=SEED) for w in WORD.findall(t))
common = {w for w, _ in freq.most_common(250)}


def sense(word, texts):
    near, spans = collections.Counter(), collections.Counter()
    word_toks = word.split()
    first = word_toks[0]
    tok_lists = [[w.lower() for w in WORD.findall(t)] for t in texts]
    for toks in tok_lists:
        for i, w in enumerate(toks):
            if w == first:
                for j in range(max(0, i - 5), min(len(toks), i + 6)):
                    if j != i and toks[j] not in common and toks[j] not in word_toks and len(toks[j]) > 2:
                        near[toks[j]] += 1
    if not near:
        return None, word
    partner = near.most_common(1)[0][0]
    for toks in tok_lists:
        for i, w in enumerate(toks):
            if w != first:
                continue
            for j in range(max(0, i - 5), min(len(toks), i + 6)):
                if toks[j] == partner:
                    a, b = min(i, j), max(i, j)
                    end = b + 1
                    if b == j and end < len(toks) and toks[end] not in common:
                        end += 1
                    spans[" ".join(toks[a:end])] += 1
    return partner, spans.most_common(1)[0][0] if spans else word


def snippet(text, a, b):
    sents = SPLIT.split(text)
    hit = [i for i, s in enumerate(sents) if re.search(r"\b" + re.escape(a) + r"\b", s, re.I) and re.search(r"\b" + re.escape(b) + r"\b", s, re.I)]
    if not hit:
        return None
    keep = sorted({j for i in hit for j in (i - 1, i, i + 1) if 0 <= j < len(sents)})
    return " ".join(sents[j] for j in keep)



PER_MP, CAP = 2, 120
s2 = pd.read_json(ROOT / "data/quick/step2b_summary.json")
s2["none_share"] = s2.no_position / s2.read
s2["content_words"] = [sum(w not in common for w in WORD.findall(p.lower())) for p in s2.phrase]
keep = s2[s2.contested & (s2.none_share <= 0.6) & (s2.content_words >= 2)]
print("dropped as noise:", s2[s2.contested & ~s2.phrase.isin(keep.phrase)].phrase.tolist())
print("kept:", keep.phrase.tolist(), flush=True)
a2 = pd.DataFrame(json.loads(l) for l in (ROOT / "data/quick/step2b_answers.jsonl").read_text().splitlines())
phrase_target = a2.drop_duplicates("phrase").set_index("phrase").target
plans = {}
for ph in keep.phrase:
    t = phrase_target[ph]
    sub = cand[cand.target == t]
    partner, phrase = sense(t, sub.text.tolist())
    sub = sub.assign(snip=[snippet(x, t.split()[0], partner) for x in sub.text]).dropna(subset=["snip"])
    sub = sub.sample(frac=1, random_state=SEED).drop_duplicates(["member", "debate"]).groupby("member").head(PER_MP)
    # Spread across debates first
    sub = sub.assign(r=sub.groupby("debate").cumcount()).sort_values("r").head(CAP)
    plans[ph] = {"target": t, "phrase": phrase, "rows": sub}
    print(f"  {phrase!r}: {len(sub)} snippets from {sub.member.nunique()} MPs in {sub.debate.nunique()} debates", flush=True)
key = next(m[1].strip().strip("\"'") for m in (re.match(r"\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)", l) for l in (ROOT / ".env").read_text().splitlines()) if m)
session = requests.Session()
session.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
used = {"tokens": 0, "requests": 0, "failed": 0}


def ask(job):
    ph, r = job
    phrase = plans[ph]["phrase"]
    qs = {"support": {"type": "noul", "instructions": f"Does the speaker themselves express support for {phrase}?"},
          "oppose": {"type": "noul", "instructions": f"Does the speaker themselves express opposition to {phrase}?"}}
    for attempt in range(6):
        try:
            resp = session.post("https://api.typesafe.ai/v1/systemone", json={"model": "jev-1.14.0", "state": r.snip, "questions": qs}, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt); continue
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(float(resp.headers.get("retry-after") or 2 ** attempt)); continue
        resp.raise_for_status()
        j = resp.json()
        used["tokens"] += j.get("usage", {}).get("input_tokens", 0)
        used["requests"] += 1
        a = j["answers"]
        side = 1 if a["support"]["noul"] >= 0.5 else (-1 if a["oppose"]["noul"] >= 0.5 else 0)
        return {"phrase": phrase, "pid": r.pid, "member": int(r.member), "debate": r.debate, "side": side, "support": a["support"]["noul"], "oppose": a["oppose"]["noul"], "snip": r.snip}
    used["failed"] += 1
    return None


jobs = [(ph, r) for ph, p in plans.items() for r in p["rows"].itertuples()]
start = time.time()
with ThreadPoolExecutor(WORKERS) as ex, OUT.open("w") as f:
    for row in ex.map(ask, jobs):
        if row:
            f.write(json.dumps(row) + "\n")
print(f"\n{used['requests']} requests in {time.time() - start:.0f}s | tokens {used['tokens']} | cost ${used['tokens'] * 0.042 / 1e6:.4f} | failed {used['failed']}")
