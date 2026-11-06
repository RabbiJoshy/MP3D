"""Step 2: data-made phrases on 100 candidates, short snippets, early stopping.

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
OUT = ROOT / "data/quick/step2b_answers.jsonl"

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


plans = {}
for t in chosen:
    sub = cand[cand.target == t]
    partner, phrase = sense(t, sub.text.tolist())
    if partner is None:
        continue
    sub = sub.assign(snip=[snippet(x, t.split()[0], partner) for x in sub.text]).dropna(subset=["snip"])
    # One snippet per debate first, then per MP, so whoever speaks most on an issue can't fill the sample
    sub = sub.sample(frac=1, random_state=SEED).drop_duplicates("debate").drop_duplicates("member")
    if phrase in {p_["phrase"] for p_ in plans.values()}:
        continue
    plans[t] = {"partner": partner, "phrase": phrase, "rows": sub.head(MAX_SNIPPETS)}
print(f"phrases built for {len(plans)}; examples: {[p['phrase'] for p in list(plans.values())[:12]]}", flush=True)

key = next(m[1].strip().strip("\"'") for m in (re.match(r"\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)", l) for l in (ROOT / ".env").read_text().splitlines()) if m)
session = requests.Session()
session.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
used = {"tokens": 0, "requests": 0, "failed": 0}
lock_out = []


def run_target(t):
    plan = plans[t]
    sides, rows = [], []
    for r in plan["rows"].itertuples():
        qs = {"support": {"type": "noul", "instructions": f"Does the speaker themselves express support for {plan['phrase']}?"},
              "oppose": {"type": "noul", "instructions": f"Does the speaker themselves express opposition to {plan['phrase']}?"}}
        for attempt in range(6):
            try:
                resp = session.post("https://api.typesafe.ai/v1/systemone", json={"model": "jev-1.14.0", "state": r.snip, "questions": qs}, timeout=60)
            except requests.RequestException:
                time.sleep(2 ** attempt); continue
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(float(resp.headers.get("retry-after") or 2 ** attempt)); continue
            resp.raise_for_status()
            break
        else:
            used["failed"] += 1
            continue
        j = resp.json()
        used["tokens"] += j.get("usage", {}).get("input_tokens", 0)
        used["requests"] += 1
        a = j["answers"]
        side = 1 if a["support"]["noul"] >= 0.5 else (-1 if a["oppose"]["noul"] >= 0.5 else 0)
        sides.append(side)
        rows.append({"target": t, "phrase": plan["phrase"], "pid": r.pid, "member": int(r.member), "debate": r.debate, "side": side, "support": a["support"]["noul"], "oppose": a["oppose"]["noul"], "snip": r.snip})
        taken = [s for s in sides if s != 0]
        if len(sides) >= EARLY_STOP_AT and len({x["debate"] for x in rows}) >= MIN_DEBATES_BEFORE_STOP and len(set(taken)) <= 1:
            break
    return rows


start = time.time()
with ThreadPoolExecutor(WORKERS) as ex, OUT.open("w") as f:
    for rows in ex.map(run_target, list(plans)):
        for r in rows:
            f.write(json.dumps(r) + "\n")
print(f"\n{used['requests']} requests in {time.time() - start:.0f}s | tokens {used['tokens']} | cost ${used['tokens'] * 0.042 / 1e6:.4f} | failed {used['failed']}")

res = pd.DataFrame(json.loads(l) for l in OUT.read_text().splitlines())
members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
res["party"] = members.party_abbr.reindex(res.member).to_numpy()
summary = []
for t, g in res.groupby("target"):
    sup, opp, none = (g.side == 1).sum(), (g.side == -1).sum(), (g.side == 0).sum()
    by_party = g[g.side != 0].groupby("party").side.mean().round(1).to_dict()
    summary.append({"phrase": g.phrase.iloc[0], "read": len(g), "support": int(sup), "oppose": int(opp), "no_position": int(none),
                    "contested": bool(sup >= 2 and opp >= 2), "parties": {k: v for k, v in by_party.items() if k in ("Lab", "Con", "LD", "RUK", "SNP", "Green")}})
S = pd.DataFrame(summary)
S.to_json(ROOT / "data/quick/step2b_summary.json", orient="records")
print(f"\nCONTESTED (at least 2 snippets on each side): {S.contested.sum()} of {len(S)}")
for r in S[S.contested].sort_values("read", ascending=False).itertuples():
    print(f"  {r.phrase:38} support {r.support} oppose {r.oppose} none {r.no_position} | {r.parties}")
print(f"\nNOT CONTESTED HERE ({(~S.contested).sum()}):", ", ".join(S[~S.contested].phrase.tolist()))
