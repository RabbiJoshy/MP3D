"""Tiny, cheap stance pilot: data-made phrases, short snippets, several targets per request.

Step 0 (free): each target word is extended to the phrase it most often appears in, counted from the passages that
mention it. Step 1: 20 snippets per target (the sentences mentioning it, plus one either side), drawn from passages
that were already scored in full, so snippet and full-passage answers can be compared for free.

    python prototypes/stance_pilot.py phrases
    python prototypes/stance_pilot.py pilot
"""
import collections
import json
import pathlib
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGETS = ["scrapping", "threshold", "impose", "rosebank", "echr"]
PER_TARGET, SEED = 20, 7
SPLIT = re.compile(r"(?<=[.?!])\s+(?=[\"“‘A-Z(])")
WORD = re.compile(r"[a-zA-Z][a-zA-Z\-']*")

table = pq.read_table(ROOT / "data/quick/contest_candidates.parquet", columns=["target", "pid", "text"], filters=[("target", "in", TARGETS)])
cand = table.to_pandas().drop_duplicates(["pid", "target"])


def function_words(texts, n=250):
    """The most frequent words in this text are grammar and filler, found from the data rather than a list."""
    c = collections.Counter(w.lower() for t in texts for w in WORD.findall(t))
    return {w for w, _ in c.most_common(n)}


def phrase_for(word, texts, common):
    """The most common 3-6 word phrase containing the word, starting and ending on content words, with at least two
    content words besides the target. Data only."""
    counts = collections.Counter()
    for t in texts:
        toks = [w.lower() for w in WORD.findall(t)]
        for i, w in enumerate(toks):
            if w != word:
                continue
            for n in (3, 4, 5, 6):
                for start in range(max(0, i - n + 1), i + 1):
                    gram = toks[start:start + n]
                    if len(gram) < n or gram[0] in common or gram[-1] in common:
                        continue
                    if sum(g not in common and g != word for g in gram) >= 2:
                        counts[" ".join(gram)] += 1
    if not counts:
        return word, 0
    best, n_best = max(counts.items(), key=lambda kv: (kv[1] * (1 + 0.15 * len(kv[0].split())), kv[1]))
    return (best, n_best) if n_best >= 0.05 * len(texts) else (word, n_best)


def snippet(text, word):
    sents = SPLIT.split(text)
    hit = [i for i, s in enumerate(sents) if re.search(r"\b" + re.escape(word) + r"\b", s, re.I)]
    if not hit:
        return None
    keep = sorted({j for i in hit for j in (i - 1, i, i + 1) if 0 <= j < len(sents)})
    return " ".join(sents[j] for j in keep)


def sense(word, texts, common):
    """Split a vague word by the content word most often within five words of it, then write the pair as the
    shortest phrase MPs most often use that contains both."""
    near = collections.Counter()
    spans = collections.Counter()
    for t in texts:
        toks = [w.lower() for w in WORD.findall(t)]
        for i, w in enumerate(toks):
            if w == word:
                for j in range(max(0, i - 5), min(len(toks), i + 6)):
                    if j != i and toks[j] not in common and len(toks[j]) > 2:
                        near[toks[j]] += 1
    partner = near.most_common(1)[0][0]
    for t in texts:
        toks = [w.lower() for w in WORD.findall(t)]
        for i, w in enumerate(toks):
            if w != word:
                continue
            for j in range(max(0, i - 5), min(len(toks), i + 6)):
                if toks[j] == partner:
                    a, b = min(i, j), max(i, j)
                    end_ = b + 1
                    if end_ < len(toks) and toks[end_] not in common and b == j:
                        end_ += 1
                    spans[" ".join(toks[a:end_])] += 1
    return partner, spans.most_common(1)[0][0]


if sys.argv[1] == "phrases":
    out = {}
    common = function_words(cand.text.tolist())
    for t in TARGETS:
        partner, phrase = sense(t, cand[cand.target == t].text.tolist(), common)
        out[t] = {"partner": partner, "phrase": phrase}
        print(f"{t:10} + {partner:12} -> {phrase!r}")
    (ROOT / "data/quick/pilot_phrases.json").write_text(json.dumps(out))
    sys.exit()

phrases = json.loads((ROOT / "data/quick/pilot_phrases.json").read_text())
full = pd.DataFrame(json.loads(l) for l in (ROOT / "data/quick/contest_answers.jsonl").read_text().splitlines() if l.strip()).drop_duplicates(["pid", "target"])
full["side_full"] = np.where(full.support >= 0.5, 1, np.where(full.oppose >= 0.5, -1, 0))
d = cand.merge(full[["pid", "target", "side_full"]], on=["pid", "target"])
d = d[[bool(re.search(r"\b" + re.escape(phrases[t]["partner"]) + r"\b", x, re.I)) for x, t in zip(d.text, d.target)]]
print("passages mentioning both words:", d.groupby("target").size().to_dict())
# Mix of full-passage answers, so the pilot sees both sides and no-position cases
picks = []
for t in TARGETS:
    g = d[d.target == t].sample(frac=1, random_state=SEED)
    picks.append(g.groupby("side_full").head(PER_TARGET // 3 + 1).head(PER_TARGET))
p = pd.concat(picks)
p["snip"] = [snippet(x, t) for x, t in zip(p.text, p.target)]
p = p.dropna(subset=["snip"])
key = next(m[1].strip().strip("\"'") for m in (re.match(r"\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)", l) for l in (ROOT / ".env").read_text().splitlines()) if m)
s = requests.Session()
s.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
tokens, rows = 0, []
start = time.time()
for r in p.itertuples():
    target = phrases[r.target]["phrase"]
    qs = {"support": {"type": "noul", "instructions": f"Does the speaker themselves express support for {target}?"},
          "oppose": {"type": "noul", "instructions": f"Does the speaker themselves express opposition to {target}?"}}
    resp = s.post("https://api.typesafe.ai/v1/systemone", json={"model": "jev-1.14.0", "state": r.snip, "questions": qs}, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    tokens += j.get("usage", {}).get("input_tokens", 0)
    a = j["answers"]
    side = 1 if a["support"]["noul"] >= 0.5 else (-1 if a["oppose"]["noul"] >= 0.5 else 0)
    rows.append({"target": r.target, "phrase": target, "side_snip": side, "side_full": r.side_full, "sup": round(a["support"]["noul"], 2), "opp": round(a["oppose"]["noul"], 2), "snip": r.snip, "full_words": len(r.text.split())})
res = pd.DataFrame(rows)
res.to_json(ROOT / "data/quick/pilot_results.json", orient="records")
print(f"{len(res)} requests in {time.time() - start:.0f}s | tokens {tokens} ({tokens / max(1, len(res)):.0f} per request, against about 450 for full passages) | cost ${tokens * 0.042 / 1e6:.5f}")
print("\nSNIPPET vs FULL PASSAGE (same passage, same model; full used the bare word, snippet uses the data's phrase)")
for t, g in res.groupby("target", sort=False):
    agree = (g.side_snip == g.side_full).mean()
    flips = ((g.side_snip * g.side_full) == -1).sum()
    print(f"  {t:10} -> {g.phrase.iloc[0]!r}: agree {agree:.0%}, opposite answers {flips}, snippet takes a side {(g.side_snip != 0).mean():.0%} vs full {(g.side_full != 0).mean():.0%}")
lab = {1: "SUPPORTS", -1: "OPPOSES", 0: "no position"}
print("\nEXAMPLES TO READ")
for t, g in res.groupby("target", sort=False):
    print(f"\n=== {g.phrase.iloc[0]}")
    for r in g.head(4).itertuples():
        print(f"  snippet: {lab[r.side_snip]} | full passage: {lab[r.side_full]}\n    {r.snip[:330]!r}")
