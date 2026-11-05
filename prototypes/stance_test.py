"""Have an LLM read asylum passages and extract the claim and stance, then compare with word counts.

    python prototypes/stance_test.py extract MODEL [limit]   # resumable; results go to data/quick/stance_asylum.jsonl
    python prototypes/stance_test.py compare                 # scores MPs and runs the checks

The model sees only the passage: no MP name, party or date. Votes and party are only used to test the result.
Reads GEMINI_API_KEY from .env.
"""
import json
import os
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
import numpy as np
import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data/quick/stance_asylum.jsonl"
ASYLUM_TOPIC, BATCH, WORKERS = 28, 20, 4

PROMPT = """You are coding passages from UK House of Commons speeches about asylum and immigration, for research.
Some words were removed in cleaning, such as MPs' names and place names, so sentences may look slightly clipped.

For each passage, judge what the speaker is arguing, not which words they use. Someone who says "deterrents don't work" is
arguing against deterrence even though they use the word. If they quote or describe a view in order to reject it, code the
view they hold.

Fields:
- id: copy the passage id.
- claim: the speaker's main claim in one plain sentence of at most 20 words. Write it as a claim, not a description.
- stance: the speaker's position on asylum and immigration policy, from -2 to 2, or null if the passage takes no position.
    -2 strongly favours protection: safe routes, refugee rights, welcome, humane treatment; opposes harsher rules.
    -1 leans towards protection.
     0 balanced or mixed, taking a real position between the two.
     1 leans towards control.
     2 strongly favours control: deterrence, removals, lower numbers, tougher rules.
  Judge the policy direction only. Attacking a government as incompetent is not a stance, unless it also says what policy should be.
- attack: true if a main purpose of the passage is criticising a party or government's record or competence, whichever direction.
- evidence: the shortest exact phrase from the passage that best shows the stance, or an empty string if stance is null.
- confidence: low, medium or high.

Return one object per passage. Passages:
"""
SCHEMA = {"type": "ARRAY", "items": {"type": "OBJECT", "required": ["id", "claim", "stance", "attack", "evidence", "confidence"], "properties": {
    "id": {"type": "STRING"}, "claim": {"type": "STRING"}, "stance": {"type": "INTEGER", "nullable": True, "minimum": -2, "maximum": 2},
    "attack": {"type": "BOOLEAN"}, "evidence": {"type": "STRING"}, "confidence": {"type": "STRING", "enum": ["low", "medium", "high"]}}}}


def api_key():
    for line in (ROOT / ".env").read_text().splitlines():
        m = re.match(r"\s*(?:export\s+)?GEMINI_API_KEY=(.*)", line)
        if m:
            return m[1].strip().strip("\"'")
    raise SystemExit("GEMINI_API_KEY not in .env")


def load():
    q = np.load(ROOT / "data/quick/topic_space.npz", allow_pickle=True)
    df = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "member_id", "text", "role", "words", "date"])
    df["topic"] = q["topic"]
    return df[(df.topic == ASYLUM_TOPIC) & (df.role != "minister")].reset_index(drop=True)


def extract(model, limit):
    key = api_key()
    df = load()
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in OUT.read_text().splitlines() if l.strip()}
    todo = df[~df.passage_id.isin(done)]
    if limit:
        todo = todo.sample(min(limit, len(todo)), random_state=7)
    print(f"{len(df)} passages, {len(done)} done, running {len(todo)} on {model}", flush=True)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    usage = {"in": 0, "out": 0, "failed": 0}

    def run(chunk):
        body = PROMPT + "\n\n".join(f"[id: {t.passage_id}]\n{t.text}" for t in chunk.itertuples())
        payload = {"contents": [{"role": "user", "parts": [{"text": body}]}],
                   "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": SCHEMA}}
        wanted = set(chunk.passage_id)
        for attempt in range(7):
            try:
                r = requests.post(url, headers={"x-goog-api-key": key}, json=payload, timeout=180)
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(90, 5 * 2 ** attempt))
                    continue
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                d = r.json()
                text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
                rows = [x for x in json.loads(text) if x.get("id") in wanted]
                u = d.get("usageMetadata", {})
                usage["in"] += u.get("promptTokenCount", 0)
                usage["out"] += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
                return rows
            except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
                time.sleep(min(90, 5 * 2 ** attempt))
                last = e
        usage["failed"] += len(chunk)
        print("gave up on a batch", flush=True)
        return []

    chunks = [todo.iloc[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    n = 0
    with ThreadPoolExecutor(WORKERS) as ex, OUT.open("a") as f:
        for fut in as_completed([ex.submit(run, c) for c in chunks]):
            for row in fut.result():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
            f.flush()
            print(f"{n}/{len(todo)} usage {usage}", flush=True)
    print("DONE", n, usage, flush=True)


def compare():
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import normalize

    df = load()
    rows = {}
    for l in pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else OUT).read_text().splitlines():
        if l.strip():
            x = json.loads(l)
            rows[x["id"]] = x
    res = pd.DataFrame(rows.values()).rename(columns={"id": "passage_id"})
    d = df.merge(res, on="passage_id")
    print(f"coded {len(d)} of {len(df)} passages; no stance {d.stance.isna().mean():.0%}; attack {d.attack.mean():.0%}; confidence {d.confidence.value_counts(normalize=True).round(2).to_dict()}")
    print("stance counts", d.stance.round().value_counts(dropna=False).sort_index().to_dict())
    members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
    votes = pd.read_parquet(ROOT / "data/clean/votes_long.parquet")
    d["party"] = members.party_abbr.reindex(d.member_id).to_numpy()
    d["name"] = members.name.reindex(d.member_id).to_numpy()

    print("\nPassage stance by party (mean, share taking a position):")
    print(d.groupby("party").stance.agg(["mean", "count"]).sort_values("count", ascending=False).head(8).round(2).to_string())

    s = d.dropna(subset=["stance"])
    counts = s.member_id.value_counts()
    ids = counts.index[counts >= 3]
    llm = s[s.member_id.isin(ids)].groupby("member_id").stance.mean()

    STOP = """hon right friend friends member members gentleman lady minister ministers secretary state government house bill clause
    clauses amendment amendments committee debate speaker sir madam people country uk british britain think want wants know need needs make
    makes made said say says going just like does did ask asked agree important new way ensure work working year years time today constituency
    constituents local support issue issues point points absolutely thank thanks labour conservative conservatives liberal democrat democrats
    snp reform party parties tory tories opposition lord lords chancellor prime shadow previous last""".split()
    docs = df[df.member_id.isin(ids)].groupby("member_id").text.apply(" ".join).reindex(ids)
    vec = TfidfVectorizer(stop_words=sorted(ENGLISH_STOP_WORDS | set(STOP)), ngram_range=(1, 2), min_df=3, max_df=0.6, max_features=6000, sublinear_tf=True, token_pattern=r"[a-zA-Z][a-zA-Z\-]{2,}")
    Z = PCA(3, random_state=7).fit_transform(normalize(vec.fit_transform(docs)).toarray())
    party = members.party_abbr.reindex(ids).to_numpy()
    sc = {"LLM stance": llm.reindex(ids).to_numpy(), "word axis 1": Z[:, 0], "word axis 2": Z[:, 1], "word axis 3": Z[:, 2]}

    def auc(x, y):
        a = roc_auc_score(y, x)
        return max(a, 1 - a)

    print(f"\nMPs with 3+ stance passages: {len(ids)}", pd.Series(party).value_counts().head(6).to_dict())
    print(f"(word axes are flipped to whichever direction scores best, which flatters them; the LLM stance has a fixed direction)")
    lc = np.isin(party, ["Lab", "Con"])
    tests = [("Labour vs Conservative", None, None)] + [(f"Vote {v}: {t}", v, g) for v, t, g in [
        (2406, "2026 asylum bill", "all"), (2406, "2026 asylum bill", "Lab"), (2071, "assisted dying, unrelated check", "Lab"), (2074, "welfare bill, unrelated check", "Lab")]]
    for name, div, group in tests:
        row = f"{name:44}"
        for label, x in sc.items():
            if div is None:
                y, xx = party[lc] == "Con", x[lc]
                n = f"n={lc.sum()}"
            else:
                v = votes[votes.division_id == div].drop_duplicates("member_id", keep=False).set_index("member_id").vote
                m = np.array([i in v.index for i in ids]) & (party == group if group != "all" else True)
                y, xx = v.reindex(ids[m]).to_numpy() == 1, x[m]
                n = f"{group} n={m.sum()} minority={min(y.sum(), (~y).sum())}"
            row += f" {label} {auc(xx, y):.2f} |"
        print(row, n)

    # Reliability: two random halves of an MP's coded passages
    rng = np.random.default_rng(7)
    half = rng.random(len(s)) < .5
    a, b = s[half].groupby("member_id").stance.mean(), s[~half].groupby("member_id").stance.mean()
    na, nb = s[half].member_id.value_counts(), s[~half].member_id.value_counts()
    both = a.index.intersection(b.index)
    both = both[(na.reindex(both) >= 3) & (nb.reindex(both) >= 3)]
    print(f"\nsplit-half correlation of MP stance: {np.corrcoef(a[both], b[both])[0, 1]:.2f} ({len(both)} MPs)")
    print("within Labour:", round(float(np.corrcoef(a[both][members.party_abbr.reindex(both) == 'Lab'], b[both][members.party_abbr.reindex(both) == 'Lab'])[0, 1]), 2))

    order = llm.sort_values()
    lab = order[members.party_abbr.reindex(order.index) == "Lab"]
    con = order[members.party_abbr.reindex(order.index) == "Con"]
    fmt = lambda x: [f"{members.name[i]} ({members.party_abbr[i]}) {v:.1f}" for i, v in x.items()]
    print("\nMost pro-protection:", fmt(order.head(10)))
    print("Most pro-control:", fmt(order.tail(10)))
    print("Labour pro-protection end:", fmt(lab.head(8)))
    print("Labour pro-control end:", fmt(lab.tail(8)))
    print("Conservative pro-protection end:", fmt(con.head(6)))
    v = votes[votes.division_id == 2406].drop_duplicates("member_id", keep=False).set_index("member_id").vote
    rebels = [i for i in v.index[(v == -1)] if members.party_abbr.get(i) == "Lab"]
    print("Labour rebels on asylum bill:", fmt(llm.reindex([r for r in rebels if r in llm.index]).dropna().sort_values()), "| Labour mean", round(float(lab.mean()), 2))

    trap = d[d.text.str.contains(r"\b(deterr|illegal migrant|smash|safe route|safe and legal|refugees? (?:are|is) welcome)", case=False, regex=True)]
    print("\nWORD TRAPS: passages using 'deterrent' or 'deterrence'")
    det = d[d.text.str.contains(r"\bdeterr", case=False, regex=True)]
    print("stance counts:", det.stance.round().value_counts(dropna=False).sort_index().to_dict())
    for t in det[det.stance < 0].sample(min(6, (det.stance < 0).sum()), random_state=7).itertuples():
        i = t.text.lower().find("deterr")
        print(f"  [{t.stance:+.0f} {t.confidence}] {t.party} | {t.text[max(0, i - 130):i + 110]!r}\n      claim: {t.claim}")
    print("\nRANDOM SAMPLE TO READ")
    for t in d.sample(12, random_state=11).itertuples():
        print(f"  [{t.stance} attack={t.attack} {t.confidence}] {t.party} | {t.text[:230]!r}\n      claim: {t.claim} | evidence: {t.evidence}")


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    else:
        compare()
