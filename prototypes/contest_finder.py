"""Let the data find what's contested: which targets do MPs actually split on?

prepare: rebuild the 110 topics; for each, take its most distinctive phrases as candidate targets and keep every
         backbench or opposition passage that mentions each one.
screen:  Jev reads a small sample of passages per candidate: does the speaker support it, and do they oppose it?
deep:    candidates where both sides appear, from several MPs, get every mentioning passage scored.
The only human input is the wording of those two general questions and the thresholds below. Votes and party are only
used to test results. Reads TYPESAFE_API_KEY from .env.
"""
import sys
MODE = sys.argv[1]
if MODE == "prepare":
    import json
    import os
    import pathlib
    import re
    import time
    import xml.etree.ElementTree as ET

    os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.stats import spearmanr
    from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.manifold import MDS
    from sklearn.preprocessing import normalize

    ROOT = pathlib.Path(__file__).resolve().parent.parent
    RAW = ROOT / "data/raw"
    K_THINGS, STANCES_PER_THING, SVD_DIMS, CHUNK_WORDS, MIN_WORDS, MAX_DEBATE_SHARE, MIN_MP_SPEECHES, SPACE_DIMS, SEED = 110, 3, 100, 120, 25, 0.10, 12, 25, 7
    rng = np.random.default_rng(SEED)
    lines = []
    t0 = time.time()
    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)

    people = json.loads((RAW / "people.json").read_text())
    person_to_member = {p["id"]: int(i["identifier"]) for p in people["persons"] for i in p.get("identifiers", []) if i.get("scheme") == "datadotparl_id"}
    members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
    gov_posts = {m: json.loads(v) for m, v in members.gov_posts.items()}
    chair_posts = {m: [p for p in json.loads(v) if re.search("speaker|ways and means", p.get("name") or "", re.I)] for m, v in members.other_posts.items()}
    def in_post(posts, date):
        return any((p.get("startDate") or "")[:10] <= date <= (p.get("endDate") or "9999")[:10] for p in posts)

    BRACKET = re.compile(r"\[[^\]]*\]")
    SPLIT = re.compile(r"(?<=[.?!])\s+(?=[\"“‘A-Z(])")
    tidy = lambda t: BRACKET.sub("", " ".join("".join(t.itertext()).split())).strip(" —-")
    rows = []
    for kind in ("debates", "westminhall", "standing"):
        for f in sorted((RAW / "hansard" / kind).glob("*.xml")):
            date = re.search(r"\d{4}-\d{2}-\d{2}", f.name).group()
            root = ET.parse(f).getroot()
            major, minor = "", ""
            chairs = {m.get("person_id") for m in root.findall("./committee/chairmen/mpname")}
            bill = root.find("bill")
            if kind == "standing":
                major = bill.get("title") if bill is not None else ""
            for el in root.iter():
                if el.tag in ("major-heading", "oral-heading") and kind != "standing":
                    major, minor = tidy(el), ""
                elif el.tag == "minor-heading" and kind != "standing":
                    minor = tidy(el)
                elif el.tag == "speech" and el.get("person_id"):
                    pid = el.get("person_id")
                    member = person_to_member.get(pid)
                    if member not in members.index or pid in chairs or in_post(gov_posts.get(member, []), date) or in_post(chair_posts.get(member, []), date):
                        continue
                    text = " ".join(" ".join("".join(p.itertext()).split()) for p in el.findall("p") if p.get("class") != "italic" and p.get("pwmotiontext") != "yes")
                    buf, n = [], 0
                    for sent in SPLIT.split(re.sub(r"\bhon\.\s", "hon ", text)):
                        w = len(sent.split())
                        if n + w > CHUNK_WORDS * 1.4 and n >= MIN_WORDS:
                            rows.append((el.get("id"), member, date, minor or major, " ".join(buf)))
                            buf, n = [], 0
                        buf.append(sent)
                        n += w
                    if n >= MIN_WORDS:
                        rows.append((el.get("id"), member, date, minor or major, " ".join(buf)))
    df = pd.DataFrame(rows, columns=["speech", "member", "date", "heading", "text"])
    df["debate"] = df.date + "|" + df.heading
    debate_idx, debates = pd.factorize(df.debate)
    say(f"{len(df)} non-ministerial passages by {df.member.nunique()} MPs, {time.time() - t0:.0f}s")

    vec = CountVectorizer(ngram_range=(1, 2), min_df=25, max_features=100000, token_pattern=r"[a-zA-Z][a-zA-Z\-']+", lowercase=True, dtype=np.int32)
    vec.fit(df.text.iloc[rng.choice(len(df), min(100000, len(df)), replace=False)])
    C = vec.transform(df.text)
    terms = np.array(vec.get_feature_names_out())
    D = sp.csr_matrix((np.ones(len(df)), (np.arange(len(df)), debate_idx)), shape=(len(df), len(debates)))
    share = np.asarray((((C > 0).astype(np.int32).T @ D) > 0).sum(1)).ravel() / len(debates)
    identity = {w for col in ["name", "constituency", "party"] for v in members[col].dropna() for w in re.findall(r"[a-zA-Z][a-zA-Z\-']+", str(v).lower())}
    generic = share > MAX_DEBATE_SHARE
    named = np.array([any(w in identity for w in t.split()) for t in terms])
    generic_words = set(terms[generic & np.array([" " not in t for t in terms])])
    pair = np.array([" " in t and all(w in generic_words for w in t.split()) for t in terms])
    keep = ~(generic | named | pair)
    C, terms = C[:, keep], terms[keep]
    enough = np.asarray((C > 0).sum(1)).ravel() >= 3
    df, C = df[enough].reset_index(drop=True), C[enough]
    X = TfidfTransformer(sublinear_tf=True).fit_transform(C)
    L = normalize(TruncatedSVD(SVD_DIMS, random_state=SEED, n_iter=4).fit_transform(X)).astype("float32")
    say(f"{len(df)} passages as {SVD_DIMS}-number vectors, vocabulary {len(terms)}, {time.time() - t0:.0f}s")

    # Parameter 1: the thing
    thing = MiniBatchKMeans(K_THINGS, n_init=5, batch_size=8192, max_iter=150, random_state=SEED).fit_predict(L)

    CANDIDATES, MIN_MENTIONS = 10, 30
    df["thing"] = thing
    overall = np.asarray(X.mean(0)).ravel()
    lower = df.text.str.lower()
    rows = []
    for k in range(K_THINGS):
        idx = np.where(thing == k)[0]
        mean = np.asarray(X[idx].mean(0)).ravel()
        ranked = terms[np.argsort(-(mean * np.log1p(mean / (overall + 1e-6))))[:60]]
        picked = []
        for w in ranked:
            if any(w in p or p in w for p in picked):
                continue
            hits = lower.str.contains(r"\b" + re.escape(w) + r"\b", regex=True)
            if hits.sum() >= MIN_MENTIONS:
                picked.append(w)
                sub = df[hits.to_numpy()]
                for r in sub.itertuples():
                    rows.append({"topic": int(k), "target": w, "pid": f"{r.speech}#{r.Index}", "member": int(r.member), "date": r.date, "debate": r.debate, "text": r.text})
            if len(picked) == CANDIDATES:
                break
    c = pd.DataFrame(rows)
    c.to_parquet(ROOT / "data/quick/contest_candidates.parquet", index=False)
    say(f"{c.target.nunique()} candidate targets across {c.topic.nunique()} topics; {len(c)} passage-target pairs; median mentions {int(c.groupby('target').size().median())}")
else:

    import json, pathlib, re, time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import numpy as np, pandas as pd, requests
    ROOT = pathlib.Path(__file__).resolve().parent.parent
    URL, MODEL, WORKERS = "https://api.typesafe.ai/v1/systemone", "jev-1.14.0", 8
    SCREEN_N, DEEP_CAP, MIN_SIDE_MPS, MIN_CONTEST = 16, 300, 3, 0.2
    key = next(m[1].strip().strip("\"'") for m in (re.match(r"\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)", l) for l in (ROOT / ".env").read_text().splitlines()) if m)
    c = pd.read_parquet(ROOT / "data/quick/contest_candidates.parquet")
    OUT = ROOT / "data/quick/contest_answers.jsonl"
    done = set()
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            if l.strip():
                x = json.loads(l)
                done.add((x["pid"], x["target"]))


    def contested_targets():
        a = pd.DataFrame(json.loads(l) for l in OUT.read_text().splitlines() if l.strip())
        a = a.merge(c[["pid", "target", "member", "topic"]].drop_duplicates(["pid", "target"]), on=["pid", "target"])
        a["side"] = np.where(a.support >= 0.5, 1, np.where(a.oppose >= 0.5, -1, 0))
        g = a[a.side != 0].groupby("target")
        stats = pd.DataFrame({"n": a.groupby("target").size(), "positions": g.size(), "sup_mps": g.apply(lambda x: x[x.side == 1].member.nunique()), "opp_mps": g.apply(lambda x: x[x.side == -1].member.nunique())}).fillna(0)
        stats["contest"] = stats[["sup_mps", "opp_mps"]].min(1) / stats[["sup_mps", "opp_mps"]].sum(1).clip(1)
        return stats, a


    if MODE == "screen":
        todo = c.sample(frac=1, random_state=7).groupby("target").head(SCREEN_N)
    elif MODE == "deep":
        stats, _ = contested_targets()
        keep = stats[(stats.contest >= MIN_CONTEST) & (stats.sup_mps >= MIN_SIDE_MPS) & (stats.opp_mps >= MIN_SIDE_MPS)].index
        print(f"contested after screening: {len(keep)} of {len(stats)} targets", flush=True)
        todo = c[c.target.isin(keep)].sample(frac=1, random_state=7).groupby("target").head(DEEP_CAP)
    todo = todo[[(p, t) not in done for p, t in zip(todo.pid, todo.target)]]
    print(f"running {len(todo)} passage-target pairs", flush=True)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    used = {"tokens": 0, "failed": 0}


    def ask(r):
        qs = {"support": {"type": "noul", "instructions": f"Does the speaker themselves express support for {r.target}?"},
              "oppose": {"type": "noul", "instructions": f"Does the speaker themselves express opposition to {r.target}?"}}
        for attempt in range(7):
            try:
                resp = session.post(URL, json={"model": MODEL, "state": r.text, "questions": qs}, timeout=90)
            except requests.RequestException:
                time.sleep(min(60, 2 ** attempt)); continue
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(float(resp.headers.get("retry-after") or min(60, 2 ** attempt))); continue
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            d = resp.json()
            used["tokens"] += d.get("usage", {}).get("input_tokens", 0)
            return {"pid": r.pid, "target": r.target, "support": d["answers"]["support"]["noul"], "oppose": d["answers"]["oppose"]["noul"]}
        used["failed"] += 1
        return None


    start = time.time()
    n = 0
    with ThreadPoolExecutor(WORKERS) as ex, OUT.open("a") as f:
        for fut in as_completed([ex.submit(ask, r) for r in todo.itertuples()]):
            row = fut.result()
            if row:
                f.write(json.dumps(row) + "\n"); n += 1
            if n % 2000 == 0:
                f.flush()
                print(f"{n}/{len(todo)} {n / max(1, time.time() - start) * 60:.0f}/min", flush=True)
    print(f"DONE {MODE}: {n}/{len(todo)} in {time.time() - start:.0f}s | tokens {used['tokens']} est ${used['tokens'] * 0.042 / 1e6:.3f} | failed {used['failed']}", flush=True)
    if MODE == "screen":
        stats, _ = contested_targets()
        print("\nMOST CONTESTED IN THE SCREEN (share of MPs on the smaller side)")
        print(stats.sort_values("contest", ascending=False).head(40).to_string())
        print("\nhow many pass:", int(((stats.contest >= MIN_CONTEST) & (stats.sup_mps >= MIN_SIDE_MPS) & (stats.opp_mps >= MIN_SIDE_MPS)).sum()), "of", len(stats))
