"""Stance on targets the data picks: does the speaker support or oppose it?

prepare: rebuild the same 110 topics, choose 10 at random from those with enough MPs, take each topic's two most
         distinctive multi-word phrases as targets, and sample passages, up to 4 per MP from MPs with 2 or more speeches.
run:     Jev answers four yes/no questions per passage: support and oppose, for each target. Asked separately, so the
         order of options can't bias it. Resumable.
The only human input is the wording of those two general questions. Votes and party are only used to test results.
Reads TYPESAFE_API_KEY from .env.

    python prototypes/stance_targets.py prepare
    python prototypes/stance_targets.py run [limit]
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

    N_TOPICS, PER_MP, MAX_PASSAGES = 10, 4, 320
    df["thing"] = thing
    overall = np.asarray(X.mean(0)).ravel()
    mps_per = df.groupby("thing").member.nunique()
    pool = [k for k in range(K_THINGS) if mps_per[k] >= 100]
    chosen = sorted(np.random.default_rng(2026).choice(pool, N_TOPICS, replace=False).tolist())
    rows = []
    for k in chosen:
        mean = np.asarray(X[thing == k].mean(0)).ravel()
        ranked = terms[np.argsort(-(mean * np.log1p(mean / (overall + 1e-6))))[:40]]
        targets = [w for w in ranked if " " in w][:2]
        fr = df[df.thing == k]
        n_sp = fr.groupby("member").speech.nunique()
        keep_mps = n_sp.index[n_sp >= 2]
        sample = fr[fr.member.isin(keep_mps)].sample(frac=1, random_state=SEED).groupby("member").head(PER_MP)
        if len(sample) > MAX_PASSAGES:
            sample = sample.sample(MAX_PASSAGES, random_state=SEED)
        say(f"topic {k}: {mps_per[k]} MPs | top words {', '.join(ranked[:5])} | targets {targets} | sample {len(sample)} passages from {sample.member.nunique()} MPs")
        for r in sample.itertuples():
            rows.append({"pid": f"{r.speech}#{r.Index}", "topic": int(k), "targets": targets, "member": int(r.member), "date": r.date, "debate": r.debate, "text": r.text})
    out = pd.DataFrame(rows)
    out.to_parquet(ROOT / "data/quick/stance_targets_sample.parquet", index=False)
    say(f"total {len(out)} passages, {out.member.nunique()} MPs, {len(chosen) * 2} targets")
else:

    import json, pathlib, re, time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import numpy as np, pandas as pd, requests
    ROOT = pathlib.Path(__file__).resolve().parent.parent
    OUT = ROOT / "data/quick/stance_targets.jsonl"
    URL, MODEL, WORKERS = "https://api.typesafe.ai/v1/systemone", "jev-1.14.0", 8
    key = next(m[1].strip().strip("\"'") for m in (re.match(r"\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)", l) for l in (ROOT / ".env").read_text().splitlines()) if m)
    s = pd.read_parquet(ROOT / "data/quick/stance_targets_sample.parquet")
    done = {json.loads(l)["pid"] for l in OUT.read_text().splitlines() if l.strip()} if OUT.exists() else set()
    todo = s[~s.pid.isin(done)]
    if len(sys.argv) > 2:
        todo = todo.groupby("topic").head(max(1, int(sys.argv[2]) // s.topic.nunique()))
    print(f"{len(s)} passages, {len(done)} done, running {len(todo)}", flush=True)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    used = {"tokens": 0, "failed": 0, "secs": []}


    def ask(r):
        qs = {}
        for i, t in enumerate(r.targets):
            qs[f"support{i}"] = {"type": "noul", "instructions": f"Does the speaker themselves express support for {t}?"}
            qs[f"oppose{i}"] = {"type": "noul", "instructions": f"Does the speaker themselves express opposition to {t}?"}
        for attempt in range(7):
            t0 = time.time()
            try:
                resp = session.post(URL, json={"model": MODEL, "state": r.text, "questions": qs}, timeout=90)
            except requests.RequestException:
                time.sleep(min(60, 2 ** attempt)); continue
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(float(resp.headers.get("retry-after") or min(60, 2 ** attempt))); continue
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            used["secs"].append(time.time() - t0)
            d = resp.json()
            used["tokens"] += d.get("usage", {}).get("input_tokens", 0)
            a = d["answers"]
            return {"pid": r.pid, **{q: a[q]["noul"] for q in qs}}
        used["failed"] += 1
        return None


    start = time.time()
    n = 0
    with ThreadPoolExecutor(WORKERS) as ex, OUT.open("a") as f:
        for fut in as_completed([ex.submit(ask, r) for r in todo.itertuples()]):
            row = fut.result()
            if row:
                f.write(json.dumps(row) + "\n"); n += 1
            if n % 100 == 0:
                f.flush()
                print(f"{n}/{len(todo)} {n / max(1, time.time() - start) * 60:.0f}/min tokens {used['tokens']}", flush=True)
    lat = np.median(used["secs"]) if used["secs"] else 0
    print(f"DONE {n}/{len(todo)} in {time.time() - start:.0f}s ({n / max(1, time.time() - start) * 60:.0f}/min, median request {lat:.2f}s) | tokens {used['tokens']} est ${used['tokens'] * 0.042 / 1e6:.4f} | failed {used['failed']}", flush=True)
