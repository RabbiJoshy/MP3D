"""Expand two-sided + one-sided propositions at sim>=0.56 using jev-1.14.0, verify hits, and merge into accepted.json."""
import collections
import concurrent.futures
import json
import pathlib
import re
import sys
import threading

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

R = pathlib.Path(__file__).resolve().parents[2]
O = R / "data/quick/research_2026_11_12/bridge"
O.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R / "prototypes/research_2026_11_10"))
import fp_dense2_run as J  # noqa: E402

PRICE = 0.042 / 1e6
CAP_USD = 1.00
SIM_MIN = 0.56
WORKERS = 16

EXTRA_VERIFY = (
    " Judge the EXACT proposal, including its named bill, scope, numerical limits and deadlines. "
    "General support for a bill does not establish support for an individual provision unless the speaker addresses it. "
    "Support for a related goal without the specified scope or deadline is insufficient. "
    "Resolve references such as this Bill using the supplied debate heading. "
    "If it refers to a different bill, or the exact proposal is not established, answer false. "
    "Do not infer the position from party affiliation."
)


def select_candidates():
    html = (R / "docs/index.html").read_text(encoding="utf-8")
    D = json.loads(re.search(r'<script id="d" type="application/json">(.*?)</script>', html, re.S).group(1))
    MPs = D["mps"] + D["po"]
    mi = {m["id"]: i for i, m in enumerate(MPs)}
    P = pd.read_parquet(R / "data/clean/passages.parquet")
    E = np.load(R / "data/embed/passages.npy", mmap_mode="r")
    assert len(P) == len(E)

    pidrow = {p: i for i, p in enumerate(P.passage_id)}
    ids = P.member_id.fillna(-1).astype(int).to_numpy()
    words = pd.to_numeric(P.words)
    okay = P.member_id.isin(mi) & (words >= 60) & ~P.speech_type.fillna("").str.contains("Chair")

    cache = set()
    for fp in [
        R / "data/quick/research_2026_11_10/fp2/dense_sides.jsonl",
        R / "data/quick/research_2026_11_10/pl/judge_out.jsonl",
        O / "judgments.jsonl",
        O / "judgments_full.jsonl",
        O / "judgments_expand.jsonl",
    ]:
        if fp.exists():
            for line in fp.open():
                x = json.loads(line)
                if x.get("verdict") in ("FOR", "AGAINST", "NEITHER"):
                    cache.add((x["label"], x["passage_id"]))

    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu", local_files_only=True)
    model.max_seq_length = 256

    counts = np.array([
        sum(f["sc"][i] is not None for f in D["fights"]) if i < len(D["mps"]) else 0
        for i in range(len(MPs))
    ])

    selected = []
    # 1. Two-sided fights (fill remaining gaps at sim>=0.56)
    Q1 = model.encode([f["label"] for f in D["fights"]], normalize_embeddings=True, show_progress_bar=False)
    for k, f in enumerate(D["fights"]):
        pids = [p for claims in D["mpclaims"].values() for fi, side, cs in claims if fi == k for p, _ in cs]
        rows = [pidrow[p] for p in pids if p in pidrow]
        headings = set(P.iloc[rows].major.dropna()) - {"", "Oral Answers to Questions", "Business of the House"}
        if not headings:
            continue
        ix = np.where((okay & P.major.isin(headings)).to_numpy())[0]
        if not len(ix):
            continue
        sims = np.asarray(E[ix]) @ Q1[k]
        seen = set()
        for a in np.argsort(-sims):
            if sims[a] < SIM_MIN:
                break
            r = int(ix[a])
            mid = int(ids[r])
            i = mi[mid]
            pid = P.iloc[r].passage_id
            if mid in seen or (i < len(D["mps"]) and f["sc"][i] is not None):
                continue
            seen.add(mid)
            if (f["label"], pid) in cache:
                continue
            selected.append(dict(
                pool="two_sided",
                fight=k,
                label=f["label"],
                member_id=mid,
                mp=MPs[i]["n"],
                party=MPs[i]["p"],
                passage_id=pid,
                similarity=float(sims[a]),
                existing_positions=int(counts[i]),
                heading=str(P.iloc[r].major),
                date=str(P.iloc[r].date)[:10],
                text=str(P.iloc[r].text)[:3000],
            ))

    # 2. One-sided propositions (promote into two-sided)
    Q2 = model.encode([o["label"] for o in D["os"]], normalize_embeddings=True, show_progress_bar=False)
    for k, o in enumerate(D["os"]):
        pids = [p for claims in D["mpone"].values() for oi, side, cs in claims if oi == k for p, _ in cs]
        rows = [pidrow[p] for p in pids if p in pidrow]
        headings = set(P.iloc[rows].major.dropna()) - {"", "Oral Answers to Questions", "Business of the House"}
        if not headings:
            continue
        ix = np.where((okay & P.major.isin(headings)).to_numpy())[0]
        if not len(ix):
            continue
        sims = np.asarray(E[ix]) @ Q2[k]
        seen = set()
        for a in np.argsort(-sims):
            if sims[a] < SIM_MIN:
                break
            r = int(ix[a])
            mid = int(ids[r])
            i = mi[mid]
            pid = P.iloc[r].passage_id
            if mid in seen:
                continue
            seen.add(mid)
            if (o["label"], pid) in cache:
                continue
            selected.append(dict(
                pool="one_sided",
                fight=k,
                label=o["label"],
                member_id=mid,
                mp=MPs[i]["n"],
                party=MPs[i]["p"],
                passage_id=pid,
                similarity=float(sims[a]),
                existing_positions=int(counts[i]),
                heading=str(P.iloc[r].major),
                date=str(P.iloc[r].date)[:10],
                text=str(P.iloc[r].text)[:3000],
            ))

    # Prioritise high similarity + thin MPs
    selected.sort(key=lambda x: -(x["similarity"] + 0.10 * (x["existing_positions"] < 3)))
    return selected


def run_judge(rows):
    out = O / "judgments_expand.jsonl"
    old = [json.loads(l) for l in out.open()] if out.exists() else []
    done = {(x["label"], x["passage_id"]) for x in old if x.get("verdict") != "ERR"}
    spent = sum(x.get("tokens", 0) for x in old)
    calls = sum(not x.get("reused", False) for x in old)
    reserved = 0
    lock = threading.Lock()
    stop = threading.Event()
    f = out.open("a")

    def work(row):
        nonlocal spent, reserved, calls
        key = (row["label"], row["passage_id"])
        if key in done or stop.is_set():
            return
        q = {
            "for": {"type": "noul", "instructions": J.PROMPT_FOR.format(m=row["label"])},
            "against": {"type": "noul", "instructions": J.PROMPT_AGAINST.format(m=row["label"])},
        }
        reserve = len((row["text"] + json.dumps(q)).encode()) + 512
        with lock:
            if (spent + reserved + reserve) * PRICE > CAP_USD:
                stop.set()
                return
            reserved += reserve
            calls += 1
        result = {k: v for k, v in row.items() if k != "text"}
        try:
            resp = J.S.post(
                "https://api.typesafe.ai/v1/systemone",
                json={"model": "jev-1.14.0", "state": row["text"], "questions": q},
                timeout=35,
            )
            if resp.status_code != 200:
                result.update(verdict="ERR", status=resp.status_code, tokens=0, reused=False)
                if resp.status_code in (401, 402, 403):
                    stop.set()
            else:
                body = resp.json()
                a = body["answers"]
                pf, pa = a["for"]["noul"], a["against"]["noul"]
                result.update(
                    p_for=pf,
                    p_against=pa,
                    verdict=J.verdict(pf, pa),
                    tokens=body["usage"]["input_tokens"],
                    reused=False,
                )
        except Exception:
            result.update(verdict="ERR", status="request_or_response_error", tokens=0, reused=False)
        with lock:
            reserved -= reserve
            spent += result["tokens"]
            f.write(json.dumps(result) + "\n")
            f.flush()
            done.add(key)
            if len(done) % 500 == 0:
                print(f"judge: {len(done)} done, USD {spent * PRICE:.4f}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(work, rows))
    f.close()
    return spent


def run_verify(rows_by_key, spent_so_far):
    j_path = O / "judgments_expand.jsonl"
    v_path = O / "verified_expand.jsonl"
    all_j = [json.loads(l) for l in j_path.open()] if j_path.exists() else []
    hits = [
        x for x in all_j
        if x.get("verdict") in ("FOR", "AGAINST")
        and max(x.get("p_for", 0), x.get("p_against", 0)) >= 0.8
        and min(x.get("p_for", 1), x.get("p_against", 1)) <= 0.2
    ]
    old_v = [json.loads(l) for l in v_path.open()] if v_path.exists() else []
    done = {(x["label"], x["passage_id"]) for x in old_v if x.get("verdict") != "ERR"}
    todo = [x for x in hits if (x["label"], x["passage_id"]) not in done and (x["label"], x["passage_id"]) in rows_by_key]
    print(f"verify: {len(hits)} high-conf hits, {len(todo)} to verify", flush=True)

    spent = spent_so_far + sum(x.get("tokens", 0) for x in old_v)
    reserved = 0
    lock = threading.Lock()
    stop = threading.Event()
    f = v_path.open("a")

    def work(x):
        nonlocal spent, reserved
        key = (x["label"], x["passage_id"])
        if stop.is_set():
            return
        c = rows_by_key[key]
        state = f"DEBATE HEADING: {c['heading']}\nDATE: {c['date']}\nSPEAKER: {x['mp']}\nPASSAGE:\n{c['text']}"
        q = {
            "for": {"type": "noul", "instructions": J.PROMPT_FOR.format(m=x["label"]) + EXTRA_VERIFY},
            "against": {"type": "noul", "instructions": J.PROMPT_AGAINST.format(m=x["label"]) + EXTRA_VERIFY},
        }
        reserve = len((state + json.dumps(q)).encode()) + 512
        with lock:
            if (spent + reserved + reserve) * PRICE > CAP_USD:
                stop.set()
                return
            reserved += reserve
        out = {**x, "initial_verdict": x["verdict"]}
        try:
            r = J.S.post(
                "https://api.typesafe.ai/v1/systemone",
                json={"model": "jev-1.14.0", "state": state, "questions": q},
                timeout=35,
            )
            if r.status_code != 200:
                out.update(verdict="ERR", tokens=0, status=r.status_code)
                if r.status_code in (401, 402, 403):
                    stop.set()
            else:
                b = r.json()
                pf, pa = b["answers"]["for"]["noul"], b["answers"]["against"]["noul"]
                out.update(verdict=J.verdict(pf, pa), p_for=pf, p_against=pa, tokens=b["usage"]["input_tokens"])
        except Exception:
            out.update(verdict="ERR", tokens=0, status="request_or_response_error")
        with lock:
            reserved -= reserve
            spent += out["tokens"]
            f.write(json.dumps(out) + "\n")
            f.flush()

    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(work, todo))
    f.close()
    return spent


def merge_accepted():
    acc_path = O / "accepted.json"
    existing = json.loads(acc_path.read_text()) if acc_path.exists() else []
    seen = {(e["label"], e["passage_id"]) for e in existing}
    v_path = O / "verified_expand.jsonl"
    added = 0
    if v_path.exists():
        for line in v_path.open():
            x = json.loads(line)
            if x.get("verdict") in ("FOR", "AGAINST") and x.get("verdict") == x.get("initial_verdict"):
                key = (x["label"], x["passage_id"])
                if key not in seen:
                    existing.append(x)
                    seen.add(key)
                    added += 1
    acc_path.write_text(json.dumps(existing))
    print(f"merged {added} new verified positions -> accepted.json total = {len(existing)}", flush=True)


def main():
    rows = select_candidates()
    print(f"selected {len(rows)} uncached candidates (sim >= {SIM_MIN})", flush=True)
    rows_by_key = {(r["label"], r["passage_id"]): r for r in rows}
    spent = run_judge(rows)
    spent = run_verify(rows_by_key, spent)
    merge_accepted()
    print(f"DONE total tokens={spent}, total USD={spent * PRICE:.4f}", flush=True)


if __name__ == "__main__":
    main()
