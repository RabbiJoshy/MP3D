"""Score each dense candidate against its final unit label, with JSONL checkpoints."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D = ROOT / "data/quick/research_2026_11_10/fp2"
R = HERE / "results"
VERDICTS = {"FOR", "AGAINST", "NEITHER", "ERR"}


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def index_rows(rows, path):
    indexed = {}
    for row in rows:
        key = row["unit"], row["passage_id"]
        if key in indexed:
            raise ValueError(f"{path}: duplicate candidate {key}")
        indexed[key] = row
    return indexed


def check_sides(candidates, sides, *, complete=False):
    unexpected = set(sides) - set(candidates)
    missing = set(candidates) - set(sides)
    if unexpected or (complete and missing):
        raise ValueError(f"Dense checkpoint coverage: {len(unexpected)} unexpected, {len(missing)} missing")
    for key, row in sides.items():
        if row["label"] != candidates[key]["label"]:
            raise ValueError(f"Dense checkpoint uses a stale label for {key}")
        if row["verdict"] not in VERDICTS:
            raise ValueError(f"Dense checkpoint has an invalid verdict for {key}: {row['verdict']}")
    return missing


def repair_partial_line(path):
    if not path.exists():
        return
    # A process interrupted during a write may leave one incomplete record at EOF.
    with path.open("rb+") as stream:
        content = stream.read()
        if content and not content.endswith(b"\n"):
            stream.truncate(content.rfind(b"\n") + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    candidate_path = D / "dense_cands.jsonl"
    sides_path = D / "dense_sides.jsonl"
    candidates = index_rows(read_jsonl(candidate_path), candidate_path)
    repair_partial_line(sides_path)
    sides = index_rows(read_jsonl(sides_path), sides_path) if sides_path.exists() else {}
    missing = check_sides(candidates, sides)
    todo = [row for key, row in candidates.items() if key in missing]
    if args.limit: todo = todo[:args.limit]
    print(f"Dense candidates: {len(candidates)} total, {len(sides)} checkpointed, {len(todo)} pending", flush=True)
    if not todo:
        return
    passages = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "text"])
    texts = passages.drop_duplicates("passage_id").set_index("passage_id")["text"]
    absent = set(row["passage_id"] for row in todo) - set(texts.index)
    if absent:
        raise ValueError(f"{len(absent)} dense candidates have no source passage")
    rows = [dict(row, text=texts.at[row["passage_id"]]) for row in todo]
    if any(not isinstance(row["text"], str) or not row["text"].strip() for row in rows):
        raise ValueError("Dense candidates include empty or invalid speech text")

    from fp_dense2_run import ask, used, verdict

    stats_path = R / "fp2_dense_tokens.json"
    previous_tokens = json.loads(stats_path.read_text())["tokens"] if stats_path.exists() else 0
    try:
        with sides_path.open("a") as stream, ThreadPoolExecutor(args.workers) as executor:
            pending = {executor.submit(ask, row): row for row in rows}
            completed = len(sides)
            failure = None
            for future in as_completed(pending):
                try:
                    p_for, p_against = future.result()
                except Exception as exc:
                    # Save other in-flight responses before propagating a budget or API error.
                    if failure is None:
                        failure = exc
                        for outstanding in pending:
                            outstanding.cancel()
                    continue
                row = {k: v for k, v in pending[future].items() if k != "text"}
                row.update(p_for=p_for, p_against=p_against, verdict=verdict(p_for, p_against))
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                completed += 1
                if completed % 100 == 0:
                    print(f"Dense checkpoint: {completed}/{len(candidates)}", flush=True)
            if failure is not None:
                raise failure
    finally:
        counts = Counter(row["verdict"] for row in read_jsonl(sides_path))
        tokens = previous_tokens + used[0]
        stats = {"tokens": tokens, "usd": round(tokens * 0.042 / 1e6, 6), "verdicts": dict(counts)}
        stats_path.write_text(json.dumps(stats, indent=1) + "\n")
        print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
