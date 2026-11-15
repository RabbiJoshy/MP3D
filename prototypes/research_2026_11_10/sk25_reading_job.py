"""sk25: reading job for the remaining unread positions, plus the merge step.
  build : write blind chunk files (proposal + debate + passage, no machine verdict) for
          A) two-sided map positions with judge confidence >= 0.9 not yet re-read
          B) one-sided positions with confidence 0.7-0.9 not yet re-read (shown in MP profiles)
  merge : validate the readers' marks and APPEND them to overrides/positions.jsonl (never rewrites it)
Usage: .venv/bin/python prototypes/research_2026_11_10/sk25_reading_job.py build|merge
"""
import json, sys, glob, collections, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D"); RES = ROOT / "prototypes/research_2026_11_10/results"
OV = RES / "overrides"; OUT = OV / "final_read"; POS = OV / "positions.jsonl"
def done(): return {(x["label"], x["passage_id"]) for x in map(json.loads, POS.open())}
def build():
    OUT.mkdir(exist_ok=True)
    W = [json.loads(l) for l in open(OV / "older/worklist_older.jsonl")]; d = done()
    A = [w for w in W if w["two_sided"] and w["conf"] >= 0.9 and (w["label"], w["passage_id"]) not in d]
    B = [w for w in W if not w["two_sided"] and 0.7 <= w["conf"] < 0.9 and (w["label"], w["passage_id"]) not in d]
    F = {f["label"]: f for f in json.loads((RES / "fp2_fights.json").read_text())["fights"]}
    T = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "major", "minor", "text"]).set_index("passage_id")
    todo = []
    for job, rows in (("A", A), ("B", B)):
        rows.sort(key=lambda w: (w["label"], w["passage_id"]))
        for w in rows: todo.append(dict(w, job=job, fid=len(todo)))
    json.dump(todo, open(OUT / "worklist.json", "w"))
    for old in glob.glob(str(OUT / "*_input.jsonl")): pathlib.Path(old).unlink()
    sizes = {}
    for job in ("A", "B"):
        rows = [w for w in todo if w["job"] == job]; n = max(1, int(np.ceil(len(rows) / 250)))
        for c in range(n):
            fn = OUT / f"{job}{c + 1:02d}_input.jsonl"
            with fn.open("w") as f:
                for w in rows[c::n]:
                    r = T.loc[w["passage_id"]]; m = F[w["label"]]
                    f.write(json.dumps({"id": w["fid"], "proposal": w["label"], "supporter_means": m.get("for_means") or "", "opponent_means": m.get("against_means") or "",
                                        "debate": r.major + (" — " + r.minor if r.minor else ""), "passage": r.text[:2200]}) + "\n")
            sizes[fn.name] = len(rows[c::n])
    print("A (map positions, confidence >= 0.9):", len(A), "| B (profile positions, 0.7-0.9):", len(B)); print(sizes)
def merge():
    W = {w["fid"]: w for w in json.load(open(OUT / "worklist.json"))}; d = done(); rows = []; bad = 0
    for f in sorted(glob.glob(str(OUT / "*_marks.jsonl"))):
        I = {json.loads(l)["id"] for l in open(f.replace("_marks", "_input"))}; R = [json.loads(l) for l in open(f) if l.strip()]; ids = [r["id"] for r in R]
        miss, foreign, dups, notes = len(I - set(ids)), sum(i not in I for i in ids), len(ids) - len(set(ids)), len({r.get("note") for r in R})
        ok = not miss and not foreign and not dups and notes >= 0.9 * len(R) and all(r.get("human_side") in ("FOR", "AGAINST", "NEITHER") for r in R)
        print(pathlib.Path(f).name, "rows", len(R), "missing", miss, "foreign", foreign, "dups", dups, "distinct notes", notes, "OK" if ok else "REJECTED (not merged)")
        if not ok: bad += 1; continue
        for m in R:
            w = W[m["id"]]
            if (w["label"], w["passage_id"]) in d: continue
            agree = m["human_side"] == w["machine_side"]      # a reader landing on the opposite side is treated as unclear and dropped, as on 13-14 Nov
            rows.append({"label": w["label"], "passage_id": w["passage_id"], "member_id": w["member_id"], "machine_side": w["machine_side"], "p_for": w["p_for"], "p_against": w["p_against"],
                         "human_side": m["human_side"], "action": "confirm" if agree else "drop", "reason": "correct" if agree else (m.get("reason", "other") if m["human_side"] == "NEITHER" else "wrong_polarity"),
                         "note": m.get("note", ""), "source": w["source"], "group": "final_read_" + w["job"], "blind": True, "reviewer": m.get("reviewer"), "date": "2026-11-15"})
    with POS.open("a") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    c = collections.Counter((r["group"], r["action"]) for r in rows); print("appended", len(rows), dict(c), "| rejected files:", bad)
if __name__ == "__main__": {"build": build, "merge": merge}[sys.argv[1]]()
