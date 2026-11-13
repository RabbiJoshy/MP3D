"""sk23: worklist of OLDER positions (fp2 dense, pilot, bridge) for a blind second read.
Replays the builder's loading (fp2_fights + pilot + bridge), tags each position with a review group, and writes
blind chunk files (proposal + debate + passage, no machine verdict) for readers. Free, local, no API calls.
Groups: A = two-sided argument, judge confidence < 0.7 or unknown;  B = random 8 per two-sided argument (not in A);
        D = confidence 0.7-0.9 on a two-sided argument for an MP with <= 4 positions;  later = everything else (not sent).
"""
import json, pathlib, collections, random
import pandas as pd
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D"); RES = ROOT / "prototypes/research_2026_11_10/results"
OV = RES / "overrides"; OUT = OV / "older"; OUT.mkdir(exist_ok=True); Q = ROOT / "data/quick"
F = json.loads((RES / "fp2_fights.json").read_text())["fights"]
pos = {}      # (label, passage_id) -> row
for f in F:
    for k in ("mps_for", "mps_against", "mps_both_sides"):
        for m in f[k]:
            for c in m["claims"]: pos[(f["label"], c["passage_id"])] = {"label": f["label"], "passage_id": c["passage_id"], "member_id": int(m["member_id"]), "machine_side": c["side"], "source": "fp2", "p_for": None, "p_against": None}
j = pd.read_json(Q / "research_2026_11_10/pl/judge_out.jsonl", lines=True); j = j[j.verdict.isin(["FOR", "AGAINST"])]
for r in j.itertuples():
    k = (F[int(r.fight)]["label"], r.passage_id)
    if k not in pos: pos[k] = {"label": k[0], "passage_id": r.passage_id, "member_id": int(r.member_id), "machine_side": r.verdict, "source": "pilot", "p_for": getattr(r, "p_for", None), "p_against": getattr(r, "p_against", None)}
labels = {f["label"] for f in F}
for r in json.loads((Q / "research_2026_11_12/bridge/accepted.json").read_text()):
    k = (r["label"], r["passage_id"])
    if r.get("verdict") in ("FOR", "AGAINST") and r["label"] in labels and k not in pos:
        pos[k] = {"label": r["label"], "passage_id": r["passage_id"], "member_id": int(r["member_id"]), "machine_side": r["verdict"], "source": "bridge", "p_for": r.get("p_for"), "p_against": r.get("p_against")}
for fn in ("research_2026_11_10/fp2/dense_sides.jsonl", "research_2026_11_10/fp/dense_sides.jsonl", "research_2026_11_10/fp/jev_sides.jsonl"):      # judge confidence for fp2 claims
    fp = Q / fn
    if fp.exists():
        for line in fp.open():
            x = json.loads(line); k = (x.get("label"), x.get("passage_id"))
            if k in pos and pos[k]["p_for"] is None and x.get("p_for") is not None: pos[k]["p_for"], pos[k]["p_against"] = x["p_for"], x["p_against"]
done = {(x["label"], x["passage_id"]) for x in map(json.loads, (OV / "positions.jsonl").open())}
W = [p for k, p in pos.items() if k not in done]
# two-sided after pilot+bridge are added (same status rule as the builder)
side = collections.defaultdict(lambda: collections.defaultdict(int))
for p in W: side[p["label"]][p["member_id"]] += 1 if p["machine_side"] == "FOR" else -1
two = {l for l, d in side.items() if sum(v > 0 for v in d.values()) >= 2 and sum(v < 0 for v in d.values()) >= 2}
npos = collections.Counter(p["member_id"] for p in W)
random.seed(23)
for p in W:
    p["two_sided"] = p["label"] in two
    c = None if p["p_for"] is None else max(p["p_for"], p["p_against"]); p["conf"] = c
    p["group"] = "later" if not p["two_sided"] else "A" if (c is None or c < 0.7) else "D" if (c < 0.9 and npos[p["member_id"]] <= 4) else "later"
by = collections.defaultdict(list)
for p in W:
    if p["two_sided"] and p["group"] == "later": by[p["label"]].append(p)
for l, ps in by.items():
    for p in random.sample(ps, min(8, len(ps))): p["group"] = "B"
send = sorted([p for p in W if p["group"] != "later"], key=lambda p: (p["label"], p["passage_id"]))
for i, p in enumerate(send): p["id"] = i
with (OUT / "worklist_older.jsonl").open("w") as f:
    for p in W: f.write(json.dumps(p) + "\n")
P = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "major", "minor", "text"]).set_index("passage_id")
meta = {f["label"]: f for f in F}
chunks, cur = [], []
for l in sorted({p["label"] for p in send}):      # never split an argument across chunks
    ps = [p for p in send if p["label"] == l]
    if cur and len(cur) + len(ps) > 300: chunks.append(cur); cur = []
    cur += ps
if cur: chunks.append(cur)
for n, ch in enumerate(chunks, 1):
    with (OUT / f"chunk_{n:02d}_input.jsonl").open("w") as f:
        for p in ch:
            r = P.loc[p["passage_id"]]; m = meta[p["label"]]
            f.write(json.dumps({"id": p["id"], "proposal": p["label"], "supporter_means": m.get("for_means") or "", "opponent_means": m.get("against_means") or "",
                                "debate": r.major + (" — " + r.minor if r.minor else ""), "passage": r.text[:3000]}) + "\n")
C = collections.Counter
S = {"positions_total_older": len(W), "by_source": C(p["source"] for p in W), "by_side": C(p["machine_side"] for p in W), "two_sided_positions": sum(p["two_sided"] for p in W),
     "two_sided_arguments": len(two), "confidence_unknown": sum(p["conf"] is None for p in W), "groups": C(p["group"] for p in W), "sent_for_reading": len(send),
     "sent_by_source": C(p["source"] for p in send), "sent_by_side": C(p["machine_side"] for p in send), "chunks": len(chunks), "chunk_sizes": [len(c) for c in chunks]}
json.dump(S, open(OUT / "worklist_summary.json", "w"), indent=1); print(json.dumps(S, indent=1))
