"""Recount fuller-pipeline fights from the final labels and fresh dense judgments."""

from collections import Counter, defaultdict
import json

import pandas as pd

from fp2_dense_run import D, R, ROOT, check_sides, index_rows, read_jsonl

SIDES = ("mps_for", "mps_against", "mps_both_sides")


def status(n_for, n_against):
    if n_for >= 2 and n_against >= 2:
        return "fight"
    if min(n_for, n_against) == 1 and max(n_for, n_against) >= 2:
        return "one_short"
    return "one_sided"


def recount(unit, preliminary, rows, passages, members):
    by_member = defaultdict(list)
    dropped = []
    label = unit["label"]
    for row in sorted(rows, key=lambda row: row["passage_id"]):
        pid, verdict = row["passage_id"], row["verdict"]
        if verdict in ("NEITHER", "ERR"):
            dropped.append({"passage_id": pid, "reason": verdict})
        else:
            by_member[int(passages.at[pid, "member_id"])].append(
                {"passage_id": pid, "measure": label, "side": verdict}
            )

    positions = {side: [] for side in SIDES}
    for mid, claims in sorted(by_member.items()):
        net = sum(1 if claim["side"] == "FOR" else -1 for claim in claims)
        name, party = members.loc[mid, ["name", "party_abbr"]] if mid in members.index else (None, None)
        positions[SIDES[0] if net > 0 else SIDES[1] if net < 0 else SIDES[2]].append({
            "member_id": mid,
            "name": str(name) if pd.notna(name) else f"MP {mid}",
            "party": str(party) if pd.notna(party) else "?",
            "claims": claims,
            "stance": round(net / len(claims), 3),
            "focus": len(claims),
        })
    groups = [
        dict(group, from_fight=fight_id)
        for fight_id in unit["members"]
        for group in preliminary[fight_id]["groups"]
    ]
    return {
        "fight_id": unit["unit"],
        "topic": unit["topic"],
        "topics": unit["topics"],
        "label": label,
        "old_label": unit["old_label"],
        "description": unit["description"],
        "for_means": unit["for_means"],
        "against_means": unit["against_means"],
        "groups": groups,
        "merged_from": unit["members"] if unit["merged"] else None,
        "status": status(len(positions[SIDES[0]]), len(positions[SIDES[1]])),
        "dense": True,
        "n_dense_candidates": len(rows),
        "n_dropped": len(dropped),
        "dropped": dropped,
        **positions,
    }


def main():
    units = json.loads((D / "units.json").read_text())
    _c = read_jsonl(D / "dense_cands.jsonl"); _s = {(r["unit"], r["passage_id"]) for r in read_jsonl(D / "dense_sides.jsonl")}
    _need = Counter(r["unit"] for r in _c); _have = Counter(r["unit"] for r in _c if (r["unit"], r["passage_id"]) in _s)
    skipped = [u for u in units if _have[u["unit"]] < _need[u["unit"]]]
    units = [u for u in units if _have[u["unit"]] == _need[u["unit"]]]
    (R / "fp2_skipped_units.json").write_text(json.dumps([{"unit": u["unit"], "label": u["label"], "n_mps": u["n_mps"], "scored": _have[u["unit"]], "cands": _need[u["unit"]]} for u in skipped], indent=1))
    print("units scored", len(units), "skipped", len(skipped), flush=True)
    preliminary = {f["fight_id"]: f for f in json.loads((D / "fights_pre.json").read_text())}
    labels = {unit["unit"]: unit["label"] for unit in units}
    candidate_path, sides_path = D / "dense_cands.jsonl", D / "dense_sides.jsonl"
    keep_units = {u["unit"] for u in units}
    candidates = index_rows([r for r in read_jsonl(candidate_path) if r["unit"] in keep_units], candidate_path)
    sides = index_rows([r for r in read_jsonl(sides_path) if r["unit"] in keep_units], sides_path)
    check_sides(candidates, sides, complete=True)
    rows_by_unit = defaultdict(list)
    for row in sides.values():
        if labels.get(row["unit"]) != row["label"]:
            raise ValueError(f"Dense candidate label does not match the final unit: {row['unit']}")
        rows_by_unit[row["unit"]].append(row)
    report = json.loads((D / "dense_cands_report.json").read_text())
    report_counts = {row["unit"]: row["n"] for row in report if row["unit"] in keep_units}
    counts = {unit: len(rows_by_unit[unit]) for unit in labels}
    if report_counts != counts:
        raise ValueError("Dense candidate counts do not match the report for every unit")

    passages = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "member_id"])
    passages = passages.drop_duplicates("passage_id").set_index("passage_id")
    pids = {pid for _, pid in sides}
    if not pids.issubset(passages.index) or passages.loc[sorted(pids), "member_id"].isna().any():
        raise ValueError("Some dense candidates lack source passages or member IDs")
    members = pd.read_parquet(ROOT / "data/clean/members.parquet", columns=["member_id", "name", "party_abbr"])
    members = members.drop_duplicates("member_id").set_index("member_id")
    all_fights = [recount(unit, preliminary, rows_by_unit[unit["unit"]], passages, members) for unit in units]
    fights = [f for f in all_fights if len(f[SIDES[0]]) + len(f[SIDES[1]]) >= 3]

    claims = read_jsonl(D / "claims.jsonl")
    measures = pd.read_json(D / "measures.jsonl", lines=True, dtype={"passage_id": str})
    by_topic = defaultdict(Counter)
    for f in fights:
        by_topic[f["topic"]][f["status"]] += 1
    topics = {}
    for row in claims:
        entry = topics.setdefault(row["topic"], {"passages": 0, "claims": 0, "groups": 0})
        entry["passages"] += 1
        entry["claims"] += row["stance"] in ("FOR", "AGAINST")
    for topic, n in measures.groupby("topic")["gid"].nunique().items():
        topics[topic]["groups"] = int(n)
    for topic, entry in topics.items():
        entry["fights_by_status"] = dict(by_topic[topic])
    by_status = dict(Counter(f["status"] for f in fights))
    totals = {
        "passages_extracted": len({row["passage_id"] for row in claims}),
        "claims": sum(entry["claims"] for entry in topics.values()),
        "groups": int(measures["gid"].nunique()),
        "fights_pre": len(preliminary),
        "fights_total": len(units),
        "fights_reported": len(fights),
        "by_status": by_status,
        "dense_by_status": by_status,
        "dense_candidates": len(sides),
        "claims_judged": sum(row["verdict"] != "ERR" for row in sides.values()),
        "passages_judged": len({row["passage_id"] for row in sides.values() if row["verdict"] != "ERR"}),
        "dense_verdicts": dict(Counter(row["verdict"] for row in sides.values())),
    }
    cost = {}
    for stage, filename in (
        ("merge", "fp2_merge_tokens.json"),
        ("relabel", "fp2_labels.json"),
        ("xmerge", "fp2_xmerge.json"),
        ("dense", "fp2_dense_tokens.json"),
    ):
        path = R / filename
        if path.exists():
            accounting = json.loads(path.read_text())
            cost[stage] = {k: accounting[k] for k in ("tokens", "usd") if k in accounting}
    result = {"totals": totals, "per_topic": topics, "cost": cost, "fights": fights}
    (R / "fp2_fights.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()
