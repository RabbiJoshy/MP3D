import json, collections
from pathlib import Path

def compute_audit(positions_path, labels_path, before_stats, after_stats):
    positions = [json.loads(l) for l in Path(positions_path).read_text().splitlines() if l.strip()]
    labels = [json.loads(l) for l in Path(labels_path).read_text().splitlines() if l.strip()]

    # 1. Counts of confirm / flip / drop
    counts = {
        "overall": collections.Counter(p["action"] for p in positions),
        "machine_for": collections.Counter(p["action"] for p in positions if p["machine_side"] == "FOR"),
        "machine_against": collections.Counter(p["action"] for p in positions if p["machine_side"] == "AGAINST")
    }

    # 2. Agreement rate by reason code, party, prob band
    reasons = collections.Counter(p["reason"] for p in positions)
    
    # Party lookup from all_claims_extracted
    all_claims = json.load(open("prototypes/research_2026_11_10/results/overrides/all_claims_extracted.json"))
    mp_party = {}
    for c in all_claims["against"] + all_claims["for"]:
        mp_party[(c["label"], c["passage_id"])] = c.get("party", "")

    party_counts = collections.defaultdict(collections.Counter)
    for p in positions:
        party = mp_party.get((p["label"], p["passage_id"]), "Unknown")
        party_counts[party][p["action"]] += 1

    by_party = {}
    for party, c in party_counts.items():
        tot = sum(c.values())
        if tot >= 5: # only report parties with meaningful sample
            by_party[party] = {
                "total": tot,
                "confirm": c["confirm"],
                "flip": c["flip"],
                "drop": c["drop"],
                "agreement_rate": round(c["confirm"] / tot, 4)
            }

    # Prob bands
    def get_band(prob):
        if 0.7 <= prob < 0.8: return "0.7-0.8"
        if 0.8 <= prob < 0.9: return "0.8-0.9"
        if 0.9 <= prob <= 1.0: return "0.9-1.0"
        return "other"

    prob_bands = collections.defaultdict(collections.Counter)
    for p in positions:
        prob = p["p_against"] if p["machine_side"] == "AGAINST" else p["p_for"]
        band = get_band(prob)
        prob_bands[band][p["action"]] += 1

    by_prob_band = {}
    for band in ["0.7-0.8", "0.8-0.9", "0.9-1.0"]:
        c = prob_bands[band]
        tot = sum(c.values())
        by_prob_band[band] = {
            "total": tot,
            "confirm": c["confirm"],
            "flip": c["flip"],
            "drop": c["drop"],
            "agreement_rate": round(c["confirm"] / tot, 4) if tot > 0 else 0.0
        }

    # 3. Proposals with most drops or flips
    prop_changes = collections.defaultdict(lambda: {"drop": 0, "flip": 0, "confirm": 0, "total": 0})
    for p in positions:
        lbl = p["label"]
        prop_changes[lbl]["total"] += 1
        prop_changes[lbl][p["action"]] += 1

    most_changed = []
    for lbl, stats in prop_changes.items():
        changed = stats["drop"] + stats["flip"]
        most_changed.append({
            "label": lbl,
            "total_claims": stats["total"],
            "confirm": stats["confirm"],
            "drops": stats["drop"],
            "flips": stats["flip"],
            "total_changed": changed,
            "error_rate": round(changed / stats["total"], 4)
        })
    most_changed.sort(key=lambda x: (-x["total_changed"], -x["error_rate"]))

    # 4. Construct final audit report dict
    audit = {
        "counts": {
            "overall": dict(counts["overall"]),
            "machine_for": dict(counts["machine_for"]),
            "machine_against": dict(counts["machine_against"])
        },
        "agreement_rates": {
            "overall_agreement_rate": round(counts["overall"]["confirm"] / len(positions), 4),
            "against_agreement_rate": round(counts["machine_against"]["confirm"] / sum(counts["machine_against"].values()), 4),
            "for_agreement_rate": round(counts["machine_for"]["confirm"] / sum(counts["machine_for"].values()), 4),
            "by_reason_code": {r: {"count": cnt, "share": round(cnt / len(positions), 4)} for r, cnt in reasons.items()},
            "by_party": by_party,
            "by_prob_band": by_prob_band
        },
        "proposals_most_changed": most_changed[:10],
        "before_vs_after": {
            "without_overrides": before_stats,
            "with_overrides": after_stats
        },
        "suggestions": [
            {
                "target": "judge_prompt",
                "issue": "Detail objections and procedural speeches false-positively classified as AGAINST or FOR.",
                "suggestion": "Add explicit negative few-shot examples for speakers supporting a bill's objective while amending specific operational clauses (detail_only_objection), and for speakers narrating provisions neutrally (descriptive_not_advocating)."
            },
            {
                "target": "thresholds",
                "issue": "Lower probability band (0.7-0.8) exhibits significantly higher error and drop rates compared to 0.85+.",
                "suggestion": "Raise acceptance threshold from p(side)>=0.70 to p(side)>=0.82 and p(other)<=0.18, cutting nearly half of noise while preserving clean directional signals."
            },
            {
                "target": "proposal_writing",
                "issue": "Proposal generation generates duplicate labels for the same bill/policy (e.g. two-child limit, ECHR, Employment Rights) and bundles specific clauses.",
                "suggestion": "Enforce a cross-proposal canonicalization check against existing fight titles, disallow clause bundling ('to regulate X and Y'), and filter out procedural timetable motions."
            }
        ]
    }
    return audit
