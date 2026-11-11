"""Pilot of the agenda pipeline: 20 proposals written from the audit's topic evidence packs, each tested on fresh passages.

Topics: policy topics (not procedural, not already covered by the 27 existing proposals), taken in order of how many MPs
speak on them (data/quick/agenda_audit/pilot_topics.json). Walking down that list, a proposal was written only where a
topic's evidence pack showed a concrete action people take sides on; the rest are recorded as "no proposal".
Proposals were written in-session by Claude (Opus 5.5) from the packs, citing a passage; none of the cited passages is scored.

  collect   find fresh passages for each proposal (not from the writer packs), one per MP, up to 30
  score     ask Jev: is it about the proposal, does the speaker support it, oppose it
  report    which proposals have clear speakers on both sides
"""
import json
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data/quick/agenda_audit"
N, SEED, MIN_SIDE = 30, 7, 4
SPLIT = re.compile(r"(?<=[.!?])\s+")

# (id, topic, label, what the speaker would support, search pattern, cited passage)
PROPOSALS = [
    ("vat_private_schools", 39, "Charge VAT on private school fees", "charging VAT on private school fees",
     r"VAT[^.]{0,60}(private|independent) schools?|(private|independent) school\w*[^.]{0,40}VAT|tax breaks?[^.]{0,30}private schools?", "2025-03-03a.29.0#0"),
    ("_obr_once", 10, "Have the OBR check the fiscal rules once a year instead of twice", "the OBR assessing the fiscal rules only once a year instead of twice",
     r"(OBR|Office for Budget Responsibility)[^.]{0,120}(once a year|once per year|one assessment|two to one|twice)|one (major )?(annual )?fiscal event", "2025-12-03g.997.3#0"),
    ("employer_nics", 19, "Keep the rise in employer National Insurance", "the increase in employer National Insurance contributions",
     r"employers?'?'?s?'? national insurance|national insurance contributions? (rise|increase|hike)|jobs tax", "2025-06-19a.634.2#1"),
    ("rent_cap", 38, "Cap how much rents can rise during a tenancy", "capping or controlling rent increases (rent controls)",
     r"rent controls?|cap(ping)? (on )?rents?\b|rent (increase|rise)s?[^.]{0,20}cap|limit\w*[^.]{0,20}rent increases", "standing5301_RENTERS'_03"),
    ("day_one_dismissal", 42, "Protect workers from unfair dismissal from their first day", "giving employees protection from unfair dismissal from day one of employment",
     r"unfair dismissal[^.]{0,80}(day one|first day|qualifying period)|(day one|day-one)[^.]{0,40}(rights|unfair dismissal|protection)|qualifying period[^.]{0,40}(dismissal|employment)", "standing5339_EMPLOYMENT_10"),
    ("eu_ai_act", 52, "Align UK tech rules with the EU's AI Act", "the UK aligning its technology or AI regulation with the EU's rules such as the AI Act",
     r"\bAI Act\b|EU tech rules|align\w*[^.]{0,40}EU[^.]{0,30}(AI|digital|tech)", "2026-05-20c.534.8#0"),
    ("rail_public", 64, "Bring passenger railways back into public ownership", "bringing passenger railways into public ownership",
     r"(rail\w*|train operat\w+)[^.]{0,60}(public ownership|nationalis\w+)|(public ownership|nationalis\w+)[^.]{0,60}(rail\w*|train)", "2024-09-11e.821.3#0"),
    ("mayors", 24, "Create elected mayors across all of England", "creating elected mayors and mayoral combined authorities across England",
     r"mayoral (combined )?(county )?authorit\w+|elected mayors?\b|devolution deals?", "2025-04-02c.296.2#0"),
    ("football_regulator", 36, "Create an independent football regulator", "creating an independent regulator for football",
     r"football regulator|independent regulator[^.]{0,30}football|Football Governance Bill", "2025-07-08d.896.2#0"),
    ("votes_16", 15, "Lower the voting age to 16", "lowering the voting age to 16",
     r"votes at 16|voting age|16[- ]and 17[- ]year[- ]olds?[^.]{0,40}vot", "standing5615_REPRESENTAT"),
    ("pr", 15, "Replace first past the post with proportional representation", "replacing first past the post with proportional representation",
     r"proportional representation|first past the post", "2025-07-14d.18.5#0"),
    ("bus_franchise", 55, "Let councils take control of local buses", "letting councils franchise or run their own bus services",
     r"bus franchising|franchis\w+[^.]{0,40}bus|municipal bus|council-owned bus|local authority bus compan", "2026-01-07d.346.0#0"),
    ("planning_override", 94, "Approve new homes faster even over local objections", "planning changes so new homes and infrastructure get approved faster even over local objections",
     r"Planning and Infrastructure Bill|grey belt|planning (reforms?|rules?)[^.]{0,60}(homes|housing|build)", "2025-03-26b.974.3#0"),
    ("farm_iht", 76, "Charge inheritance tax on farms worth over £1 million", "charging inheritance tax on farms (the changes to agricultural property relief)",
     r"(farm\w*|agricultur\w+)[^.]{0,80}inheritance tax|inheritance tax[^.]{0,80}(farm|agricultur)|family farm tax|agricultural property relief", "2025-02-26c.809.0#0"),
    ("minimum_wage", 82, "Keep raising the minimum wage faster than inflation", "raising the national minimum or living wage substantially",
     r"(minimum|living) wage[^.]{0,60}(increase|rise|uplift|rais)|(increase|rise|uplift|rais)\w*[^.]{0,40}(minimum|living) wage", "2025-03-13a.1245.1#0"),
    ("borrow_more", 68, "Borrow more to fund public spending", "the government borrowing more to fund spending",
     r"borrow(ing)? more|more borrowing|increas\w+ borrowing|borrowing to (fund|pay)", "2026-04-28d.723.0#0"),
    ("water_public", 68, "Bring water companies into public ownership", "bringing water companies into public ownership",
     r"(water compan\w+|Thames Water|water industry)[^.]{0,80}(public ownership|nationalis\w+|special administration)|(public ownership|nationalis\w+)[^.]{0,60}water", "2025-03-28b.1224.4#0"),
    ("bank_branches", 18, "Require banks to keep a branch or banking hub in every town", "requiring banks to keep local branches or banking hubs open",
     r"banking hubs?|bank branch(es)? clos\w+|branch closures?", "2024-12-11c.363.2#0"),
    ("unitary", 103, "Merge district councils into larger unitary councils", "abolishing district councils and merging them into larger unitary councils",
     r"local government reorganisation|unitary (authorit\w+|council\w*)|abolish\w*[^.]{0,30}district councils?", "2025-02-05b.772.4#0"),
    ("green_levies", 25, "Remove green levies from energy bills", "removing green or social levies from energy bills",
     r"(green|social and environmental|policy) levies|levies (on|from) (energy |electricity )?bills", "2026-03-24c.133.5#0"),
]
NO_PROPOSAL = {35: "integrated care", 81: "knife crime", 107: "heritage", 4: "dispatch box (procedural)", 97: "contracts", 7: "noise words",
               65: "manufacturing", 2: "sexual abuse", 22: "roads", 12: "Ukraine", 73: "Parkinson's", 63: "suicide prevention",
               37: "GPs", 69: "infected blood", 27: "sewage", 86: "pensions", 59: "fire safety", 96: "maternity", 11: "Sudan",
               87: "armed forces", 58: "Iran", 17: "cancer", 57: "China", 61: "temporary accommodation"}


def snippet(text, pattern):
    sents = SPLIT.split(text)
    hit = [i for i, s in enumerate(sents) if re.search(pattern, s, re.I)]
    if not hit:
        return None
    keep = sorted({j for i in hit[:1] for j in (i - 1, i, i + 1) if 0 <= j < len(sents)})
    return " ".join(sents[j] for j in keep)


def collect(n_for=None):
    """n_for: {proposal: sample size}; default N for every proposal."""
    packs = {json.loads(l)["topic"]: json.loads(l) for l in open(OUT / "coherent_writer_requests.jsonl")}
    writer_ids = {e["id"] for x in packs.values() for e in (x["evidence_metadata"] if isinstance(x["evidence_metadata"], list) else eval(x["evidence_metadata"]))}
    d = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "member_id", "date", "major", "minor", "role", "text"])
    d = d[~d.passage_id.isin(writer_ids)]
    rows = []
    for pid_, topic, label, ask, pat, cite in PROPOSALS:
        if pid_.startswith("_"): continue
        m = d[d.text.str.contains(pat, case=False, regex=True)]
        m = m.sample(frac=1, random_state=SEED).drop_duplicates("member_id").head((n_for or {}).get(pid_, N))   # one passage per MP, spread across MPs
        for r in m.itertuples():
            s = snippet(r.text, pat)
            if s: rows.append({"proposal": pid_, "pid": r.passage_id, "member": r.member_id, "date": r.date, "role": r.role,
                               "debate": f"{r.date}|{r.major}|{r.minor}", "snip": s})
        print(f"{pid_:20} {len(d[d.text.str.contains(pat, case=False, regex=True)]):5} matching passages, {len(m)} sampled")
    pd.DataFrame(rows).to_parquet(OUT / "pilot_snippets.parquet", index=False)
    json.dump({"rule": "walk down pilot_topics order; write a proposal only where the evidence pack shows a concrete action people take sides on",
               "writer": "Claude (Opus 5.5), in-session, from coherent_writer_requests.jsonl packs; cited passages are excluded from scoring",
               "proposals": [dict(zip(["id", "topic", "label", "ask", "pattern", "cited"], p)) for p in PROPOSALS],
               "no_proposal": {str(k): v for k, v in NO_PROPOSAL.items()}}, open(OUT / "pilot_proposals.json", "w"), indent=1)


def score():
    sn = pd.read_parquet(OUT / "pilot_snippets.parquet")
    f = OUT / "pilot_scores.parquet"
    done = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["pid", "proposal"])
    todo = sn.merge(done[["pid", "proposal"]], on=["pid", "proposal"], how="left", indicator=True)
    todo = todo[todo._merge == "left_only"]
    ask_of = {p[0]: p[3] for p in PROPOSALS}
    if "employer_nics" in set(done.proposal) and "--rescore-nics" in sys.argv: done = done[done.proposal != "employer_nics"]
    key = next(m[1].strip().strip("\"'") for m in (re.match(r"\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)", l) for l in (ROOT / ".env").read_text().splitlines()) if m)
    sess = requests.Session()
    sess.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    used = {"tokens": 0, "failed": 0}

    def ask(r):
        a = ask_of[r.proposal]
        qs = {"about": {"type": "noul", "instructions": f"Is this passage about {a}?"},
              "support": {"type": "noul", "instructions": f"Does the speaker themselves express support for {a}?"},
              "oppose": {"type": "noul", "instructions": f"Does the speaker themselves express opposition to {a}?"}}
        for attempt in range(6):
            try:
                resp = sess.post("https://api.typesafe.ai/v1/systemone", json={"model": "jev-1.14.0", "state": r.snip, "questions": qs}, timeout=20)
            except requests.RequestException:
                time.sleep(2 ** attempt); continue
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(float(resp.headers.get("retry-after") or 2 ** attempt)); continue
            resp.raise_for_status()
            j = resp.json()
            used["tokens"] += j.get("usage", {}).get("input_tokens", 0)
            return {"pid": r.pid, "proposal": r.proposal, **{k: v["noul"] for k, v in j["answers"].items()}}
        used["failed"] += 1

    start = time.time()
    with ThreadPoolExecutor(8) as ex:
        res = [x for x in ex.map(ask, todo.itertuples()) if x]
    pd.concat([done, pd.DataFrame(res)], ignore_index=True).to_parquet(f, index=False)
    print(f"{len(res)} scored in {time.time() - start:.0f}s | tokens {used['tokens']} | cost ${used['tokens'] * 0.042 / 1e6:.4f} | failed {used['failed']}")


def report():
    sn = pd.read_parquet(OUT / "pilot_snippets.parquet")
    d = sn.merge(pd.read_parquet(OUT / "pilot_scores.parquet"), on=["pid", "proposal"])
    d["about_"] = d.about >= .5
    d["sup"] = d.about_ & (d.support >= .5) & (d.oppose < .5)
    d["opp"] = d.about_ & (d.oppose >= .5) & (d.support < .5)
    d["both"] = d.about_ & (d.oppose >= .5) & (d.support >= .5)
    meta = {p[0]: p for p in PROPOSALS}
    d = d[d.proposal.isin(meta)]
    out = []
    for pid_, g in d.groupby("proposal"):
        bb = g[g.role != "minister"]
        out.append({"id": pid_, "label": meta[pid_][2], "topic": meta[pid_][1], "sampled": len(g), "about": int(g.about_.sum()),
                    "support": int(g.sup.sum()), "oppose": int(g.opp.sum()), "ambiguous": int(g.both.sum()),
                    "support_backbench": int(bb.sup.sum()), "oppose_backbench": int(bb.opp.sum()),
                    "survives": bool(g.sup.sum() >= MIN_SIDE and g.opp.sum() >= MIN_SIDE)})
    out.sort(key=lambda x: (-x["survives"], -min(x["support"], x["oppose"])))
    json.dump({"min_each_side": MIN_SIDE, "per_proposal": out, "no_proposal": {str(k): v for k, v in NO_PROPOSAL.items()}}, open(OUT / "pilot_report.json", "w"), indent=1)
    for x in out:
        print(f"{'KEEP' if x['survives'] else '    '} {x['label'][:52]:52} about {x['about']:2}/{x['sampled']:2}  for {x['support']:2}  against {x['oppose']:2}  both {x['ambiguous']}")


def expand(n=90):
    """Stage 2: grow the sample for every proposal that is mostly on-topic but not yet clearly two-sided."""
    rep_ = json.load(open(OUT / "pilot_report.json"))["per_proposal"]
    grow = {x["id"]: n for x in rep_ if not x["survives"] and x["about"] >= x["sampled"] / 2}
    grow["employer_nics"] = n
    print("expanding", sorted(grow))
    collect(grow)


if __name__ == "__main__":
    {"collect": collect, "score": score, "report": report, "expand": expand}[sys.argv[1]]()
