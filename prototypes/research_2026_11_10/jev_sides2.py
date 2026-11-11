"""Jev sides v2: generic FOR/AGAINST/NEITHER judge on hand-checked edu claims. Prompt fixed before running (single test)."""
import re, json, pathlib, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
key = next(m[1].strip().strip("\"'") for l in (ROOT / ".env").read_text().splitlines() if (m := re.match(r"\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)", l)))
S = requests.Session(); S.headers.update({"Authorization": f"Bearer {key}"}); used = [0]
HERE = ROOT / "prototypes/research_2026_11_10"
hc = json.load(open(HERE / "results/edu_handcheck.json"))
P = pd.read_parquet(ROOT / "data/clean/passages.parquet").set_index("passage_id").text
PROMPT_FOR = ("Measure: \"{m}\". Answer yes only if the speaker themself clearly wants this measure to happen. "
 "Answer no if the speaker objects only to a detail, cost, burden or timing, or seeks an exemption or amendment while accepting the measure; "
 "only reports others' views; merely describes rather than advocates; or the passage is about a different measure. "
 "A speaker defending a policy against an amendment is FOR that policy.")
PROMPT_AGAINST = ("Measure: \"{m}\". Answer yes only if the speaker themself clearly wants this measure NOT to happen at all. "
 "Answer no if the speaker objects only to a detail, cost, burden or timing, or seeks an exemption or amendment while accepting the measure; "
 "only reports others' views; merely describes rather than advocates; or the passage is about a different measure. "
 "A speaker defending a policy against an amendment is FOR that policy, not against it.")
def ask(r):
    q = {"for": {"type": "noul", "instructions": PROMPT_FOR.format(m=r["label"])},
         "against": {"type": "noul", "instructions": PROMPT_AGAINST.format(m=r["label"])}}
    for _ in range(6):
        try:
            x = S.post("https://api.typesafe.ai/v1systemone".replace("v1systemone", "v1/systemone"), json={"model": "jev-1.14.0", "state": r["text"][:3000], "questions": q}, timeout=60)
            if x.status_code == 200:
                d = x.json(); used[0] += d["usage"]["input_tokens"]; return d["answers"]["for"]["noul"], d["answers"]["against"]["noul"]
        except requests.RequestException: pass
    return None, None
rows = []
for g in hc["groups"]:
    for c in g["claims"]:
        rows.append(dict(passage_id=c["passage_id"], mp=c["mp"], party=c["party"], group=g["label"], label=g["label"], label_side=c["label_side"],
                         same_measure=c["same_measure"], side_correct=c["side_correct"], text=P[c["passage_id"]]))
with ThreadPoolExecutor(8) as ex: res = list(ex.map(ask, rows))
def verdict(f, a):
    if f is None: return "ERR"
    if f >= .5 and f > a: return "FOR"
    if a >= .5 and a > f: return "AGAINST"
    return "NEITHER"
for r, (f, a) in zip(rows, res): r["p_for"], r["p_against"], r["new_verdict"] = f, a, verdict(f, a)
opp = {"FOR": "AGAINST", "AGAINST": "FOR"}
gk = fl = dr = bc = bw = 0; misses = []
for r in rows:
    r["good"] = bool(r["same_measure"]) and r["side_correct"] == "yes"; v = r["new_verdict"]
    if r["good"]:
        if v == r["label_side"]: gk += 1
        elif v == opp[r["label_side"]]: fl += 1; misses.append(r)
        else: dr += 1; misses.append(r)
    else:
        if v == r["label_side"]: bw += 1; misses.append(r)
        else: bc += 1
ng = sum(r["good"] for r in rows)
groups = {}
for g in hc["groups"]:
    kept = [r for r in rows if r["group"] == g["label"] and r["new_verdict"] == r["label_side"]]
    f = len({r["mp"] for r in kept if r["label_side"] == "FOR"}); a = len({r["mp"] for r in kept if r["label_side"] == "AGAINST"})
    groups[g["label"]] = dict(mps_for=f, mps_against=a, contested=f >= 2 and a >= 2)
cost = used[0] * 0.042 / 1e6
summary = dict(n=len(rows), n_good=ng, good_kept_right=gk, good_flipped=fl, good_dropped=dr, n_bad=len(rows) - ng, bad_rejected=bc, bad_wrongly_kept=bw, groups=groups, input_tokens=used[0], est_cost_usd=round(cost, 5))
print(json.dumps(summary, indent=1))
for r in misses: print(f"MISS {r['mp']} | {r['group'][:40]} | hand good={r['good']} side_correct={r['side_correct']} same={r['same_measure']} label={r['label_side']} | judge={r['new_verdict']}")
out = dict(prompt_for=PROMPT_FOR, prompt_against=PROMPT_AGAINST, decision_rule="FOR if p_for>=.5 and p_for>p_against; AGAINST if p_against>=.5 and p_against>p_for; else NEITHER",
           summary=summary, rows=[{k: v for k, v in r.items() if k not in ("text", "label")} for r in rows])
json.dump(out, open(HERE / "results/edu_jev2_test.json", "w"), indent=1)
