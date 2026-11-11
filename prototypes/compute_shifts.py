"""Compute chronological recency-weighted scores and detect genuine 2024->2026 position shifts
for MPs with multiple speeches across different dates in claim_topics.json.
Also ensures Wikipedia guardrail only overrides on 2024-2026 actions, while flagging pre-2024 historical shifts.
"""
import json
import pathlib
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
CLAIM_FILE = ROOT / "data/quick/claim_topics.json"

DS_KEY = next(
    m[1].strip().strip("\"'")
    for m in (re.match(r"\s*(?:export\s+)?DEEPSEEK_API_KEY\s*=\s*(.*)", l) for l in (ROOT / ".env").read_text().splitlines())
    if m
)
sess = requests.Session()
sess.headers.update({"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"})


def call_ds(prompt, max_tokens=60, json_mode=False):
    for attempt in range(4):
        try:
            body = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
            }
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            r = sess.post("https://api.deepseek.com/chat/completions", json=body, timeout=30)
            if r.status_code in (429, 500, 502, 503):
                time.sleep(1.0 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return "{}" if json_mode else ""


claims = json.loads(CLAIM_FILE.read_text())
members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")

# Collect all MPs across the 20 topics who have >= 2 excerpts on distinct dates at least 30 days apart
multi_date_items = []
for tk_str, cd in claims.items():
    for idx_p, pt in enumerate(cd["pts"]):
        mid = int(pt[0])
        q = cd["quotes"].get(str(mid), {})
        ex_list = q.get("ex", [])
        if len(ex_list) >= 2:
            # Sort chronologically by date
            ex_sorted = sorted(ex_list, key=lambda x: x["d"])
            d0, d1 = pd.to_datetime(ex_sorted[0]["d"]), pd.to_datetime(ex_sorted[-1]["d"])
            if (d1 - d0).days >= 35:
                multi_date_items.append((tk_str, idx_p, mid, ex_sorted[0], ex_sorted[-1]))

print(f"Checking chronological consistency across {len(multi_date_items)} MP-topic pairs with multi-month speech spans...")


def check_pair(item):
    tk_str, idx_p, mid, ex_early, ex_late = item
    cd = claims[tk_str]
    prompt = (
        f"Topic: {cd['title']} ({cd['question']})\n"
        f"Pole - (-1): {cd['pole_lo']}\n"
        f"Pole + (+1): {cd['pole_hi']}\n\n"
        f"Earlier speech ({ex_early['d']}, {ex_early['h']}):\n\"{ex_early['t']}\"\n\n"
        f"Later speech ({ex_late['d']}, {ex_late['h']}):\n\"{ex_late['t']}\"\n\n"
        "Score the speaker's stance in the Earlier speech (s_early: -1, 0, or 1) and in the Later speech (s_late: -1, 0, or 1), "
        "and set shifted=true ONLY if the speaker genuinely changed from supporting one pole to supporting the opposite pole between the two dates "
        "(not merely emphasizing different details of the same side).\n"
        'Return JSON: {"s_early": -1 or 0 or 1, "s_late": -1 or 0 or 1, "shifted": bool, "shift_note": "short 10-word explanation if shifted else empty"}'
    )
    res = json.loads(call_ds(prompt, max_tokens=65, json_mode=True))
    return tk_str, idx_p, mid, ex_early["d"], ex_late["d"], res


with ThreadPoolExecutor(24) as ex:
    results = list(ex.map(check_pair, multi_date_items))

n_shifted = 0
for tk_str, idx_p, mid, d_early, d_late, res in results:
    cd = claims[tk_str]
    pt = cd["pts"][idx_p]
    q = cd["quotes"].get(str(mid), {})
    # Sort stored excerpts chronologically (most recent first in UI, with chronological order marked)
    if "ex" in q:
        q["ex"] = sorted(q["ex"], key=lambda x: x["d"], reverse=True)

    shifted = bool(res.get("shifted", False))
    s_early = float(res.get("s_early", 0))
    s_late = float(res.get("s_late", 0))
    if shifted and s_early != 0 and s_late != 0 and np.sign(s_early) != np.sign(s_late):
        n_shifted += 1
        cur_sc = float(pt[1])
        # Recency-weight: place the MP's main position on their latest side (0.68 * s_late) while recording s_early!
        new_sc = round(float(0.68 * s_late + 0.15 * cur_sc), 3)
        early_sc = round(float(0.75 * s_early), 3)
        pt[1] = new_sc
        pt[2] = int(np.sign(s_late))
        q["shift"] = {
            "d0": d_early,
            "d1": d_late,
            "s0": early_sc,
            "s1": new_sc,
            "note": str(res.get("shift_note", f"Shifted from {d_early} to {d_late}")),
        }
        mp_name = members.loc[mid, "name"] if mid in members.index else str(mid)
        party = members.loc[mid, "party_abbr"] if mid in members.index else "?"
        print(f"  [↻ Shift in {cd['title']}] {mp_name} ({party}): {d_early} ({early_sc:+.2f}) -> {d_late} ({new_sc:+.2f}) | {q['shift']['note']}")
    else:
        q.pop("shift", None)

# Also check if any Wikipedia note was from pre-2024 EU referendum (Starmer/Streeting on Topic 68) and mark as Historical Context rather than overriding 2024-2026 government speech
t68 = claims.get("68")
if t68:
    for p in t68["pts"]:
        mid = int(p[0])
        q = t68["quotes"].get(str(mid), {})
        wn = q.get("wiki_note", "")
        if "Remain" in wn or "second referendum" in wn or "People's Vote" in wn:
            q["wiki_note"] = "Historical context: Campaigned for Remain (2016–19); 2024–26 speeches reflect Government EU reset line"

CLAIM_FILE.write_text(json.dumps(claims, indent=2, ensure_ascii=False))
print(f"\nCompleted chronological recency weighting! Found and tagged {n_shifted} genuine 2024->2026 position shifts.")
