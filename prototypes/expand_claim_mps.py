"""Expand MP coverage on all 20 unsupervised Claim-Anchor topics in data/quick/claim_topics.json
beyond the initial 80-MP cap (up to 180 MPs per topic).
Only scores MPs not already in cd["pts"], preserving all existing audited scores, Wikipedia checks, and shifts.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import pathlib
import re
import time

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
CLAIM_PATH = ROOT / "data/quick/claim_topics.json"

DS_KEY = next(
    m[1].strip().strip("\"'")
    for m in (re.match(r"\s*(?:export\s+)?DEEPSEEK_API_KEY\s*=\s*(.*)", l) for l in (ROOT / ".env").read_text().splitlines())
    if m
)
ds_sess = requests.Session()
ds_sess.headers.update({"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"})

passages = pd.read_parquet(
    ROOT / "data/clean/passages.parquet",
    columns=["passage_id", "member_id", "role", "date", "major", "minor", "text", "words"],
)
ts = np.load(ROOT / "data/quick/topic_space.npz", allow_pickle=True)
passages["topic"] = ts["topic"]
passages["sim"] = ts["sim"]
members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
known = set(members.index)

bge = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True)
PROCEDURAL_RE = re.compile(r"\b(new clause|amendment \d+|clause \d+ stand part|commencement|schedule \d+|third reading)\b", re.I)


def score_additional_mps(cd, mp_bundles):
    if not mp_bundles:
        return {}
    out = {}
    items = list(mp_bundles.items())
    batches = [items[i : i + 15] for i in range(0, len(items), 15)]

    def one_batch(batch):
        lines = [f"MP_{mid} ({info['name']}, {info['party']}):\n{info['bundle']}" for mid, info in batch]
        prompt = (
            f"Topic: {cd['title']}\n"
            f"Core Belief Question: {cd['question']}\n"
            f"Pole - : {cd['pole_lo']}\n"
            f"Pole + : {cd['pole_hi']}\n\n"
            "For each MP below, read their House of Commons speeches and state their underlying CORE POLICY BELIEF "
            "on this topic in one clear sentence (10-18 words), independent of procedural bill mechanics.\n"
            "Also classify side: -1 (aligns with Pole -), +1 (aligns with Pole +), or 0 (purely neutral/procedural/local/unclear).\n"
            'Return ONLY valid JSON: {"MP_<id>": {"claim": "...", "side": -1|0|1}}\n\n'
            + "\n\n".join(lines)
        )
        for attempt in range(4):
            try:
                r = ds_sess.post(
                    "https://api.deepseek.com/chat/completions",
                    json={
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 1300,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=55,
                )
                if r.status_code == 200:
                    txt = r.json()["choices"][0]["message"]["content"]
                    parsed = json.loads(txt)
                    res = {}
                    for k, v in parsed.items():
                        digits = re.sub(r"\D", "", k)
                        if digits:
                            res[int(digits)] = v
                    return res
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        return {}

    with ThreadPoolExecutor(12) as ex:
        for res in ex.map(one_batch, batches):
            out.update(res)
    return out


claim_data = json.loads(CLAIM_PATH.read_text())
total_added = 0

for tk_str, cd in claim_data.items():
    tk = int(tk_str)
    existing_mids = {int(p[0]) for p in cd["pts"]}
    if len(existing_mids) >= 150:
        print(f"Skipping [{tk}] {cd['title']} (already has {len(existing_mids)} MPs)", flush=True)
        continue

    sub = passages[(passages.topic == tk) & (passages.words >= 45) & (passages.sim >= 0.27) & (passages.member_id.isin(known))].copy()
    is_proc = sub["text"].str.contains(PROCEDURAL_RE)
    sub["weight"] = sub["sim"] * (np.minimum(sub["words"], 220) ** 0.35) * np.where(is_proc, 0.55, 1.0)

    # Find MPs not yet in existing_mids who have >= 2 speeches on this topic (or >= 1 speech with >= 90 words)
    rem = sub[~sub.member_id.isin(existing_mids)]
    mp_w = rem.groupby("member_id")["weight"].sum().sort_values(ascending=False)
    mp_cnt = rem.groupby("member_id").size()
    mp_maxw = rem.groupby("member_id")["words"].max()

    cand_mids = [
        int(m)
        for m in mp_w.index
        if mp_cnt[m] >= 2 or mp_maxw[m] >= 90
    ][:105]

    mp_bundles = {}
    for mid in cand_mids:
        grp = rem[rem.member_id == mid]
        top3 = grp.sort_values("weight", ascending=False).head(3).sort_values("date")
        ex_strs = []
        ex_list = []
        for r in top3.itertuples():
            hd = f"{r.major} ({r.minor})" if r.minor else r.major
            ex_strs.append(f"[{r.date}, {hd}]: {r.text[:420]}")
            ex_list.append({"t": r.text[:280], "d": r.date, "h": r.minor or r.major})
        mp_bundles[mid] = {
            "name": members.loc[mid, "name"],
            "party": members.loc[mid, "party_abbr"] or "Ind",
            "n_sp": int(len(grp)),
            "words": int(grp["words"].sum()),
            "bundle": "\n".join(ex_strs),
            "ex": ex_list,
            "latest": ex_list[-1],
        }

    print(f"Expanding [{tk}] {cd['title']}: scoring {len(mp_bundles)} additional MPs beyond current {len(existing_mids)}...", flush=True)
    new_claims = score_additional_mps(cd, mp_bundles)

    lo_texts = [f"{cd['title']}: {cd['pole_lo']}"] + [e["c"] for e in cd.get("ex_lo", [])[:4]]
    hi_texts = [f"{cd['title']}: {cd['pole_hi']}"] + [e["c"] for e in cd.get("ex_hi", [])[:4]]
    v_lo = normalize(bge.encode(lo_texts, normalize_embeddings=True).mean(0, keepdims=True))[0]
    v_hi = normalize(bge.encode(hi_texts, normalize_embeddings=True).mean(0, keepdims=True))[0]
    axis_vec = v_hi - v_lo
    axis_vec /= np.linalg.norm(axis_vec) + 1e-9

    valid_mids = [
        m for m, c in new_claims.items()
        if m in mp_bundles and isinstance(c, dict) and c.get("side", 0) != 0 and c.get("claim")
    ]
    added_here = 0
    if valid_mids:
        c_texts = [f"{cd['title']}: {new_claims[m]['claim']}" for m in valid_mids]
        c_embs = bge.encode(c_texts, normalize_embeddings=True)
        raw_proj = c_embs @ axis_vec
        raw_proj = (raw_proj - np.mean(raw_proj)) / (np.std(raw_proj) + 1e-9)
        for idx_m, m in enumerate(valid_mids):
            side = int(new_claims[m]["side"])
            sc = round(float(np.clip(0.55 * side + 0.45 * np.clip(raw_proj[idx_m] * 0.55, -0.9, 0.9), -1.15, 1.15)), 3)
            info = mp_bundles[m]
            cd["pts"].append([m, sc, 1 if sc > 0 else -1, info["n_sp"], info["words"]])
            cd["quotes"][str(m)] = {
                "c": new_claims[m]["claim"],
                "t": info["latest"]["t"],
                "d": info["latest"]["d"],
                "h": info["latest"]["h"],
                "ex": info["ex"],
            }
            added_here += 1

    cd["n_lo"] = sum(1 for p in cd["pts"] if float(p[1]) < -0.15)
    cd["n_hi"] = sum(1 for p in cd["pts"] if float(p[1]) > 0.15)
    total_added += added_here
    CLAIM_PATH.write_text(json.dumps(claim_data, indent=2, ensure_ascii=False))
    print(f"  -> Added +{added_here} MPs to [{tk}] {cd['title']} (now {len(cd['pts'])} MPs: {cd['n_lo']} Pole - vs {cd['n_hi']} Pole +)", flush=True)

print(f"\nDone! Added +{total_added} new MP-topic core-belief scores across the 20 topics.", flush=True)
