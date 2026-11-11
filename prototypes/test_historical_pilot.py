"""Historical Speech Pilot Test (2022-01-01 to 2024-05-30 vs 2024-07-09 to 2026):
1. Downloads 2022-2024 Commons debates XML into data/raw/hansard_hist/debates (cached).
2. Cleans & extracts passages for returning MPs (who sit in the 2024-2026 Parliament).
3. Assigns topics using our existing TF-IDF + LSA + KMeans topic model.
4. Runs our Multi-Passage Claim-Anchor pipeline on 6 core topics for 2022-2024 speeches
   and evaluates:
   - Coverage gain (especially current Ministers / Frontbenchers silent in 2024-26)
   - Cross-parliament belief correlation (2022-24 Opposition/Gov vs 2024-26 Gov/Opposition)
   - Validation accuracy (Commons Vote AUC, Jevons |r|, Wikipedia agreement) before vs after blending.
"""
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import pathlib
import re
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
RAW_HIST = ROOT / "data/raw/hansard_hist/debates"
RAW_HIST.mkdir(parents=True, exist_ok=True)
HIST_PARQUET = ROOT / "data/quick/passages_hist_2022_2024.parquet"
HIST_RESULTS = ROOT / "data/quick/historical_pilot_report.json"

# Load DeepSeek API key safely from .env without printing
DS_KEY = next(
    m[1].strip().strip("\"'")
    for m in (re.match(r"\s*(?:export\s+)?DEEPSEEK_API_KEY\s*=\s*(.*)", l) for l in (ROOT / ".env").read_text().splitlines())
    if m
)
ds_sess = requests.Session()
ds_sess.headers.update({"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"})

session = requests.Session()
session.headers.update({"User-Agent": "MP3D historical pilot test"})
adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
session.mount("https://", adapter)

# 1. Download 2022-2024 debate XMLs
base = "https://www.theyworkforyou.com/pwdata/scrapedxml/debates"
html = session.get(f"{base}/", timeout=60).text
pat = re.compile(r'href="([^"]+?(\d{4}-\d{2}-\d{2})([a-z])\.xml)"')
newest = {}
for name, day, letter in pat.findall(html):
    if "2022-01-01" <= day < "2024-06-01":
        key = name[: -len(letter) - 4]
        if key not in newest or letter > newest[key][1]:
            newest[key] = (name, letter)
wanted = [name for name, _ in sorted(newest.values())]
print(f"2022-2024 debate XMLs: {len(wanted)} sittings", flush=True)


def fetch_xml(name):
    p = RAW_HIST / name
    if p.exists() and p.stat().st_size > 100:
        return
    for attempt in range(4):
        try:
            r = session.get(f"{base}/{name}", timeout=45)
            if r.status_code == 200:
                p.write_bytes(r.content)
                return
        except Exception:
            time.sleep(1.5 * (attempt + 1))


with ThreadPoolExecutor(10) as ex:
    futs = [ex.submit(fetch_xml, n) for n in wanted]
    for i, f in enumerate(as_completed(futs), 1):
        f.result()
        if i % 100 == 0 or i == len(wanted):
            print(f"  downloaded {i}/{len(wanted)}", flush=True)

# 2. Parse & clean historical passages for returning MPs
members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
known = set(members.index)

if not HIST_PARQUET.exists():
    pdata = json.loads((ROOT / "data/raw/people.json").read_text())
    lookup = {}
    for p in pdata["persons"]:
        for ident in p.get("identifiers", []):
            if ident.get("scheme") == "datadotparl_id":
                did = int(ident["identifier"])
                lookup[p["id"]] = did
    for m in pdata.get("memberships", []):
        pid = m.get("person_id")
        if pid in lookup and m.get("id"):
            lookup[m["id"]] = lookup[pid]

    SKIP_HEADING = re.compile(
        r"point of order|business of the house|business statement|royal assent|speaker.s statement|"
        r"deferred division|programme motion|sittings of the house|delegated legislation|preamble|"
        r"business without debate|petition|tributes?\b|oaths?\b|swearing|engagements",
        re.I,
    )
    CHAIR = re.compile(r"\b(speaker|deputy speaker|in the chair|the chair|chairman|chairwoman|chair)\b", re.I)

    rows = []
    for xml_path in sorted(RAW_HIST.glob("*.xml")):
        m_date = re.search(r"(\d{4}-\d{2}-\d{2})", xml_path.name)
        if not m_date:
            continue
        date = m_date.group(1)
        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            continue
        major = minor = ""
        for el in root.iter():
            if el.tag in ("major-heading", "oral-heading"):
                major, minor = " ".join("".join(el.itertext()).split()), ""
            elif el.tag == "minor-heading":
                minor = " ".join("".join(el.itertext()).split())
            elif el.tag == "speech":
                if SKIP_HEADING.search(f"{major} {minor}"):
                    continue
                pid = el.get("person_id") or el.get("speakerid")
                mid = lookup.get(pid)
                if mid not in known:
                    continue
                sp_label = el.get("speakername") or ""
                if CHAIR.search(sp_label):
                    continue
                paras = [
                    " ".join("".join(p.itertext()).split())
                    for p in el.findall("p")
                    if p.get("class") != "italic" and p.get("pwmotiontext") != "yes"
                ]
                text = " ".join(paras)
                words = len(text.split())
                if words < 35:
                    continue
                rows.append({
                    "passage_id": el.get("id") or f"{date}_{len(rows)}",
                    "member_id": int(mid),
                    "date": date,
                    "major": major,
                    "minor": minor,
                    "text": text[:1600],
                    "words": words,
                })
    df_hist = pd.DataFrame(rows)
    df_hist.to_parquet(HIST_PARQUET, index=False)
    print(f"Saved {len(df_hist)} historical speeches across {df_hist.member_id.nunique()} returning MPs", flush=True)
else:
    df_hist = pd.read_parquet(HIST_PARQUET)
    print(f"Loaded {len(df_hist)} historical speeches across {df_hist.member_id.nunique()} returning MPs", flush=True)

# 3. Assign topics to historical speeches using TF-IDF + LSA matching our 90 topics
df_cur = pd.read_parquet(
    ROOT / "data/clean/passages.parquet",
    columns=["passage_id", "member_id", "date", "major", "minor", "text", "words"],
)
ts = np.load(ROOT / "data/quick/topic_space.npz", allow_pickle=True)
df_cur["topic"] = ts["topic"]
df_cur["sim"] = ts["sim"]
top_terms = ts["top_terms"]

# Score historical passages against each topic's top terms + debate headings
TEST_TOPICS = [67, 61, 34, 41, 15, 46]  # Rail, Two-Child Cap, Israel-Gaza, PR/Elections, North Sea Oil/Gas, Water
TOPIC_PATTERNS = {
    67: re.compile(r"\b(rail|railway|railways|train|trains|avanti|northern rail|transpennine|great british railways|network rail|rolling stock|franchis)\b", re.I),
    61: re.compile(r"\b(two-child|two child|child poverty|universal credit|benefit cap|household support|child benefit)\b", re.I),
    34: re.compile(r"\b(gaza|israel|israeli|palestin|hamas|west bank|unrwa|rafah|ceasefire|arms export|middle east)\b", re.I),
    41: re.compile(r"\b(proportional representation|first past the post|electoral system|voting system|voter id|automatic registration|votes at 16|general election)\b", re.I),
    15: re.compile(r"\b(north sea|oil and gas|fossil fuel|rosebank|windfall tax|energy profits levy|offshore|clean power|carbon capture)\b", re.I),
    46: re.compile(r"\b(water compan|sewage|ofwat|thames water|united utilities|severn trent|storm overflow|pollut|water industry)\b", re.I),
}

claim_data = json.loads((ROOT / "data/quick/claim_topics.json").read_text())
jev = pd.read_parquet(ROOT / "data/quick/jev_stance_pilot.parquet") if (ROOT / "data/quick/jev_stance_pilot.parquet").exists() else None
votes_long = pd.read_parquet(ROOT / "data/clean/votes_long.parquet")

# Map topics to their validation division & Jevons key (from build_claim_topics.py)
VAL_MAP = {
    67: {"div": 1832, "sign": 1, "jev": "rail_public"},
    61: {"div": 2131, "sign": -1, "jev": "two_child_cap"},
    34: {"div": None, "sign": 1, "jev": "gaza_arms"},
    41: {"div": None, "sign": 1, "jev": "pr_elections"},
    15: {"div": None, "sign": -1, "jev": "north_sea"},
    46: {"div": None, "sign": 1, "jev": "water_public"},
}

bge = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True)


def call_gemini_claims(topic_title, question, pole_lo, pole_hi, mp_bundles):
    """Extract 1-line timeless core-belief claim per MP from their 2022-2024 speeches using parallel DeepSeek calls."""
    if not mp_bundles:
        return {}
    out = {}
    items = list(mp_bundles.items())
    batches = [items[i : i + 15] for i in range(0, len(items), 15)]

    def one_batch(batch):
        lines = [f"MP_{mid} ({info['name']}, {info['party']}):\n{info['bundle']}" for mid, info in batch]
        prompt = (
            f"Topic: {topic_title}\n"
            f"Core Belief Question: {question}\n"
            f"Pole - : {pole_lo}\n"
            f"Pole + : {pole_hi}\n\n"
            "Below are speeches from the 2022-2024 UK Parliament (when Conservatives were in Government and Labour in Opposition).\n"
            "For each MP, state their underlying CORE POLICY BELIEF on this topic in one clear sentence (10-18 words), "
            "independent of which party was in government at the time.\n"
            "Also classify side: -1 (aligns with Pole -), +1 (aligns with Pole +), or 0 (unclear/off-topic/purely local).\n"
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
                        "max_tokens": 1200,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=55,
                )
                if r.status_code == 200:
                    txt = r.json()["choices"][0]["message"]["content"]
                    parsed = json.loads(txt)
                    res = {}
                    for k, v in parsed.items():
                        m_id = int(re.sub(r"\D", "", k))
                        res[m_id] = v
                    return res
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        return {}

    with ThreadPoolExecutor(8) as ex:
        for res in ex.map(one_batch, batches):
            out.update(res)
    return out


report = {}
for tk in TEST_TOPICS:
    cd = claim_data[str(tk)]
    pat = TOPIC_PATTERNS[tk]
    # Filter historical speeches matching this topic
    mask = df_hist["text"].str.contains(pat) | df_hist["major"].str.contains(pat) | df_hist["minor"].str.contains(pat)
    sub = df_hist[mask].copy()
    # Count matches per passage
    sub["hits"] = sub["text"].str.findall(pat).str.len() + 2 * (
        sub["major"].str.contains(pat).astype(int) + sub["minor"].str.contains(pat).astype(int)
    )
    sub = sub[sub["hits"] >= 2].copy()
    sub["weight"] = sub["hits"] * (np.minimum(sub["words"], 220) ** 0.35)

    # Bundle top 3 historical speeches for MPs with >= 2 matching speeches (or >=1 strong speech with hits>=4)
    mp_bundles = {}
    for mid, grp in sub.groupby("member_id"):
        if len(grp) < 2 and grp["hits"].max() < 4:
            continue
        top3 = grp.sort_values("weight", ascending=False).head(3).sort_values("date")
        ex_strs = []
        for r in top3.itertuples():
            hd = f"{r.major} ({r.minor})" if r.minor else r.major
            ex_strs.append(f"[{r.date}, {hd}]: {r.text[:420]}")
        mp_bundles[int(mid)] = {
            "name": members.loc[mid, "name"],
            "party": members.loc[mid, "party_abbr"] or "Ind",
            "n_sp": len(grp),
            "bundle": "\n".join(ex_strs),
            "latest_date": top3.date.max(),
            "sample_quote": top3.iloc[-1].text[:240],
            "sample_hd": top3.iloc[-1].minor or top3.iloc[-1].major,
        }

    print(f"\n=== Topic [{tk}] {cd['title']}: {len(mp_bundles)} returning MPs with 2022-24 speeches ===", flush=True)
    hist_claims = call_gemini_claims(cd["title"], cd["question"], cd["pole_lo"], cd["pole_hi"], mp_bundles)

    # Embed the extracted claims with BGE and project onto the Pole - <-> Pole + anchor axis!
    cur_pts = {int(p[0]): float(p[1]) for p in cd["pts"]}
    # Use the topic's pole descriptions + example anchor claims to form the BGE direction vector
    lo_texts = [f"{cd['title']}: {cd['pole_lo']}"] + [e["c"] for e in cd.get("ex_lo", [])[:4]]
    hi_texts = [f"{cd['title']}: {cd['pole_hi']}"] + [e["c"] for e in cd.get("ex_hi", [])[:4]]
    v_lo = normalize(bge.encode(lo_texts, normalize_embeddings=True).mean(0, keepdims=True))[0]
    v_hi = normalize(bge.encode(hi_texts, normalize_embeddings=True).mean(0, keepdims=True))[0]
    axis_vec = v_hi - v_lo
    axis_vec /= np.linalg.norm(axis_vec) + 1e-9

    hist_scores = {}
    valid_mids = [m for m, c in hist_claims.items() if isinstance(c, dict) and c.get("side", 0) != 0 and c.get("claim")]
    if valid_mids:
        c_texts = [f"{cd['title']}: {hist_claims[m]['claim']}" for m in valid_mids]
        c_embs = bge.encode(c_texts, normalize_embeddings=True)
        raw_proj = c_embs @ axis_vec
        raw_proj = (raw_proj - np.mean(raw_proj)) / (np.std(raw_proj) + 1e-9)
        for idx_m, m in enumerate(valid_mids):
            side = int(hist_claims[m]["side"])
            sc = 0.55 * side + 0.45 * np.clip(raw_proj[idx_m] * 0.55, -0.9, 0.9)
            hist_scores[m] = round(float(np.clip(sc, -1.15, 1.15)), 3)

    # Compare 2022-24 scores vs 2024-26 scores!
    overlap = sorted(set(hist_scores.keys()) & set(cur_pts.keys()))
    new_mps = sorted(set(hist_scores.keys()) - set(cur_pts.keys()))
    if len(overlap) >= 5:
        x_hist = np.array([hist_scores[m] for m in overlap])
        y_cur = np.array([cur_pts[m] for m in overlap])
        r_cross = float(np.corrcoef(x_hist, y_cur)[0, 1])
        same_sign = float(np.mean(np.sign(x_hist) == np.sign(y_cur)))
    else:
        r_cross, same_sign = None, None

    # Find specific MPs who flipped between 2022-24 (Opposition/Gov) and 2024-26 (Gov/Opposition)
    flips = []
    for m in overlap:
        s_old, s_new = hist_scores[m], cur_pts[m]
        if abs(s_old) >= 0.35 and abs(s_new) >= 0.35 and s_old * s_new < 0 and abs(s_new - s_old) >= 0.85:
            flips.append({
                "member_id": m,
                "name": members.loc[m, "name"],
                "party": members.loc[m, "party_abbr"],
                "s_2022_24": s_old,
                "s_2024_26": s_new,
                "claim_2022_24": hist_claims[m]["claim"],
                "claim_2024_26": cd["quotes"].get(str(m), {}).get("c", ""),
            })

    # Build Combined (2022-2026) score dictionary:
    # - If MP spoke in both, weight 0.65 * current + 0.35 * historical (or expose shift)
    # - If MP only spoke in 2022-24 (e.g. now a silent Minister/Whip), use their 2022-24 core belief!
    comb_scores = dict(cur_pts)
    for m, s_old in hist_scores.items():
        if m in comb_scores:
            comb_scores[m] = round(0.65 * comb_scores[m] + 0.35 * s_old, 3)
        else:
            comb_scores[m] = s_old

    # Evaluate Vote AUC on 2024-26 only vs Combined 2022-26
    div_id = VAL_MAP[tk]["div"]
    v_sign = VAL_MAP[tk]["sign"]
    auc_cur = auc_comb = n_v_cur = n_v_comb = None
    if div_id is not None:
        v_sub = votes_long[votes_long.division_id == div_id].set_index("member_id")["vote"] * v_sign
        v_sub = v_sub[v_sub.isin([-1, 1])]
        v_sub = v_sub[~v_sub.index.duplicated(keep="first")]
        common_cur = [m for m in cur_pts if m in v_sub.index]
        common_comb = [m for m in comb_scores if m in v_sub.index]
        if len(common_cur) >= 10 and len(set(v_sub.loc[common_cur])) == 2:
            auc_cur = round(float(roc_auc_score((v_sub.loc[common_cur] > 0).astype(int), [cur_pts[m] for m in common_cur])), 3)
            n_v_cur = len(common_cur)
        if len(common_comb) >= 10 and len(set(v_sub.loc[common_comb])) == 2:
            auc_comb = round(float(roc_auc_score((v_sub.loc[common_comb] > 0).astype(int), [comb_scores[m] for m in common_comb])), 3)
            n_v_comb = len(common_comb)

    print(
        f"  Scored 2022-24: {len(hist_scores)} MPs | Newly unlocked MPs (silent in 2024-26): +{len(new_mps)} "
        f"(total {len(cur_pts)} -> {len(comb_scores)})",
        flush=True,
    )
    print(
        f"  Overlap ({len(overlap)} MPs): Cross-Parliament r = {r_cross:.2f} | Same-side consistency = {same_sign*100:.0f}% | Flips = {len(flips)}"
        if r_cross is not None
        else f"  Overlap: {len(overlap)} MPs",
        flush=True,
    )
    if auc_cur is not None:
        print(f"  Vote AUC: 2024-26 = {auc_cur} (n={n_v_cur}) -> Combined 2022-26 = {auc_comb} (n={n_v_comb})", flush=True)
    for fl in flips[:4]:
        print(
            f"    FLIP: {fl['name']} ({fl['party']}): 2022-24 [{fl['s_2022_24']:+.2f}] '{fl['claim_2022_24']}' -> 2024-26 [{fl['s_2024_26']:+.2f}] '{fl['claim_2024_26']}'",
            flush=True,
        )

    report[str(tk)] = {
        "title": cd["title"],
        "n_2024_26": len(cur_pts),
        "n_2022_24": len(hist_scores),
        "n_new_unlocked": len(new_mps),
        "n_combined": len(comb_scores),
        "n_overlap": len(overlap),
        "cross_parliament_r": round(r_cross, 3) if r_cross is not None else None,
        "same_side_pct": round(same_sign * 100, 1) if same_sign is not None else None,
        "auc_2024_26": auc_cur,
        "n_vote_2024_26": n_v_cur,
        "auc_combined": auc_comb,
        "n_vote_combined": n_v_comb,
        "flips": flips,
        "new_unlocked_examples": [
            {"member_id": m, "name": members.loc[m, "name"], "party": members.loc[m, "party_abbr"], "score": hist_scores[m], "claim": hist_claims[m]["claim"]}
            for m in new_mps[:8]
        ],
    }

HIST_RESULTS.write_text(json.dumps(report, indent=2))
print("\nSaved historical pilot report to data/quick/historical_pilot_report.json", flush=True)
