"""Build 2022-2024 historical core-belief scores, quotes, and focus counts for all 20 Claim-Anchor topics
and 72 Focus topics, saving to:
  - data/quick/claim_topics_hist.json
  - data/quick/hist_focus_counts.npz
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
HIST_PARQUET = ROOT / "data/quick/passages_hist_2022_2024.parquet"
OUT_HIST_JSON = ROOT / "data/quick/claim_topics_hist.json"

DS_KEY = next(
    m[1].strip().strip("\"'")
    for m in (re.match(r"\s*(?:export\s+)?DEEPSEEK_API_KEY\s*=\s*(.*)", l) for l in (ROOT / ".env").read_text().splitlines())
    if m
)
ds_sess = requests.Session()
ds_sess.headers.update({"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"})

members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
df_hist = pd.read_parquet(HIST_PARQUET)
claim_data = json.loads((ROOT / "data/quick/claim_topics.json").read_text())
ts = np.load(ROOT / "data/quick/topic_space.npz", allow_pickle=True)
top_terms = ts["top_terms"]

bge = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True)

TOPIC_PATTERNS = {
    48: re.compile(r"\b(assisted dying|assisted death|assisted suicide|terminal illness|terminally ill|palliative care|euthanasia)\b", re.I),
    34: re.compile(r"\b(gaza|israel|israeli|palestin|hamas|west bank|unrwa|rafah|ceasefire|arms export|two-state)\b", re.I),
    28: re.compile(r"\b(asylum|small boat|illegal migration|rwanda|channel crossing|immigration detention|border security|safe routes|refugee)\b", re.I),
    73: re.compile(r"\b(winter fuel|pension credit|pensioner poverty|triple lock|state pension|cold weather payment|energy bills support)\b", re.I),
    61: re.compile(r"\b(two-child|two child|child poverty|universal credit|benefit cap|household support|child benefit)\b", re.I),
    15: re.compile(r"\b(north sea|oil and gas|fossil fuel|rosebank|windfall tax|energy profits levy|offshore petroleum|new licences)\b", re.I),
    67: re.compile(r"\b(rail|railway|railways|train|trains|avanti|northern rail|transpennine|great british railways|network rail|rolling stock|franchis)\b", re.I),
    46: re.compile(r"\b(water compan|sewage|ofwat|thames water|united utilities|severn trent|storm overflow|water industry)\b", re.I),
    68: re.compile(r"\b(european union|single market|customs union|brexit|eu trade|northern ireland protocol|windsor framework|horizon europe|veterinary agreement)\b", re.I),
    43: re.compile(r"\b(jury trial|crown court|magistrates|court backlog|remand|sentencing|criminal justice|legal aid)\b", re.I),
    62: re.compile(r"\b(european convention on human rights|echr|human rights act|strasbourg court|bill of rights|modern slavery act)\b", re.I),
    41: re.compile(r"\b(proportional representation|first past the post|electoral system|voting system|voter id|automatic registration|votes at 16|house of lords reform)\b", re.I),
    33: re.compile(r"\b(farmers|farming|agricultural|food security|sustainable farming|environmental land management|farm support|rural communities)\b", re.I),
    20: re.compile(r"\b(green belt|planning reform|housing targets|national planning policy|grey belt|mandatory targets|local plan|housebuilding)\b", re.I),
    36: re.compile(r"\b(bus services|local bus|bus routes|bus franchising|bus fares|public transport|rural transport)\b", re.I),
    86: re.compile(r"\b(cancer screening|prostate cancer|breast cancer|early diagnosis|cancer backlog|radiotherapy|chemotherapy|cancer strategy)\b", re.I),
    63: re.compile(r"\b(inheritance tax|wealth tax|capital gains|non-dom|tax relief|corporation tax|tax burden|fiscal drag)\b", re.I),
    25: re.compile(r"\b(social housing|affordable housing|council housing|renters reform|no-fault eviction|section 21|private rented|temporary accommodation)\b", re.I),
    45: re.compile(r"\b(onshore wind|wind farm|net zero|renewable energy|clean power|solar farm|peatland|decarbonis)\b", re.I),
    78: re.compile(r"\b(private school|independent school|vat on school|charitable status|state education funding|school funding|send provision)\b", re.I),
}


def call_ds_claims(topic_title, question, pole_lo, pole_hi, mp_bundles):
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
                        digits = re.sub(r"\D", "", k)
                        if digits:
                            res[int(digits)] = v
                    return res
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        return {}

    with ThreadPoolExecutor(8) as ex:
        for res in ex.map(one_batch, batches):
            out.update(res)
    return out


saved = json.loads(OUT_HIST_JSON.read_text()) if OUT_HIST_JSON.exists() else {}

for tk_str, cd in claim_data.items():
    tk = int(tk_str)
    if tk_str in saved and len(saved[tk_str].get("pts", [])) > 0:
        print(f"Skipping already-computed [{tk}] {cd['title']} ({len(saved[tk_str]['pts'])} MPs)", flush=True)
        continue

    pat = TOPIC_PATTERNS[tk]
    mask = df_hist["text"].str.contains(pat) | df_hist["major"].str.contains(pat) | df_hist["minor"].str.contains(pat)
    sub = df_hist[mask].copy()
    sub["hits"] = sub["text"].str.findall(pat).str.len() + 2 * (
        sub["major"].str.contains(pat).astype(int) + sub["minor"].str.contains(pat).astype(int)
    )
    sub = sub[sub["hits"] >= 2].copy()
    sub["weight"] = sub["hits"] * (np.minimum(sub["words"], 220) ** 0.35)

    # Cap at top 90 returning MPs by total weight on this topic so it stays fast and focused on substantive speakers
    mp_weights = sub.groupby("member_id")["weight"].sum().sort_values(ascending=False)
    candidate_mids = [int(m) for m, w in mp_weights.head(90).items() if len(sub[sub.member_id == m]) >= 2 or sub[sub.member_id == m]["hits"].max() >= 4]

    mp_bundles = {}
    for mid in candidate_mids:
        grp = sub[sub.member_id == mid]
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

    print(f"Scoring 2022-24 [{tk}] {cd['title']} ({len(mp_bundles)} returning MPs)...", flush=True)
    hist_claims = call_ds_claims(cd["title"], cd["question"], cd["pole_lo"], cd["pole_hi"], mp_bundles)

    lo_texts = [f"{cd['title']}: {cd['pole_lo']}"] + [e["c"] for e in cd.get("ex_lo", [])[:4]]
    hi_texts = [f"{cd['title']}: {cd['pole_hi']}"] + [e["c"] for e in cd.get("ex_hi", [])[:4]]
    v_lo = normalize(bge.encode(lo_texts, normalize_embeddings=True).mean(0, keepdims=True))[0]
    v_hi = normalize(bge.encode(hi_texts, normalize_embeddings=True).mean(0, keepdims=True))[0]
    axis_vec = v_hi - v_lo
    axis_vec /= np.linalg.norm(axis_vec) + 1e-9

    valid_mids = [m for m, c in hist_claims.items() if m in mp_bundles and isinstance(c, dict) and c.get("side", 0) != 0 and c.get("claim")]
    pts_hist = []
    quotes_hist = {}
    if valid_mids:
        c_texts = [f"{cd['title']}: {hist_claims[m]['claim']}" for m in valid_mids]
        c_embs = bge.encode(c_texts, normalize_embeddings=True)
        raw_proj = c_embs @ axis_vec
        raw_proj = (raw_proj - np.mean(raw_proj)) / (np.std(raw_proj) + 1e-9)
        for idx_m, m in enumerate(valid_mids):
            side = int(hist_claims[m]["side"])
            sc = round(float(np.clip(0.55 * side + 0.45 * np.clip(raw_proj[idx_m] * 0.55, -0.9, 0.9), -1.15, 1.15)), 3)
            info = mp_bundles[m]
            pts_hist.append([m, sc, 1 if sc > 0 else -1, info["n_sp"], info["words"]])
            quotes_hist[str(m)] = {
                "c": hist_claims[m]["claim"],
                "t": info["latest"]["t"],
                "d": info["latest"]["d"],
                "h": info["latest"]["h"],
                "ex": info["ex"],
            }

    saved[tk_str] = {"pts": pts_hist, "quotes": quotes_hist}
    OUT_HIST_JSON.write_text(json.dumps(saved, indent=2, ensure_ascii=False))
    print(f"  -> Saved {len(pts_hist)} scored MPs for [{tk}] {cd['title']}", flush=True)

print("All 20 historical topics scored and saved to data/quick/claim_topics_hist.json!", flush=True)
