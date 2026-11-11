"""Discover and score real opposing position axes across 20 major unsupervised Commons speech topics
from data/quick/topic_space.npz + data/clean/passages.parquet (ministers included!).

Pipeline per topic (no hand-written proposals):
1. Pull top central passages (words >= 45) for up to 75 distinct MPs in the topic (ministers + backbench).
2. Extract a 1-line normative claim per passage in isolation ("Supports...", "Opposes...", "Calls for...").
3. Cluster the 1-line claims with local BGE-base (K=6) to find the 6 main argument centroids.
4. Identify the two clearest opposing positions (Pole - vs Pole +) from the 6 centroids, plus whether
   they are direct opposites ("stance") or contrasting priorities ("focus_contrast").
5. Score each passage against the two real anchor passages (+1, -1, 0) + continuous BGE claim projection.
6. Validate against Party R^2, Within-Labour variance, Jev POC stances, and Commons division votes.
Saves incrementally to data/quick/claim_topics.json.
"""
import json
import pathlib
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
OUT_FILE = ROOT / "data/quick/claim_topics.json"

DS_KEY = next(
    m[1].strip().strip("\"'")
    for m in (re.match(r"\s*(?:export\s+)?DEEPSEEK_API_KEY\s*=\s*(.*)", l) for l in (ROOT / ".env").read_text().splitlines())
    if m
)

sess = requests.Session()
sess.headers.update({"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"})


def call_ds(prompt, max_tokens=80, json_mode=False):
    for attempt in range(5):
        try:
            body = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
            }
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            r = sess.post("https://api.deepseek.com/chat/completions", json=body, timeout=35)
            if r.status_code in (429, 500, 502, 503):
                time.sleep(1.2 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            time.sleep(1.2 * (attempt + 1))
    return "{}" if json_mode else ""


print("Loading data/clean/passages.parquet and data/quick/topic_space.npz...")
passages = pd.read_parquet(
    ROOT / "data/clean/passages.parquet",
    columns=["passage_id", "member_id", "role", "date", "major", "minor", "text", "words"],
)
ts = np.load(ROOT / "data/quick/topic_space.npz", allow_pickle=True)
passages["topic"] = ts["topic"]
passages["sim"] = ts["sim"]
top_terms = ts["top_terms"]

members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
votes = pd.read_parquet(ROOT / "data/clean/votes_long.parquet")

ns = {"__file__": str(ROOT / "prototypes/stance_poc.py"), "__name__": "stance_poc"}
exec((ROOT / "prototypes/stance_poc.py").read_text(), ns)
SUBJECTS, load_poc = ns["SUBJECTS"], ns["load"]

print("Loading cached BGE-base-en-v1.5...")
bge = SentenceTransformer("BAAI/bge-base-en-v1.5")

# 20 diverse unsupervised topics from topic_space.npz (covering contentious debates + focus topics)
# Format: (topic_k, optional_poc_check (subj, prop), optional_division_id)
TARGET_TOPICS = [
    (48, ("dying", "assisted"), 1877),       # assisted, dying, doctor
    (34, ("gaza", "arms"), None),            # gaza, israel, palestinian
    (28, ("asylum", "rwanda"), 2406),        # asylum, immigration, hotels
    (73, ("welfare", "winterfuel"), 1840),   # pensioners, winter fuel
    (61, ("welfare", "twochild"), 2131),     # poverty, child poverty, two-child
    (63, ("tax", "farms"), 1881),            # tax, inheritance tax, relief
    (25, ("work", "rentcap"), None),         # housing, social housing, tenants, landlords, rented
    (45, ("energy", "netzero"), None),       # energy, net zero, clean
    (15, ("energy", "northsea"), None),      # oil, gas, north sea
    (67, ("ownership", "rail"), 1832),       # rail, railway, passengers, gbr
    (46, ("ownership", "water"), None),      # water, sewage, thames water
    (68, ("europe", "customs"), None),       # union, trade, european union, trade union
    (43, ("justice", "jury"), None),         # court, jury, crown court, magistrates
    (62, ("asylum", "echr"), 2159),          # human rights, european convention
    (41, ("elections", "pr"), 1878),         # vote, election, electoral
    (33, ("tax", "farms"), 1881),            # farmers, farming, food, farms
    (20, None, None),                        # planning, development, green belt
    (36, None, None),                        # bus, transport, bus services, buses
    (78, None, None),                        # children, schools, send, special educational
    (86, None, None),                        # cancer, breast cancer, diagnosis, treatment
]

saved = json.loads(OUT_FILE.read_text()) if OUT_FILE.exists() else {}
MAX_MPS_PER_TOPIC = 80

for tk, poc_ref, div_id in TARGET_TOPICS:
    tk_str = str(tk)
    t_words = [str(w) for w in top_terms[tk][:8]]
    if tk_str in saved and saved[tk_str].get("multi_passage_v2"):
        print(f"[Topic {tk:2d}] Already computed v2 ({saved[tk_str]['pole_lo']} <-> {saved[tk_str]['pole_hi']}), skipping.")
        continue

    t0 = time.time()
    sub = passages[(passages.topic == tk) & (passages.words >= 45) & (passages.sim >= 0.35)].copy()
    # Exclude catch-all PMQs or Business of the House where topic assignment can be noisy
    heading = (sub.major.fillna("") + " " + sub.minor.fillna("")).str.lower()
    not_pmq = ~heading.str.contains("prime minister|business of the house|points of order|speaker's statement")
    if not_pmq.sum() >= 80:
        sub = sub[not_pmq]

    # Score substantive weight per passage: prefer longer substantive speeches and penalize procedural clause/amendment wrap-ups
    minor_l = sub.minor.fillna("").str.lower()
    is_amend = minor_l.str.contains("before clause|clause [0-9]|schedule [0-9]|lords amendment|new clause")
    sub["subst_score"] = sub["sim"] * (np.minimum(sub["words"], 220) ** 0.35) * np.where(is_amend, 0.72, 1.0)

    mp_counts = sub.groupby("member_id").size().rename("n_sp")
    sub = sub.sort_values("subst_score", ascending=False)

    # Rank MPs by number of passages in this topic + substantive centrality
    mp_rank = sub.groupby("member_id").agg(n=("passage_id", "size"), max_s=("subst_score", "max"))
    mp_rank["score"] = np.log1p(mp_rank.n) * mp_rank.max_s
    keep_mps = set(mp_rank.sort_values("score", ascending=False).head(MAX_MPS_PER_TOPIC).index)

    # Keep top 3 substantive passages per MP so frequent speakers/ministers are judged across multiple speeches
    top3_per_mp = sub[sub.member_id.isin(keep_mps)].groupby("member_id").head(3).copy()
    one_per_mp = top3_per_mp.drop_duplicates("member_id").reset_index(drop=True)
    one_per_mp["n_sp"] = one_per_mp.member_id.map(mp_counts).fillna(1).astype(int)
    one_per_mp["party"] = members.party_abbr.reindex(one_per_mp.member_id).fillna("Ind").values

    # Build multi-passage bundle per MP (up to 3 substantive excerpts with Debate Heading + Date)
    bundles = []
    lead_snips = []
    mp_excerpts = []
    for mid in one_per_mp.member_id:
        rows = top3_per_mp[top3_per_mp.member_id == mid]
        parts = []
        ex_list = []
        for idx_r, (_, r) in enumerate(rows.iterrows()):
            deb = str(r.major or r.minor or "").strip()
            sub_h = str(r.minor or "").strip()
            if sub_h and sub_h != deb:
                deb = f"{deb} ({sub_h})"
            w_lim = 125 if idx_r == 0 else 85
            txt_trim = " ".join(str(r.text).split()[:w_lim])
            parts.append(f"[Debate: {deb}, {r.date}]: {txt_trim}")
            ex_list.append({
                "t": " ".join(str(r.text).split()[:45]) + "…",
                "d": str(r.date),
                "h": deb,
            })
        bundles.append("\n---\n".join(parts))
        lead_snips.append(parts[0])
        mp_excerpts.append(ex_list)

    # 1. Extract 1-line claims in parallel across each MP's multi-speech bundle
    def get_claim(b_txt):
        return call_ds(
            "State in one short sentence (max 15 words) the speaker's substantive policy stance or argument across these Commons speech excerpts, "
            "starting with 'Supports', 'Opposes', or 'Calls for'. Ignore procedural amendment numbers (e.g. 'Lords amendment 1'); state the core policy position. "
            "Do not mention speaker names or parties.\n\n"
            f"{b_txt}",
            max_tokens=42,
        )

    with ThreadPoolExecutor(22) as ex:
        claims = list(ex.map(get_claim, bundles))

    # 2. Cluster claims with BGE-base (K=6)
    c_emb = bge.encode(claims, normalize_embeddings=True, show_progress_bar=False)
    K = min(8, len(one_per_mp) // 4)
    km = KMeans(n_clusters=K, random_state=19, n_init=10).fit(c_emb)
    reps = []
    for c in range(K):
        idx_c = np.where(km.labels_ == c)[0]
        best_i = int(idx_c[np.argmax(c_emb[idx_c] @ km.cluster_centers_[c])])
        reps.append((best_i, len(idx_c)))

    rep_desc = "\n".join(
        f"Cluster {idx} ({cnt} MPs): Claim='{claims[r]}' | Excerpt='{lead_snips[r][:190]}'"
        for idx, (r, cnt) in enumerate(reps)
    )

    # 3. Pick the two opposing poles (or main contrasting positions) from the K centroids
    pick_prompt = (
        f"Here are {K} representative clusters of MP speeches on the topic of ({', '.join(t_words[:5])}):\n\n"
        f"{rep_desc}\n\n"
        "Task: Identify the two clusters (0 to {kmax}, each having at least 5 MPs if possible) that take the clearest DIRECTLY OPPOSITE sides (For vs Against) on the same policy question. Never pick two clusters that agree with each other "
        "(or, if everyone agrees on the goal, the two clearest contrasting policy priorities/approaches).\n"
        "Avoid generic 'Government is good vs bad' if a substantive policy divide exists.\n"
        "Return JSON:\n"
        '{{"a": int, "b": int, "kind": "opposing_stances" or "contrasting_priorities", '
        '"topic_title": "short 2-4 word topic name", '
        '"question": "1-sentence neutral question that divides Pole - and Pole +", '
        '"pole_lo": "clear 4-7 word label for Cluster A position (Pole -)", '
        '"pole_hi": "clear 4-7 word label for Cluster B position (Pole +)"}}'
    ).format(kmax=K - 1)

    pick = json.loads(call_ds(pick_prompt, max_tokens=150, json_mode=True))
    ca = min(max(int(pick.get("a", 0)), 0), K - 1)
    cb = min(max(int(pick.get("b", 1)), 0), K - 1)
    if ca == cb:
        cb = (ca + 1) % K
    ia, ib = reps[ca][0], reps[cb][0]
    pole_lo = pick.get("pole_lo", claims[ia][:45])
    pole_hi = pick.get("pole_hi", claims[ib][:45])
    kind = pick.get("kind", "opposing_stances")
    topic_title = pick.get("topic_title", ", ".join(t_words[:2]).title())
    question = pick.get("question", f"{pole_lo} vs {pole_hi}")
    snip_lo, snip_hi = lead_snips[ia], lead_snips[ib]

    # 4. Score all MPs in this topic against Pole - (Cluster A) and Pole + (Cluster B) using their multi-passage bundle + extracted claim
    def score_mp(i):
        res = json.loads(
            call_ds(
                f"Topic question: {question}\n"
                f"Position - ({pole_lo}):\n{snip_lo}\n\n"
                f"Position + ({pole_hi}):\n{snip_hi}\n\n"
                f"Speaker's speeches (with debate headings):\n{bundles[i]}\n"
                f"(Summary of speaker's substantive claim: {claims[i]})\n\n"
                "Based on the speaker's substantive policy argument across these excerpts, do they align more with Position - (-1), align more with Position + (+1), or neither/neutral (0)? "
                'Return JSON: {"s": -1 or 0 or 1}',
                max_tokens=20,
                json_mode=True,
            )
        )
        try:
            return float(res.get("s", 0))
        except Exception:
            return 0.0

    with ThreadPoolExecutor(22) as ex:
        raw_s = np.array(list(ex.map(score_mp, range(len(one_per_mp)))))

    # Blend discrete judge score (-1, 0, +1) with continuous BGE claim projection so MPs have smooth resolution
    diff_v = c_emb[ib] - c_emb[ia]
    diff_v /= np.linalg.norm(diff_v) + 1e-9
    cont_s = c_emb @ diff_v
    cont_s = (cont_s - cont_s.mean()) / (cont_s.std() + 1e-9) * 0.35
    final_s = np.clip(0.75 * raw_s + 0.25 * cont_s, -1.2, 1.2)

    # 5. Compute validation stats
    n_lo = int((raw_s < 0).sum())
    n_hi = int((raw_s > 0).sum())
    n_neu = int((raw_s == 0).sum())
    split_balance = round(min(n_lo, n_hi) / max(n_lo + n_hi, 1), 2)

    s_ser = pd.Series(final_s, index=one_per_mp.index)
    g_p = s_ser.groupby(one_per_mp.party)
    party_r2 = round(float(((g_p.transform("mean") - s_ser.mean()) ** 2).sum() / ((s_ser - s_ser.mean()) ** 2 + 1e-9).sum()), 2)

    # Backbench-only party R2
    bb = one_per_mp.role != "minister"
    if bb.sum() >= 15:
        s_bb = s_ser[bb]
        g_bb = s_bb.groupby(one_per_mp.loc[bb, "party"])
        bb_party_r2 = round(float(((g_bb.transform("mean") - s_bb.mean()) ** 2).sum() / ((s_bb - s_bb.mean()) ** 2 + 1e-9).sum()), 2)
    else:
        bb_party_r2 = party_r2

    # Check against POC Jev stance & Commons Division Vote if available
    jev_r_all, jev_r_lab, vote_auc, vote_auc_lab = None, None, None, None
    if poc_ref is not None:
        subj, prop = poc_ref
        dp = load_poc(subj)
        st_gt = dp[(dp.proposal == prop) & (dp.side != 0)].groupby("member").side.mean()
        m_gt = one_per_mp.member_id.map(st_gt)
        ok = m_gt.notna() & (raw_s != 0)
        if ok.sum() >= 8 and s_ser[ok].std() > 1e-8 and m_gt[ok].std() > 1e-8:
            jev_r_all = round(float(abs(np.corrcoef(s_ser[ok], m_gt[ok])[0, 1])), 2)
        ok_lab = ok & (one_per_mp.party == "Lab")
        if ok_lab.sum() >= 6 and s_ser[ok_lab].std() > 1e-8 and m_gt[ok_lab].std() > 1e-8:
            jev_r_lab = round(float(abs(np.corrcoef(s_ser[ok_lab], m_gt[ok_lab])[0, 1])), 2)

    if div_id is not None:
        v = votes[votes.division_id == div_id].drop_duplicates("member_id", keep=False).set_index("member_id").vote
        mv = one_per_mp.member_id.map(v)
        okv = mv.isin([1, -1])
        if okv.sum() >= 10 and mv[okv].nunique() == 2:
            yv = (mv[okv] == 1).astype(int)
            auc = roc_auc_score(yv, s_ser[okv])
            vote_auc = round(float(max(auc, 1 - auc)), 2)
        okv_lab = okv & (one_per_mp.party == "Lab")
        if okv_lab.sum() >= 8 and mv[okv_lab].nunique() == 2:
            ylv = (mv[okv_lab] == 1).astype(int)
            aucl = roc_auc_score(ylv, s_ser[okv_lab])
            vote_auc_lab = round(float(max(aucl, 1 - aucl)), 2)

    # Top quotes at Pole - and Pole +
    lo_order = np.argsort(final_s)
    hi_order = np.argsort(-final_s)
    ex_lo = [
        {
            "m": int(one_per_mp.member_id.iloc[i]),
            "c": claims[i],
            "t": mp_excerpts[i][0]["t"],
            "d": mp_excerpts[i][0]["d"],
            "h": mp_excerpts[i][0]["h"],
            "min": bool(one_per_mp.role.iloc[i] == "minister"),
        }
        for i in lo_order[:3]
        if raw_s[i] < 0
    ]
    ex_hi = [
        {
            "m": int(one_per_mp.member_id.iloc[i]),
            "c": claims[i],
            "t": mp_excerpts[i][0]["t"],
            "d": mp_excerpts[i][0]["d"],
            "h": mp_excerpts[i][0]["h"],
            "min": bool(one_per_mp.role.iloc[i] == "minister"),
        }
        for i in hi_order[:3]
        if raw_s[i] > 0
    ]

    pts_list = []
    quotes_dict = {}
    for i, r in one_per_mp.iterrows():
        mid = int(r.member_id)
        pts_list.append([mid, round(float(final_s[i]), 3), int(raw_s[i]), int(r.n_sp), int(r.role == "minister")])
        quotes_dict[str(mid)] = {
            "c": claims[i],
            "t": mp_excerpts[i][0]["t"],
            "d": mp_excerpts[i][0]["d"],
            "h": mp_excerpts[i][0]["h"],
            "ex": mp_excerpts[i],
            "min": bool(r.role == "minister"),
        }

    saved[tk_str] = {
        "tk": tk,
        "multi_passage_v2": True,
        "title": topic_title,
        "words": t_words,
        "kind": kind,
        "question": question,
        "pole_lo": pole_lo,
        "pole_hi": pole_hi,
        "n_mps": len(one_per_mp),
        "n_lo": n_lo,
        "n_hi": n_hi,
        "n_neu": n_neu,
        "split_balance": split_balance,
        "party_r2": party_r2,
        "bb_party_r2": bb_party_r2,
        "jev_r_all": jev_r_all,
        "jev_r_lab": jev_r_lab,
        "vote_auc": vote_auc,
        "vote_auc_lab": vote_auc_lab,
        "ex_lo": ex_lo,
        "ex_hi": ex_hi,
        "pts": pts_list,
        "quotes": quotes_dict,
    }
    OUT_FILE.write_text(json.dumps(saved, indent=2, ensure_ascii=False))
    print(
        f"[Topic {tk:2d}: {topic_title:22s}] {time.time()-t0:4.1f}s | {n_lo:2d}- vs {n_hi:2d}+ (bal={split_balance:.2f}) | "
        f"partyR2={party_r2:.2f} | Jev r={jev_r_all} (Lab {jev_r_lab}) | Vote AUC={vote_auc} (Lab {vote_auc_lab})\n"
        f"    Pole - : {pole_lo}\n"
        f"    Pole + : {pole_hi}",
        flush=True,
    )

print("ALL 20 TOPICS COMPLETE!")
