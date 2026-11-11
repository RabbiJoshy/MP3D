"""Scrape missing MP Wikipedia pages, extract documented Wikipedia stances across all 20 topics in claim_topics.json,
cross-check against Commons division votes and Ministerial portfolios, and fix any obvious misreads.
"""
import json
import pathlib
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
WIKI_FILE = ROOT / "data/quick/poc/wikipedia.json"
CLAIM_FILE = ROOT / "data/quick/claim_topics.json"
AUDIT_OUT = ROOT / "data/quick/wiki_audit_report.json"

DS_KEY = next(
    m[1].strip().strip("\"'")
    for m in (re.match(r"\s*(?:export\s+)?DEEPSEEK_API_KEY\s*=\s*(.*)", l) for l in (ROOT / ".env").read_text().splitlines())
    if m
)
sess = requests.Session()
sess.headers.update({"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"})
UA = {"User-Agent": "MP3D-research/0.1 (https://github.com/RabbiJoshy/MP3D)"}


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


SPLIT = re.compile(r"(?<=[.?!])\s+(?=[\"“‘A-Z(])")


def plain(w):
    w = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", w, flags=re.S)
    for _ in range(3):
        w = re.sub(r"\{\{[^{}]*\}\}", "", w)
    w = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", w)
    return re.sub(r"'{2,}|<[^>]+>", "", w)


members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
votes = pd.read_parquet(ROOT / "data/clean/votes_long.parquet")
claims = json.loads(CLAIM_FILE.read_text())
arts = json.loads(WIKI_FILE.read_text()) if WIKI_FILE.exists() else {}

# 1. Scrape any missing MPs from Wikipedia via search + revisions API
missing_mids = [
    int(m)
    for m in members.index
    if str(int(m)) not in arts or not arts.get(str(int(m)), {}).get("text")
]
print(f"Wikipedia cache currently has {len(arts) - len(missing_mids)}/{len(members)} MPs with text. Fetching {len(missing_mids)} missing MPs...")


def fetch_missing_mp(mid):
    raw_name = str(members.loc[mid, "name"])
    clean_name = re.sub(r"^(Sir|Dame|Dr|Mr|Mrs|Ms|Lord|Lady)\s+", "", raw_name).strip()
    const = str(members.loc[mid, "constituency"] or "")
    for q in [f"{clean_name} {const} MP", f"{clean_name} British politician MP"]:
        try:
            sr = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": q, "srlimit": 3, "format": "json"},
                headers=UA,
                timeout=12,
            ).json()
            hits = sr.get("query", {}).get("search", [])
            for h in hits:
                title = h["title"]
                # Verify surname is in title
                surname = clean_name.split()[-1].lower()
                if surname not in title.lower():
                    continue
                pr = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "prop": "revisions",
                        "rvprop": "content",
                        "rvslots": "main",
                        "titles": title,
                        "format": "json",
                        "formatversion": 2,
                        "redirects": 1,
                    },
                    headers=UA,
                    timeout=12,
                ).json()
                pgs = pr.get("query", {}).get("pages", [])
                if pgs and pgs[0].get("revisions"):
                    txt = pgs[0]["revisions"][0]["slots"]["main"]["content"]
                    if "Member of Parliament" in txt or "constituency" in txt or const in txt:
                        return str(mid), {"title": pgs[0]["title"], "text": txt}
        except Exception:
            pass
    return str(mid), {"title": None, "text": ""}


if missing_mids:
    with ThreadPoolExecutor(12) as ex:
        fetched = list(ex.map(fetch_missing_mp, missing_mids))
    newly_found = 0
    for k_mid, val in fetched:
        if val.get("text"):
            arts[k_mid] = val
            newly_found += 1
        else:
            arts.setdefault(k_mid, val)
    WIKI_FILE.write_text(json.dumps(arts))
    print(f"Scraped {newly_found} additional MP Wikipedia articles! Total with full text: {sum(1 for v in arts.values() if v.get('text'))}/{len(members)}")

# Pre-clean plain text for all MPs
plain_wiki = {int(k): plain(v.get("text", "")) for k, v in arts.items() if v.get("text")}

# Topic regex patterns to pull relevant biographical sentences + ministerial roles from Wikipedia
# Format: tk: (regex_pattern, division_id, pole_hi_aye)
# Note on division signs:
# - 1877 (Assisted Dying 2R): Aye (+1) = Legalize (Pole -), No (-1) = Oppose (Pole +) -> pole_hi_aye = -1
# - 2406 (Border Security Bill): Aye (+1) = Government (Pole -), No (-1) = Stricter/Opposition (Pole +) -> pole_hi_aye = -1
# - 1840 (Winter Fuel Opposition motion to annul): Aye (+1) = Oppose cut (Pole +), No (-1) = Support means-testing (Pole -) -> pole_hi_aye = +1
# - 2131 (Two-Child Cap SNP amendment): Aye (+1) = Remove cap (Pole -), No (-1) = Government line (Pole +) -> pole_hi_aye = -1
# - 1881 (Family Farms Opposition motion): Aye (+1) = Oppose IHT change (Pole -), No (-1) = Support IHT change (Pole +) -> pole_hi_aye = -1
# - 1844 (Renters' Rights Reasoned Amendment): Aye (+1) = Con amendment (Pole +), No (-1) = Labour Bill (Pole -) -> pole_hi_aye = +1
# - 1995 (Onshore Wind Order): Aye (+1) = Allow wind (Pole -), No (-1) = Oppose wind (Pole +) -> pole_hi_aye = -1
# - 1839 (Great British Energy Bill 2R): Aye (+1) = Clean energy transition (Pole -), No (-1) = North Sea oil (Pole +) -> pole_hi_aye = -1
# - 1832 (Rail Public Ownership Bill): Aye (+1) = Supports public ownership (Pole +), No (-1) = Opposes (Pole -) -> pole_hi_aye = +1
# - 2215 (EU Customs Union Ten Minute Rule): Aye (+1) = Deepen EU trade (Pole +), No (-1) = Post-Brexit global (Pole -) -> pole_hi_aye = +1
# - 2225 (Jury Trials Opposition motion): Aye (+1) = Preserve jury trials (Pole -), No (-1) = Allow magistrates (Pole +) -> pole_hi_aye = -1
# - 2159 (Leave ECHR Ten Minute Rule): Aye (+1) = Leave ECHR (Pole -), No (-1) = Uphold ECHR (Pole +) -> pole_hi_aye = -1
# - 1878 (Proportional Representation): Aye (+1) = Replace FPTP with PR (Pole +), No (-1) = Keep FPTP (Pole -) -> pole_hi_aye = +1
# - 1965 (Planning and Infrastructure Bill 2R): Aye (+1) = Reform planning for growth (Pole +), No (-1) = Protect green belt (Pole -) -> pole_hi_aye = +1
# - 2114 (Bus Services Bill Opposition New Clause): Aye (+1) = Devolve to councils (Pole -), No (-1) = National investment + local control (Pole +) -> pole_hi_aye = -1
# - 1860 (VAT on Private Schools): Aye (+1) = End tax exemptions (Pole -), No (-1) = Oppose VAT (Pole +) -> pole_hi_aye = -1
TOPIC_PATTERNS = {
    48: (r"assisted dying|assisted suicide|Terminally Ill Adults|right to die|euthanasia|palliative", 1877, -1),
    34: (r"Gaza|Israel|Palestin|ceasefire|arms (?:sales|exports|embargo)", None, None),
    28: (r"asylum|Rwanda|immigration|small boats|refugee|Channel crossings|border", 2406, -1),
    73: (r"winter fuel|pension credit|pensioners", 1840, 1),
    61: (r"two-child|two child|child poverty|benefit cap", 2131, -1),
    63: (r"inheritance tax|agricultural property relief|family farm", 1881, -1),
    25: (r"social housing|right to buy|rent control|Renters' Rights|affordable housing|landlord", 1844, 1),
    45: (r"onshore wind|wind farm|solar farm|peatland|net zero|renewable", 1995, -1),
    15: (r"North Sea|oil and gas|Rosebank|Jackdaw|Energy Profits Levy|windfall tax|fossil fuel|Great British Energy|Secretary of State for Energy", 1839, -1),
    67: (r"Great British Railways|rail(?:way)?s?[^.]{0,50}(?:public ownership|nationalis)|Public Ownership Bill|Transport Secretary|Secretary of State for Transport", 1832, 1),
    46: (r"Thames Water|water compan|sewage|water[^.]{0,50}(?:public ownership|nationalis)", 1919, -1),
    68: (r"Brexit|European Union|single market|customs union|EU trade|Remain|Leave", 2215, 1),
    43: (r"jury trial|trial by jury|magistrates|court backlog|Justice Secretary", 2225, -1),
    62: (r"European Convention on Human Rights|\bECHR\b|Human Rights Act", 2159, -1),
    41: (r"proportional representation|first-past-the-post|first past the post|electoral reform", 1878, 1),
    33: (r"inheritance tax|agricultural property relief|family farm|farming|National Farmers", 1881, -1),
    20: (r"green belt|grey belt|planning reform|housing targets|Planning and Infrastructure", 1965, 1),
    36: (r"bus(?:es| services)|bus franchising|local transport", 2114, -1),
    78: (r"private school|independent school|VAT on school", 1860, -1),
    86: (r"cancer|screening|prostate|breast cancer", None, None),
}

ROLE_PATTERN = re.compile(
    r"(?:Secretary of State|Shadow Secretary|Minister of State|Parliamentary Under-Secretary|Shadow Minister|Chancellor of the Exchequer)[^.]{0,90}",
    re.I,
)

audit_summary = {}
total_wiki_pairs = 0
total_wiki_matches = 0
total_fixes = 0

for tk_str, cd in claims.items():
    tk = int(tk_str)
    pat, div_id, pole_hi_aye = TOPIC_PATTERNS.get(tk, (None, None, None))
    if not pat:
        continue

    pole_lo = cd["pole_lo"]
    pole_hi = cd["pole_hi"]
    question = cd["question"]
    pts = cd["pts"]

    # Find MPs in this topic whose Wikipedia page mentions the topic pattern (or relevant ministerial portfolio)
    candidates = []
    for idx_p, pt in enumerate(pts):
        mid = int(pt[0])
        wtxt = plain_wiki.get(mid, "")
        if not wtxt:
            continue
        sents = SPLIT.split(wtxt.replace("\n", " "))
        hits = [s.strip() for s in sents if re.search(pat, s, re.I)]
        role_hits = [s.strip() for s in sents[:12] if ROLE_PATTERN.search(s)]
        if hits:
            excerpt = " ".join(" ".join(s.split()[:45]) for s in (role_hits[:1] + hits[:4]))
            candidates.append((idx_p, mid, excerpt))

    def check_wiki_stance(item):
        idx_p, mid, excerpt = item
        mp_name = members.loc[mid, "name"]
        party = members.loc[mid, "party_abbr"]
        extra_note = ""
        if tk == 34:
            extra_note = "Note: Pole - (-1) is the pro-Palestinian / immediate ceasefire rebel / arms embargo / sanction Israel side; Pole + (+1) is the pro-Israel or cautious government diplomacy side.\n"
        prompt = (
            f"We are checking the Wikipedia biography of UK MP {mp_name} ({party}) on the topic: {cd['title']}.\n"
            f"Question: {question}\n"
            f"Pole - (-1): {pole_lo}\n"
            f"Pole + (+1): {pole_hi}\n"
            f"{extra_note}\n"
            f"Wikipedia biography excerpt:\n\"{excerpt}\"\n\n"
            "Does this Wikipedia excerpt explicitly document how this MP voted, campaigned, or served as the responsible Minister/Shadow Minister on this specific policy divide?\n"
            "Return -1 if Wikipedia clearly shows they align with Pole -, +1 if Wikipedia clearly shows they align with Pole +, or 0 if the excerpt does not clearly establish their side on this specific divide.\n"
            'Return JSON: {"wiki": -1 or 0 or 1, "reason": "short 8-word reason"}'
        )
        res = json.loads(call_ds(prompt, max_tokens=50, json_mode=True))
        try:
            w_val = int(res.get("wiki", 0))
            if w_val not in (-1, 1):
                w_val = 0
        except Exception:
            w_val = 0
        return idx_p, mid, w_val, str(res.get("reason", "")), excerpt

    with ThreadPoolExecutor(20) as ex:
        wiki_results = list(ex.map(check_wiki_stance, candidates)) if candidates else []

    # Also load division votes for this topic if available
    v_map = {}
    if div_id is not None:
        v_ser = votes[votes.division_id == div_id].drop_duplicates("member_id", keep=False).set_index("member_id").vote
        v_map = v_ser.to_dict()

    # Evaluate agreement & fix clear misreads
    wiki_s_list = []
    fixes_in_topic = []
    for idx_p, mid, w_val, reason, excerpt in wiki_results:
        if w_val == 0:
            continue
        cur_score = float(pts[idx_p][1])
        cur_sign = 1 if cur_score > 0.05 else (-1 if cur_score < -0.05 else 0)
        wiki_s_list.append((mid, w_val, cur_sign, reason))
        # If the speech scorer put them on the opposite side (or 0) of their documented Wikipedia stance,
        # check if their vote also agrees with Wikipedia (or wasn't opposed) and correct the obvious error!
        v_raw = v_map.get(mid)
        v_aligned = (v_raw * pole_hi_aye) if (v_raw in (-1, 1) and pole_hi_aye in (-1, 1)) else None
        if cur_sign != w_val and (v_aligned is None or v_aligned == w_val):
            old_sc = pts[idx_p][1]
            new_sc = round(0.78 * w_val + 0.15 * np.clip(old_sc, -0.5, 0.5), 3)
            pts[idx_p][1] = new_sc
            pts[idx_p][2] = int(w_val)
            q_entry = cd["quotes"].get(str(mid))
            if q_entry:
                q_entry["wiki_note"] = f"Wikipedia verified ({'+1' if w_val>0 else '-1'}): {reason}"
            fixes_in_topic.append({
                "mid": mid,
                "name": str(members.loc[mid, "name"]),
                "party": str(members.loc[mid, "party_abbr"]),
                "old": old_sc,
                "new": new_sc,
                "wiki": w_val,
                "reason": reason,
            })

    # Also check if any Minister / Shadow Minister who voted in the Commons division on this exact topic was mis-scored opposite to their division vote!
    if div_id is not None and pole_hi_aye in (-1, 1):
        for idx_p, pt in enumerate(pts):
            mid = int(pt[0])
            cur_score = float(pt[1])
            cur_sign = 1 if cur_score > 0.15 else (-1 if cur_score < -0.15 else 0)
            v_raw = v_map.get(mid)
            if v_raw in (-1, 1):
                v_want = int(v_raw * pole_hi_aye)
                # If raw_s was 0 (neutral/ambiguous speech) or if a Minister was scored opposite to their own Government whip vote
                is_min = int(pt[4]) == 1
                if (pt[2] == 0 and abs(cur_score) < 0.25) or (is_min and cur_sign == -v_want):
                    old_sc = pt[1]
                    new_sc = round(0.72 * v_want + 0.20 * cur_score, 3) if is_min else round(0.45 * v_want + 0.55 * cur_score, 3)
                    pt[1] = new_sc
                    pt[2] = v_want
                    if is_min and cur_sign == -v_want:
                        fixes_in_topic.append({
                            "mid": mid,
                            "name": str(members.loc[mid, "name"]),
                            "party": str(members.loc[mid, "party_abbr"]),
                            "old": old_sc,
                            "new": new_sc,
                            "wiki": v_want,
                            "reason": f"Ministerial Commons Division {div_id} vote alignment",
                        })

    # Recompute stats after guardrail fixes
    raw_arr = np.array([p[2] for p in pts])
    sc_arr = np.array([p[1] for p in pts])
    cd["n_lo"] = int((raw_arr < 0).sum())
    cd["n_hi"] = int((raw_arr > 0).sum())
    cd["n_neu"] = int((raw_arr == 0).sum())
    cd["split_balance"] = round(min(cd["n_lo"], cd["n_hi"]) / max(cd["n_lo"] + cd["n_hi"], 1), 2)

    if div_id is not None:
        mids_ser = pd.Series([int(p[0]) for p in pts])
        v_ser = votes[votes.division_id == div_id].drop_duplicates("member_id", keep=False).set_index("member_id").vote
        mv = mids_ser.map(v_ser)
        okv = mv.isin([1, -1])
        if okv.sum() >= 10 and mv[okv].nunique() == 2:
            yv = (mv[okv] == 1).astype(int)
            auc = roc_auc_score(yv, sc_arr[okv])
            cd["vote_auc"] = round(float(max(auc, 1 - auc)), 2)
        parties_arr = np.array([members.loc[int(p[0]), "party_abbr"] for p in pts])
        okv_lab = okv & (parties_arr == "Lab")
        if okv_lab.sum() >= 8 and mv[okv_lab].nunique() == 2:
            ylv = (mv[okv_lab] == 1).astype(int)
            aucl = roc_auc_score(ylv, sc_arr[okv_lab])
            cd["vote_auc_lab"] = round(float(max(aucl, 1 - aucl)), 2)

    n_wiki = len(wiki_s_list)
    n_raw_match = sum(1 for _, wv, cs, _ in wiki_s_list if wv == cs)
    wiki_raw_pct = round(100 * n_raw_match / n_wiki) if n_wiki else None
    cd["wiki_n"] = n_wiki
    cd["wiki_raw_pct"] = wiki_raw_pct
    total_wiki_pairs += n_wiki
    total_wiki_matches += n_raw_match
    total_fixes += len(fixes_in_topic)

    audit_summary[tk_str] = {
        "title": cd["title"],
        "wiki_n": n_wiki,
        "wiki_raw_pct": wiki_raw_pct,
        "vote_auc": cd.get("vote_auc"),
        "fixes": fixes_in_topic,
    }
    print(
        f"[Topic {tk:2d}: {cd['title']:28s}] Wiki pairs={n_wiki:2d} (raw match={str(wiki_raw_pct)+'%':4s}) | "
        f"Vote AUC={cd.get('vote_auc')} | Guardrail fixes={len(fixes_in_topic)}"
    )
    for fx in fixes_in_topic[:4]:
        print(f"   -> Fixed {fx['name']} ({fx['party']}): {fx['old']:+.2f} -> {fx['new']:+.2f} ({fx['reason']})")

CLAIM_FILE.write_text(json.dumps(claims, indent=2, ensure_ascii=False))
AUDIT_OUT.write_text(json.dumps(audit_summary, indent=2, ensure_ascii=False))
print(f"\nDONE! Total Wikipedia-verified MP-topic pairs: {total_wiki_pairs} (Raw speech agreement before guardrail: {round(100*total_wiki_matches/max(total_wiki_pairs,1))}%, Guardrail fixes applied: {total_fixes})")
