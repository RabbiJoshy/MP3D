"""Check the downloaded data is complete, against the live sources where possible.

    python pipeline/audit_data.py

Read-only apart from a few API requests. Prints PASS/CHECK lines plus the numbers behind them.
"""
import collections
import glob
import json
import pathlib
import re
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
START = "2024-07-09"
H = {"User-Agent": "MP3D audit", "Accept": "application/json"}


def line(ok, msg):
    print(("PASS  " if ok else "CHECK ") + msg, flush=True)


# ---------- divisions ----------
print("\n== VOTES ==")
divs = {}
for f in (RAW / "divisions").glob("*.json"):
    d = json.loads(f.read_text())
    divs[d["DivisionId"]] = d
live = requests.get("https://commonsvotes-api.parliament.uk/data/divisions.json/searchTotalResults",
                    params={"queryParameters.startDate": START}, headers=H, timeout=60).json()
line(len(divs) == live, f"divisions on disk {len(divs)}, live API count now {live}")
bad = [i for i, d in divs.items()
       if abs(len(d["Ayes"]) - d["AyeCount"]) > 2 or abs(len(d["Noes"]) - d["NoCount"]) > 2]
line(not bad, f"named voters match the official totals (within tellers) in {len(divs) - len(bad)}/{len(divs)} divisions; off: {bad[:10]}")
dates = sorted(d["Date"][:10] for d in divs.values())
print(f"      date range {dates[0]} to {dates[-1]}")

# ---------- members and photos ----------
print("\n== MPs ==")
member_files = list((RAW / "members").glob("*.json"))
voter_ids = {m["MemberId"] for d in divs.values() for s in ("Ayes", "Noes", "AyeTellers", "NoTellers", "NoVoteRecorded") for m in (d.get(s) or [])}
have = {int(f.stem) for f in member_files}
line(voter_ids <= have, f"profiles {len(have)}; everyone appearing in a division has one (missing {len(voter_ids - have)})")
empty_bio = [f.stem for f in member_files if not json.loads(f.read_text()).get("biography")]
line(not empty_bio, f"biographies with posts held: missing {len(empty_bio)}")
names = {}
for f in member_files:
    v = json.loads(f.read_text())["member"]["value"]
    names[v["id"]] = (v["nameDisplayAs"], (v.get("latestParty") or {}).get("abbreviation"), (v.get("latestHouseMembership") or {}).get("membershipEndDate"))
photos = {int(p.stem) for p in (RAW / "photos").glob("*.jpg")}
missing = have - photos
line(not missing, f"photos {len(photos)}; missing {len(missing)}: {[names[i][0] for i in sorted(missing)]}")
current = requests.get("https://members-api.parliament.uk/api/Members/Search", params={"House": 1, "IsCurrentMember": "true", "take": 1}, headers=H, timeout=60).json()["totalResults"]
print(f"      current MPs per API {current}; left since the election and captured via votes {sum(1 for v in names.values() if v[2])}")

# ---------- Hansard ----------
print("\n== SPEECHES ==")
people = json.loads((RAW / "people.json").read_text())
to_member = {p["id"]: int(i["identifier"]) for p in people["persons"] for i in p.get("identifiers", []) if i.get("scheme") == "datadotparl_id"}
files = {k: sorted((RAW / "hansard" / k).glob("*.xml")) for k in ("debates", "westminhall", "standing")}
sitting = set()
for k, fs in files.items():
    ds = sorted(re.search(r"\d{4}-\d{2}-\d{2}", f.name).group() for f in fs)
    print(f"      {k}: {len(fs)} files, {ds[0]} to {ds[-1]}")
    if k == "debates":
        sitting = set(ds)
no_debate = sorted({d[:10] for d in dates} - sitting)
line(not no_debate, f"every day with a division has a debate file; missing {no_debate[:20]}")

counts, unmatched, not_in_set, words = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
unparsable, not_latest, empty = [], [], []
for k, fs in files.items():
    for f in fs:
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError:
            unparsable.append(f.name)
            continue
        if root.get("latest") != "yes":
            not_latest.append(f.name)
        n = 0
        for s in root.iter("speech"):
            pid = s.get("person_id")
            if not pid:
                continue
            n += 1
            m = to_member.get(pid)
            if m is None:
                unmatched[(pid, s.get("speakername"))] += 1
            elif m not in have:
                not_in_set[(m, s.get("speakername"))] += 1
            else:
                counts[m] += 1
                words[m] += len(" ".join(s.itertext()).split())
        if n == 0:
            empty.append(f.name)
line(not unparsable, f"XML files that fail to parse: {len(unparsable)} {unparsable[:5]}")
line(not not_latest, f"files that aren't the latest published version: {len(not_latest)} {not_latest[:5]}")
line(not empty, f"files with no attributed speech: {len(empty)} {empty[:8]}")
total = sum(counts.values())
line(not unmatched, f"attributed speeches {total}; speaker id can't be matched to an MP: {sum(unmatched.values())} ({len(unmatched)} ids) {unmatched.most_common(6)}")
line(not not_in_set, f"speeches by someone with no MP profile: {sum(not_in_set.values())} {not_in_set.most_common(12)}")

silent = sorted(names[i][0] + f" ({names[i][1]})" for i in have if counts[i] == 0)
print(f"      MPs with zero raw speeches ({len(silent)}): {silent}")

# ---------- independent cross-check against the official Hansard API ----------
print("\n== CROSS-CHECK: official Hansard API vs bulk files ==")
sample_files = glob.glob(str(RAW / "speeches_api_sample" / "*.json"))
rows = []
for f in sample_files:
    m = int(pathlib.Path(f).stem)
    d = json.loads(open(f).read())
    if m in have:
        days = {r["SittingDate"][:10] for r in d["results"]}
        rows.append((names[m][0], d["total"], counts[m], len(d["results"]), len(days)))
x = pd.DataFrame(rows, columns=["name", "api_total", "bulk_speeches", "api_rows", "api_days"])
if len(x):
    x["ratio"] = x.bulk_speeches / x.api_total.clip(lower=1)
    print(f"      {len(x)} MPs with API data: API contributions {x.api_total.sum()}, bulk speeches {x.bulk_speeches.sum()}; "
          f"bulk/API ratio median {x.ratio.median():.2f}, min {x.ratio.min():.2f}, max {x.ratio.max():.2f}")
    print(x.sort_values("ratio").head(5).round(2).to_string(index=False))
    print("      (the API snapshot stopped earlier today, so bulk can be slightly higher)")
else:
    print("      no API sample on disk")

# ---------- clean layer ----------
print("\n== CLEANED ==")
p = pd.read_parquet(ROOT / "data/clean/passages.parquet")
print(f"      passages {len(p)}, words {p.words.sum()}, MPs {p.member_id.nunique()}, dates {p.date.min()} to {p.date.max()}")
print(f"      raw report: {json.loads((ROOT / 'data/clean/clean_report.json').read_text())}")
print(f"\nAudit run {time.strftime('%Y-%m-%d %H:%M')}")
