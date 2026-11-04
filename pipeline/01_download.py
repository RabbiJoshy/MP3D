"""Download the raw data for the current parliament. Safe to re-run: finished items are skipped.

    python pipeline/01_download.py divisions   # every Commons division + how each MP voted
    python pipeline/01_download.py members     # every MP who sat: profile, posts held, photo
    python pipeline/01_download.py speeches    # daily Hansard XML: Commons, Westminster Hall, bill committees

Output goes to data/raw/ (git-ignored).
"""
import json
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

START, END = "2024-07-09", time.strftime("%Y-%m-%d")  # first sitting of the 59th parliament -> today
RAW = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
HEADERS = {"User-Agent": "MP3D research project", "Accept": "application/json"}
session = requests.Session()
session.headers.update(HEADERS)
adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8)
session.mount("https://", adapter)


def get(url, params=None, raw=False):
    for attempt in range(6):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content if raw else r.json()
        except Exception as e:  # network blips and API 5xx: back off and retry
            if attempt == 5:
                raise RuntimeError(f"{url} {params}: {e}")
            time.sleep(2 ** attempt)


def save_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    tmp.rename(path)


def run_all(items, fn, workers=6, label=""):
    done = failed = 0
    with ThreadPoolExecutor(workers) as ex:
        futures = {ex.submit(fn, i): i for i in items}
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                failed += 1
                print(f"FAILED {futures[f]}: {e}", flush=True)
            done += 1
            if done % 50 == 0 or done == len(items):
                print(f"{label} {done}/{len(items)} failed={failed}", flush=True)
    return failed


def divisions():
    out = RAW / "divisions"
    out.mkdir(parents=True, exist_ok=True)
    ids, skip = [], 0
    while True:
        page = get(
            "https://commonsvotes-api.parliament.uk/data/divisions.json/search",
            {"queryParameters.startDate": START, "queryParameters.endDate": END,
             "queryParameters.take": 25, "queryParameters.skip": skip},
        )
        if not page:
            break
        ids += [d["DivisionId"] for d in page]
        skip += len(page)
    ids = sorted(set(ids))
    print("divisions found", len(ids), flush=True)

    def one(i):
        p = out / f"{i}.json"
        if not p.exists():
            save_json(p, get(f"https://commonsvotes-api.parliament.uk/data/division/{i}.json"))

    return run_all(ids, one, label="divisions")


def member_ids():
    ids, skip = set(), 0
    while True:
        d = get("https://members-api.parliament.uk/api/Members/Search",
                {"House": 1, "IsCurrentMember": "true", "skip": skip, "take": 20})
        ids |= {m["value"]["id"] for m in d["items"]}
        skip += 20
        if skip >= d["totalResults"]:
            break
    n_current = len(ids)
    # MPs who have since left only show up through their votes
    for f in (RAW / "divisions").glob("*.json"):
        d = json.loads(f.read_text())
        for side in ("Ayes", "Noes", "NoVoteRecorded", "AyeTellers", "NoTellers"):
            ids |= {m["MemberId"] for m in (d.get(side) or [])}
    print(f"members: {n_current} current, {len(ids)} including those who left", flush=True)
    return sorted(ids)


def members():
    out, photos = RAW / "members", RAW / "photos"
    out.mkdir(parents=True, exist_ok=True)
    photos.mkdir(parents=True, exist_ok=True)

    def one(i):
        p = out / f"{i}.json"
        if not p.exists():
            base = f"https://members-api.parliament.uk/api/Members/{i}"
            save_json(p, {"member": get(base), "biography": get(base + "/Biography")})
        ph = photos / f"{i}.jpg"
        if not ph.exists():
            img = get(f"https://members-api.parliament.uk/api/Members/{i}/Portrait",
                      {"cropType": "OneOne", "webVersion": "true"}, raw=True)
            if not img:  # some MPs have no portrait, only a thumbnail; PNG thumbnails are silhouettes
                img = get(f"https://members-api.parliament.uk/api/Members/{i}/Thumbnail", raw=True)
                if not img or not img.startswith(b"\xff\xd8"):
                    img = None
            if img:
                ph.write_bytes(img)

    return run_all(member_ids(), one, label="members")


def speeches():
    """Bulk daily Hansard XML from TheyWorkForYou/ParlParse. The official per-MP search API
    was far too slow (about 50 hours for the whole Commons)."""
    base = "https://www.theyworkforyou.com/pwdata/scrapedxml"
    pattern = re.compile(r'href="([^"]+?(\d{4}-\d{2}-\d{2})([a-z])\.xml)"')
    wanted = []
    for kind in ("debates", "westminhall", "standing"):
        (RAW / "hansard" / kind).mkdir(parents=True, exist_ok=True)
        html = session.get(f"{base}/{kind}/", timeout=120).text
        newest = {}  # a day can be republished as a, b, c...: keep only the last version
        for name, day, letter in pattern.findall(html):
            if day >= START:
                key = name[: -len(letter) - 4]
                if key not in newest or letter > newest[key][1]:
                    newest[key] = (name, letter)
        wanted += [(kind, name) for name, _ in newest.values()]
        print(kind, len(newest), "files", flush=True)

    def one(item):
        kind, name = item
        p = RAW / "hansard" / kind / name
        if not p.exists():
            p.write_bytes(get(f"{base}/{kind}/{name}", raw=True))

    (RAW / "people.json").write_bytes(
        get("https://raw.githubusercontent.com/mysociety/parlparse/master/members/people.json", raw=True))
    return run_all(wanted, one, label="hansard")


if __name__ == "__main__":
    sys.exit(1 if {"divisions": divisions, "members": members, "speeches": speeches}[sys.argv[1]]() else 0)
