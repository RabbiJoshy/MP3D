"""Bundle per-topic axes, MP details and photos into one self-contained page. Names are the data's own words."""
import base64
import io
import json
import pathlib

import pandas as pd
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
TILE, COLS = 72, 28
topics = [t for t in json.loads((ROOT / "data/quick/topic_axes.json").read_text()) if not t.get("skip")]
members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
members[["party", "party_abbr"]] = members[["party", "party_abbr"]].fillna("Unknown")
members["party_colour"] = members["party_colour"].fillna("888888")


def short_name(words, n=3):
    out = []
    for w in words:
        if not any(w in x or x in w for x in out):
            out.append(w)
        if len(out) == n:
            break
    return out


ids = sorted({p[0] for t in topics for p in t["pts"]})
atlas = Image.new("RGB", (2048, 2048), (30, 50, 44))
tiles, mps = {}, []
placed = 0
for mid in ids:
    m = members.loc[mid]
    photo = ROOT / f"data/raw/photos/{mid}.jpg"
    if photo.exists() and placed < COLS * COLS:
        img = Image.open(photo).convert("RGB")
        side = min(img.size)
        img = img.crop(((img.width - side) // 2, 0, (img.width + side) // 2, side)).resize((TILE, TILE), Image.LANCZOS)
        atlas.paste(img, (placed % COLS * TILE, placed // COLS * TILE))
        tiles[mid] = placed
        placed += 1
    mps.append({"i": int(mid), "n": m["name"], "p": m["party_abbr"], "pn": m["party"], "c": "#" + m["party_colour"], "k": m["constituency"] if isinstance(m["constituency"], str) else ""})
for t in topics:
    t["name"] = short_name(t["words"])
    t["quotes"] = {int(k): v for k, v in t["quotes"].items()}
buf = io.BytesIO()
atlas.save(buf, "JPEG", quality=76)
data = {"topics": topics, "mps": mps, "byId": {m["i"]: m for m in mps}, "tiles": tiles, "tile": TILE, "cols": COLS, "size": 2048,
        "stats": {"passages": int(sum(t["n"] for t in topics))}}
page = (HERE / "topic_axes_template.html").read_text()
page = page.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")).replace("__ATLAS__", "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
(HERE / "topic_axes.html").write_text(page)
print("page MB", round(len(page.encode()) / 1e6, 1), "| MPs", len(mps), "| photos", placed, "| topics", len(topics))
