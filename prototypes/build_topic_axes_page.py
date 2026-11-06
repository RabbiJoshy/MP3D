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
raw = json.loads((ROOT / "data/quick/topic_axes.json").read_text())
topics = [t for t in raw["topics"] if not t.get("skip")]
comb = raw["combined"]
composites = [raw["care"], raw["stance"], comb]
comb["axes"] = [{"lo": [], "hi": [], "grade": "strong", "party": d["party"]} for d in comb["dims"]]
comb["quotes"] = {}
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


ids = sorted({p[0] for t in topics for p in t["pts"]} | {p[0] for c in composites for p in c["pts"]})
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
used = [t for t in topics if t["n_axes"] > 0]
# Stance-likeness of each topic's main axis: average rank of three unsupervised signals (see topic_axes.py)
st = pd.DataFrame([{"i": i, "codebate": x["axes"][0]["codebate"], "dominance": x["axes"][0]["dominance"], "bimodal": x["axes"][0]["bimodal"]} for i, x in enumerate(topics)])
st["score"] = (st.codebate.rank(pct=True) + st.dominance.rank(pct=True) + st.bimodal.rank(pct=True)) / 3
st["rank"] = st.score.rank(ascending=False, method="min").astype(int)
for r in st.itertuples():
    topics[r.i]["stance_rank"] = int(r.rank)
    topics[r.i]["stance_score"] = round(float(r.score), 3)
buf = io.BytesIO()
atlas.save(buf, "JPEG", quality=76)
# A combined axis blends many topic axes, so each end is named by its strongest few topics, not one
def short(e):
    head, _, tail = e["topic"].partition(":")
    return head.split(",")[0].strip() + (f": {tail.split(',')[0].strip()}" if tail else "")
for c in composites:
    c["name"] = [c.get("title") or "All divides combined"]
    c["axes"] = [{"lo": [short(e) for e in d["lo"]], "hi": [short(e) for e in d["hi"]], "grade": "strong", "party": d["party"]} for d in c["dims"]]
    c["quotes"] = {}
topics = composites + topics
data = {"topics": topics, "mps": mps, "byId": {m["i"]: m for m in mps}, "tiles": tiles, "tile": TILE, "cols": COLS, "size": 2048,
        "stats": {"topicsRanked": len(st), "passages": int(comb["n"]), "axesUsed": int(sum(t["n_axes"] for t in used)), "topicsUsed": len(used)}}
page = (HERE / "topic_axes_template.html").read_text()
page = page.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")).replace("__ATLAS__", "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
(HERE / "topic_axes.html").write_text(page)
print("page MB", round(len(page.encode()) / 1e6, 1), "| MPs", len(mps), "| photos", placed, "| topics", len(topics))
