"""Bundle the quick map's results, MP photos and votes into one self-contained HTML page."""
import base64
import io
import json
import pathlib
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
TILE, COLS = 72, 28

TOPICS = ["Town centres & antisocial behaviour", "Sudan, Gaza & humanitarian aid", "Farming & animal welfare",
          "Courts & jury trials", "Steel & manufacturing", "Defence & NATO", "Asylum & small boats",
          "NHS services & cancer", "Scotland", "Child poverty & early years", "North Sea & clean energy",
          "Northern Ireland & the Troubles", "Academies & school standards", "Housing & renters", "Assisted dying",
          "Railways & transport", "Trans & LGBT rights", "Online safety & tech", "Dentists & rural services",
          "Chagos, China, Trump & Mandelson", "Wales", "Vaping & tobacco", "Tax, borrowing & the OBR",
          "Gaza, genocide & arms sales", "Race, Windrush & women's health", "Football & sport",
          "Cornwall, coast & fishing", "Water companies & sewage", "Violence against women & abuse", "Infected blood"]

q = np.load(ROOT / "data/quick/quick.npz", allow_pickle=True)
ids, xyz, shares, distinct, top_terms = q["ids"], q["xyz"], q["shares"], q["X_terms"], q["top_terms"]
members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
members[["party", "party_abbr", "constituency"]] = members[["party", "party_abbr", "constituency"]].fillna("Unknown")
members["party_colour"] = members["party_colour"].fillna("888888")
votes = pd.read_parquet(ROOT / "data/clean/vote_stats.parquet").set_index("member_id")
passages = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["member_id", "text", "role", "words", "major", "minor", "date"])

xyz = xyz - np.median(xyz, axis=0)
xyz = xyz / np.percentile(np.linalg.norm(xyz, axis=1), 95) * 10

# Nearest MPs in topic-share space
unit = shares / np.linalg.norm(shares, axis=1, keepdims=True)
sim = unit @ unit.T
np.fill_diagonal(sim, -1)

atlas = Image.new("RGB", (2048, 2048), (30, 50, 44))
groups = dict(tuple(passages.groupby("member_id")))
rows = []
for i, mid in enumerate(ids):
    m = members.loc[mid]
    r, c = i // COLS, i % COLS
    photo = ROOT / f"data/raw/photos/{mid}.jpg"
    has_photo = photo.exists()
    if has_photo:
        tile = Image.open(photo).convert("RGB")
        side = min(tile.size)
        tile = tile.crop(((tile.width - side) // 2, 0, (tile.width + side) // 2, side)).resize((TILE, TILE), Image.LANCZOS)
    else:
        tile = Image.new("RGB", (TILE, TILE), (70, 90, 84))
        initials = "".join(w[0] for w in re.sub(r"^(Mr|Mrs|Ms|Dr|Sir|Dame)\s+", "", m["name"]).split()[:2])
        ImageDraw.Draw(tile).text((TILE / 2, TILE / 2), initials, fill=(235, 235, 225), anchor="mm", font_size=26)
    atlas.paste(tile, (c * TILE, r * TILE))

    top = np.argsort(-shares[i])[:6]
    g = groups[mid]
    best = top_terms[top[0]][:14]
    pattern = re.compile("|".join(re.escape(t) for t in best), re.I)
    hits = g.text.str.count(pattern)
    quote = g.loc[hits.idxmax()]
    words = quote.text.split()
    v = votes.loc[mid] if mid in votes.index else None
    rows.append({
        "i": int(mid), "n": m["name"], "p": m["party_abbr"] or "?", "pn": m["party"] or "Unknown",
        "c": "#" + (m["party_colour"] or "888888"), "k": m["constituency"], "x": [round(float(a), 2) for a in xyz[i]],
        "t": [[int(k), round(float(shares[i, k]), 3)] for k in top], "s": [round(float(a), 3) for a in shares[i]],
        "w": list(distinct[i][:10]), "words": int(g.words.sum()), "min": round(float((g.role == "minister").mean()), 2),
        "nn": [int(a) for a in np.argsort(-sim[i])[:6]],
        "v": None if v is None else [int(v.votes_cast), int(v.rebellions)],
        "q": " ".join(words[:60]) + ("…" if len(words) > 60 else ""),
        "qd": ((quote.minor or quote.major or "") + " · " + quote.date).strip(" ·"), "f": bool(has_photo),
    })

for r in rows:
    if r["k"] is None:
        r["k"] = ""

buf = io.BytesIO()
atlas.save(buf, "JPEG", quality=78)
print("atlas KB", len(buf.getvalue()) // 1024)

data = {"mps": rows, "topics": TOPICS, "terms": [list(t[:8]) for t in top_terms], "cols": COLS, "tile": TILE, "size": 2048}
page = (HERE / "quick_map_template.html").read_text()
page = page.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
page = page.replace("__ATLAS__", "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
(HERE / "quick_map.html").write_text(page)
print("page KB", len(page.encode()) // 1024)

# Facts to check before saying anything about the map
party = np.array([r["p"] for r in rows])
big = np.isin(party, ["Lab", "Con", "LD", "SNP", "RUK", "Reform", "Green", "DUP", "PC"])
nn10 = np.argsort(-sim, axis=1)[:, :10]
print("parties", pd.Series(party).value_counts().to_dict())
print("10-NN same party, by party:", {p: round(float((party[nn10[party == p]] == p).mean()), 2) for p in ["Lab", "Con", "LD", "SNP"]})
print("expected if random:", {p: round(float((party == p).mean()), 2) for p in ["Lab", "Con", "LD", "SNP"]})
dom = shares.argmax(1)
for k, t in enumerate(TOPICS):
    s = pd.Series(party[dom == k]).value_counts().head(4).to_dict()
    print(f"{t:38} {s}")
