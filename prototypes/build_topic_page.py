"""Bundle the topic space, regions, MP photos and votes into one self-contained HTML page."""
import base64
import collections
import io
import json
import pathlib
import re

import numpy as np
import pandas as pd
import umap
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
TILE, COLS, MIN_WORDS = 72, 28, 2000
rng = np.random.default_rng(7)

REGIONS = {0: "Economy, business & towns", 1: "Housing, councils & transport", 2: "World affairs & rights",
           3: "Health & care", 4: "Crime, justice & borders", 5: "Tax, spending & welfare", 6: "Energy",
           7: "Schools & children", 8: "Countryside & environment", 9: "Defence & Ukraine", 10: "The nations"}
REGION_COLOURS = ["#f2c14e", "#f78154", "#ff7eb6", "#c6a4ff", "#64b5f6", "#5fe0d0", "#fff27a", "#ffb3c7", "#a5e27a", "#e8ecef", "#9db8ff"]
TOPICS = {7: "Economic growth", 10: "Faith & community cohesion", 18: "Small businesses", 21: "War memory & history",
          24: "Business rates & pubs", 27: "Young people & skills", 35: "Football", 44: "High streets",
          69: "Coastal & left-behind places", 72: "Steel & industry", 89: "Funding formulas", 20: "Planning & green belt",
          25: "Renters & homelessness", 32: "Mayors & devolution", 36: "Buses", 67: "Rail", 75: "Road safety",
          77: "Councils & council tax", 81: "Affordable homes", 14: "National security & China", 34: "Gaza & Israel",
          41: "Elections & voting", 55: "Chagos & Diego Garcia", 62: "Human rights & the ECHR", 65: "Mandelson & Epstein",
          66: "International law", 68: "EU, trade & Brexit", 4: "Hospitals & patients", 23: "Mental health",
          48: "Assisted dying", 49: "NHS waits & dentists", 53: "Bereaved families", 54: "Serious illness",
          58: "Social care", 86: "Cancer", 3: "Child sexual abuse", 17: "Domestic abuse", 28: "Asylum & small boats",
          29: "Online safety", 31: "Vaping & tobacco", 38: "Infected blood & compensation", 42: "Victims & criminal justice",
          43: "Courts & juries", 59: "Policing & antisocial behaviour", 74: "Royal Mail & civil service", 80: "Data & digital",
          87: "Women & girls", 1: "The Budget & the OBR", 30: "Pensions", 39: "Billions & the black hole",
          40: "Taxpayers' money", 61: "Child poverty", 63: "Inheritance & income tax", 64: "National insurance",
          73: "Winter fuel", 15: "North Sea oil & gas", 45: "Net zero & clean energy", 84: "Energy bills",
          5: "Schools & teachers", 50: "Children & families", 78: "Special educational needs", 6: "Rural services",
          11: "Climate change", 33: "Farming & food", 46: "Water & sewage", 83: "Animal welfare", 16: "Defence spending",
          22: "Ukraine & Russia", 47: "Armed forces & veterans", 9: "Wales", 56: "Scotland", 71: "Northern Ireland"}

q = np.load(ROOT / "data/quick/topic_space.npz", allow_pickle=True)
lay = np.load(ROOT / "data/quick/topic_layout.npz")
is_chatter, counts, ids, topic_of, sim_of = q["is_chatter"], q["counts"], q["ids"], q["topic"], q["sim"]
star_xyz, star_topic, star_idx, umap_centre, top_terms = q["star_xyz"], q["star_topic"], q["star_idx"], q["topic_xyz"], q["top_terms"]
topic_pos, region = lay["topic_pos"], lay["region"]
real = np.where(~is_chatter)[0]
assert set(real.tolist()) == set(TOPICS), set(real.tolist()) ^ set(TOPICS)
new = {k: i for i, k in enumerate(real)}

members = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
members[["party", "party_abbr", "constituency"]] = members[["party", "party_abbr", "constituency"]].fillna("Unknown")
members["party_colour"] = members["party_colour"].fillna("888888")
votes = pd.read_parquet(ROOT / "data/clean/vote_stats.parquet").set_index("member_id")
passages = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["member_id", "text", "role", "words", "major", "minor", "date"])
assert len(passages) == len(topic_of)
passages["topic"], passages["sim"] = topic_of, sim_of
chatter_share = float(is_chatter[topic_of].mean())

c = counts[:, real].astype("float64")
words = c.sum(1)
keep = words >= MIN_WORDS
ids_all, ids = ids, ids[keep]
c = c[keep]
share = c / c.sum(1, keepdims=True)
overall = c.sum(0) / c.sum()
lift = share / overall
pos_real = topic_pos[real]

# An MP lives at their main topic, pulled a little towards their other strong ones
mp_xyz = np.zeros((len(ids), 3))
for i in range(len(ids)):
    score = share[i] * np.log1p(lift[i])
    top = np.argsort(-score)[:4]
    w = score[top] ** 2
    blend = (pos_real[top] * w[:, None]).sum(0) / w.sum()
    mp_xyz[i] = 0.72 * pos_real[top[0]] + 0.28 * blend + rng.normal(0, 0.3, 3)

distinct = {}
quick = ROOT / "data/quick/quick.npz"
if quick.exists():
    qq = np.load(quick, allow_pickle=True)
    distinct = {int(a): list(b)[:10] for a, b in zip(qq["ids"], qq["X_terms"])}

# People space: every MP as their whole topic mix, compared with the Commons average.
# Quiet MPs are shrunk towards the average, so a few speeches can't place them at an extreme.
SHRINK = 4000
def mix_vector(word_counts):
    smoothed = (word_counts + SHRINK * overall) / (word_counts.sum(-1, keepdims=True) + SHRINK)
    return np.sqrt(smoothed) - np.sqrt(overall)
V = mix_vector(c)
unit = V / np.linalg.norm(V, axis=1, keepdims=True)
sim = unit @ unit.T
np.fill_diagonal(sim, -1)
reducer = umap.UMAP(n_components=3, n_neighbors=12, min_dist=0.18, metric="cosine", random_state=7, n_jobs=1)
people_raw = reducer.fit_transform(V)
people_centre = np.median(people_raw, axis=0)
people_scale = 14 / np.percentile(np.linalg.norm(people_raw - people_centre, axis=1), 95)
people_xyz = (people_raw - people_centre) * people_scale
cluster = KMeans(12, n_init=10, random_state=7).fit_predict(unit)

# Real Commons votes. "yes" says which lobby a Yes answer corresponds to: 1 = Aye, -1 = No.
# Wording checked against the division title, the motion in Hansard and how parties voted.
QUESTIONS = [
    {"id": 2071, "yes": 1, "q": "Should terminally ill adults be able to choose assisted dying?", "n": "Free vote, Third Reading, June 2025."},
    {"id": 2058, "yes": 1, "q": "Should women no longer face prosecution for ending their own pregnancy?", "n": "Free vote on New Clause 1, Crime and Policing Bill, June 2025."},
    {"id": 1874, "yes": 1, "q": "Should tobacco sales be banned for anyone born from 2009 onwards?", "n": "Tobacco and Vapes Bill, Second Reading, November 2024."},
    {"id": 2271, "yes": -1, "q": "Should this law ban social media for under-16s?", "n": "A Lords amendment proposed the ban, and MPs voted to remove it in March 2026. Voting to remove it doesn't mean opposing all regulation."},
    {"id": 2074, "yes": 1, "q": "Should the 2025 welfare bill go ahead, reducing health-related benefits?", "n": "Universal Credit and PIP Bill, Second Reading, July 2025."},
    {"id": 2253, "yes": 1, "q": "Should the two-child limit on benefits be scrapped?", "n": "Second Reading, February 2026."},
    {"id": 1840, "yes": 1, "q": "Should the 2024 cut to winter fuel payments be reversed?", "n": "Motion to annul the regulations, September 2024."},
    {"id": 2203, "yes": 1, "q": "Should inheritance tax reliefs be limited for farms and family businesses?", "n": "Budget Resolution, December 2025."},
    {"id": 2113, "yes": 1, "q": "Should the UK give the Chagos Islands to Mauritius and lease back Diego Garcia?", "n": "Diego Garcia Bill, Second Reading, September 2025."},
    {"id": 2078, "yes": 1, "q": "Should Palestine Action be banned as a terrorist group?", "n": "July 2025. The same vote also covered two other groups."},
    {"id": 2225, "yes": 1, "q": "Should jury trials be kept for crimes likely to carry three years or less?", "n": "Opposition motion, January 2026."},
    {"id": 2406, "yes": 1, "q": "Should the government's 2026 Immigration and Asylum Bill go ahead?", "n": "Second Reading, July 2026. Most Conservative and Reform MPs didn't vote either way here."},
    {"id": 2390, "yes": 1, "q": "Should the government stop its clinical trial of puberty blockers for children?", "n": "Opposition motion, June 2026."},
]
long_votes = pd.read_parquet(ROOT / "data/clean/votes_long.parquet")
long_votes = long_votes[long_votes.division_id.isin([qq["id"] for qq in QUESTIONS])]
vote_table = collections.defaultdict(dict)
for t in long_votes.itertuples():
    vote_table[int(t.member_id)][int(t.division_id)] = "+" if t.vote == 1 else "-"

atlas = Image.new("RGB", (2048, 2048), (30, 50, 44))
groups = dict(tuple(passages.groupby("member_id")))
rows = []
def cut(t, n=55):
    w = t.split()
    return " ".join(w[:n]) + ("…" if len(w) > n else "")
for i, mid in enumerate(ids):
    m = members.loc[mid]
    r, col = i // COLS, i % COLS
    photo = ROOT / f"data/raw/photos/{mid}.jpg"
    if photo.exists():
        tile = Image.open(photo).convert("RGB")
        side = min(tile.size)
        tile = tile.crop(((tile.width - side) // 2, 0, (tile.width + side) // 2, side)).resize((TILE, TILE), Image.LANCZOS)
    else:
        tile = Image.new("RGB", (TILE, TILE), (70, 90, 84))
        initials = "".join(w[0] for w in re.sub(r"^(Mr|Mrs|Ms|Dr|Sir|Dame)\s+", "", m["name"]).split()[:2])
        ImageDraw.Draw(tile).text((TILE / 2, TILE / 2), initials, fill=(235, 235, 225), anchor="mm", font_size=26)
    atlas.paste(tile, (col * TILE, r * TILE))
    top = np.argsort(-share[i])[:6]
    g = groups[mid]
    best = g[(g.topic == real[top[0]]) & (g.words >= 30)]
    if best.empty:
        best = g
    quote = best.loc[best.sim.idxmax()]
    v = votes.loc[mid] if mid in votes.index else None
    rows.append({
        "i": int(mid), "n": m["name"], "p": m["party_abbr"], "pn": m["party"], "c": "#" + m["party_colour"], "k": m["constituency"],
        "x": [round(float(a), 2) for a in mp_xyz[i]], "y": [round(float(a), 2) for a in people_xyz[i]],
        "b": "".join(vote_table.get(int(mid), {}).get(qq["id"], "0") for qq in QUESTIONS),
        "t": [[int(k), round(float(share[i, k]), 3)] for k in top],
        "w": distinct.get(int(mid), []), "words": int(g.words.sum()), "min": round(float((g.role == "minister").mean()), 2),
        "nn": [int(a) for a in np.argsort(-sim[i])[:6]], "v": None if v is None else [int(v.votes_cast), int(v.rebellions)],
        "q": cut(quote.text), "qd": ((quote.minor or quote.major or "") + " · " + quote.date).strip(" ·"),
    })

party_words = collections.Counter()
party_topic = np.zeros((0,))
parties = np.array([r["p"] for r in rows])
party_list = [p for p, _ in collections.Counter(parties.tolist()).most_common()]
party_colour = {r["p"]: r["c"] for r in rows}
pt = np.array([c[parties == p].sum(0) for p in party_list])
party_overall = pt.sum(1) / pt.sum()

topics_out = []
owner = pd.Series(range(len(ids)), index=ids)
for k_new, k in enumerate(real):
    tp = pt[:, k_new] / pt[:, k_new].sum()
    order = np.argsort(-tp)[:5]
    leaders = np.argsort(-share[:, k_new] * (words[keep] > 8000))[:10]
    sub = passages[(passages.topic == k) & (passages.words >= 40) & passages.member_id.isin(ids)].sort_values("sim", ascending=False).head(200)
    quotes, seen = [], set()
    for t in sub.itertuples():
        p = members.party_abbr[t.member_id]
        if p in seen:
            continue
        seen.add(p)
        quotes.append({"m": int(owner[t.member_id]), "q": cut(t.text, 45), "d": ((t.minor or t.major or "") + " · " + t.date).strip(" ·")})
        if len(quotes) == 2:
            break
    topics_out.append({"n": TOPICS[k], "r": int(region[k]), "x": [round(float(a), 2) for a in topic_pos[k]], "terms": list(top_terms[k][:8]),
                       "size": int((topic_of == k).sum()), "parties": [[party_list[j], round(float(tp[j]), 3), round(float(party_overall[j]), 3)] for j in order],
                       "lead": [[int(a), round(float(share[a, k_new]), 3)] for a in leaders], "quotes": quotes})

# Stars keep the shape of each cloud from the UMAP layout, moved to their topic's place in its region
mask = np.isin(star_topic, real) & passages.member_id.iloc[star_idx].isin(ids).to_numpy()
star_xyz, star_topic, star_idx = star_xyz[mask], star_topic[mask], star_idx[mask]
offset = star_xyz - umap_centre[star_topic]
per = {k: np.percentile(np.linalg.norm(offset[star_topic == k], axis=1), 85) for k in real}
scale = np.array([0.62 / max(per[k], 1e-6) for k in star_topic])[:, None]
final = topic_pos[star_topic] + np.clip(offset * scale, -1.5, 1.5)
limit = np.abs(final).max() * 1.01
xyz16 = np.round(final / limit * 32767).astype("int16")
star_new_topic = np.array([new[k] for k in star_topic], "uint8")
star_owner = owner.reindex(passages.member_id.iloc[star_idx]).to_numpy().astype("int16")

buf = io.BytesIO()
atlas.save(buf, "JPEG", quality=78)
b64 = lambda a: base64.b64encode(a.tobytes()).decode()
# Each people-space group is named after the topics its MPs over-use most, so these names come from the data
clusters_out = []
for g_id in range(cluster.max() + 1):
    idx = cluster == g_id
    lead_topics = np.argsort(-V[idx].mean(0))[:3]
    parties_here = collections.Counter(parties[idx].tolist()).most_common(3)
    clusters_out.append({"n": " + ".join(topics_out[k]["n"] for k in lead_topics[:2]), "x": [round(float(a), 2) for a in np.median(people_xyz[idx], axis=0)],
                         "size": int(idx.sum()), "topics": [int(k) for k in lead_topics], "parties": parties_here})
party_avgs = []
for j, p in enumerate(party_list):
    if (parties == p).sum() >= 4 and p != "Ind":  # independents are not a party
        pos_p = (reducer.transform(mix_vector(pt[j])[None]) - people_centre) * people_scale
        party_avgs.append({"p": p, "c": party_colour[p], "n": int((parties == p).sum()), "x": [round(float(a), 2) for a in pos_p[0]]})
for qq in QUESTIONS:
    split = {}
    for p in ["Lab", "Con", "LD", "RUK", "Green", "SNP"]:
        ms = [r["b"][QUESTIONS.index(qq)] for r in rows if r["p"] == p]
        cast = [b for b in ms if b != "0"]
        if len(cast) >= 2:
            split[p] = round(sum(b == ("+" if qq["yes"] == 1 else "-") for b in cast) / len(cast), 2)
    qq["split"] = split
Vq = np.round(V / np.abs(V).max() * 127).astype("int8")

data = {"mps": rows, "clusters": clusters_out, "partyAvg": party_avgs, "questions": QUESTIONS, "topics": topics_out, "regions": [{"n": REGIONS[r], "c": REGION_COLOURS[r], "x": [round(float(a), 2) for a in lay["reg_xyz"][r]]} for r in sorted(REGIONS)],
        "cols": COLS, "tile": TILE, "size": 2048, "limit": float(limit), "stars": len(final), "chatter": round(chatter_share, 3),
        "nPassages": len(passages), "excluded": int((~keep).sum())}
page = (HERE / "topic_space_template.html").read_text()
page = page.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
page = page.replace("__ATLAS__", "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
page = page.replace("__MIX__", b64(Vq)).replace("__XYZ__", b64(xyz16)).replace("__TOPIC__", b64(star_new_topic)).replace("__OWNER__", b64(star_owner))
(HERE / "quick_map.html").write_text(page)
# GitHub Pages copy: a full document, since the local template relies on the host page to supply these
(ROOT / "docs/index.html").write_text('<!doctype html>\n<html lang="en">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + page)
print("page KB", len(page.encode()) // 1024, "| MPs", len(rows), "| excluded", int((~keep).sum()), "| topics", len(topics_out), "| stars", len(final), "| chatter share", round(chatter_share, 3))
for name in ["Gaza & Israel", "Asylum & small boats", "Chagos & Diego Garcia", "Water & sewage", "Farming & food", "Winter fuel", "EU, trade & Brexit", "Net zero & clean energy", "Assisted dying"]:
    t = next(x for x in topics_out if x["n"] == name)
    print(f"{name:26}", [(p, round(a * 100), round(b * 100)) for p, a, b in t["parties"][:4]], [rows[i]["n"] for i, _ in t["lead"][:3]])
for nm in ["Farage", "Kim Leadbeater", "Zarah Sultana", "Rachel Reeves", "Jim Shannon", "Tim Farron", "Kemi Badenoch"]:
    i = next((j for j, r in enumerate(rows) if nm in r["n"]), None)
    if i is not None:
        print(f"{nm:16}", [f"{rows[j]['n']} ({rows[j]['p']})" for j in rows[i]["nn"][:5]])
print("party average positions:", [(a["p"], a["x"]) for a in party_avgs])
for cl in clusters_out:
    print(f"{cl['size']:3d} {cl['n']:60} {cl['parties']}")
print("questions:", [(qq["id"], qq["split"]) for qq in QUESTIONS])
print("overall words by party:", [(p, round(s * 100)) for p, s in zip(party_list, party_overall)][:6])
