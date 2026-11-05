"""Bundle the MP-free topic space into one self-contained page. Names come only from each group's top words."""
import base64
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
d = json.loads((ROOT / "data/quick/topic_only.json").read_text())
q = np.load(ROOT / "data/quick/topic_only.npz")


def short_name(words, n=3):
    out = []
    for w in words:
        if any(w in x or x in w for x in out):
            continue
        out.append(w)
        if len(out) == n:
            break
    return out


# A loose group sits far below the others in how tightly its passages cluster. The cut-off comes from the data.
coh = np.array([t["cohesion"] for t in d["topics"]])
cut = np.median(coh) - 2.5 * 1.4826 * np.median(np.abs(coh - np.median(coh)))
for t, c in zip(d["topics"], coh):
    t["catchall"] = bool(c < cut)
    t["name"] = short_name(t["words"], 2)
for r in d["regions"]:
    r["name"] = short_name(r["words"], 3)
print("catch-all topics:", [t["words"][:3] for t in d["topics"] if t["catchall"]])

xyz = q["star_xyz"]
limit = float(np.abs(xyz).max() * 1.01)
d["limit"] = limit
xyz16 = np.round(xyz / limit * 32767).astype("int16")
page = (HERE / "topic_only_template.html").read_text()
page = page.replace("__DATA__", json.dumps(d, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
page = page.replace("__XYZ__", base64.b64encode(xyz16.tobytes()).decode()).replace("__TOPIC__", base64.b64encode(q["star_topic"].astype("uint8").tobytes()).decode())
(HERE / "topic_only.html").write_text(page)
print("page KB", len(page.encode()) // 1024, "| topics", len(d["topics"]), "| regions", len(d["regions"]), "| stars", len(xyz), "|", d["stats"])
