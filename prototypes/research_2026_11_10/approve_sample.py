import json, pathlib, re, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
src = open("cap_test.py").read()
CAP = eval(re.search(r"CAP = (\{.*?\})\n\nbase", src, re.S).group(1))
CAP["5"] = ("Employment", "jobs, unemployment, wages, the minimum wage, workers' rights, trade unions, strikes, workplace pensions")
CAP["18"] = ("Foreign trade", "international trade deals, tariffs, exports and imports, trade with other countries")
CAP["0"] = ("No policy", "parliamentary procedure, thanking or congratulating colleagues, tributes and condolences, points of order, chairing the debate, attacks on another party's record")
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
cur = pd.read_parquet(ROOT/"data/clean/passages.parquet", columns=["passage_id","text","words"]).query("words>=40").sample(1000, random_state=5)
hist = pd.read_parquet(ROOT/"data/quick/passages_hist_2022_2024.parquet").query("words>=40").sample(1000, random_state=5)
df = pd.concat([cur.assign(parl="2024-"), hist.assign(parl="2022-24")], ignore_index=True)
bge = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True, device="mps" if torch.backends.mps.is_available() else "cpu")
keys = list(CAP)
q = bge.encode(["Represent this sentence for searching relevant passages: "+CAP[k][1] for k in keys], normalize_embeddings=True)
d = bge.encode(list(df.text), normalize_embeddings=True, batch_size=64)
sim = d @ q.T
df["topic"] = [CAP[keys[i]][0] for i in sim.argmax(1)]
df["sim"] = sim.max(1)
srt = np.sort(sim, 1); df["margin"] = srt[:, -1] - srt[:, -2]
out = []
for k in keys:
    name, desc = CAP[k]
    g = df[df.topic == name]
    # typical examples: median-confidence, clear margin; not the top (too easy)
    ex = g[g.margin > g.margin.median()] if len(g) > 4 else g
    ex = ex.sample(min(2, len(ex)), random_state=1)
    out.append({"code": k, "name": name, "desc": desc, "n": int(len(g)),
                "ex": [re.sub(r"\s+", " ", t)[:260] for t in ex.text]})
json.dump(out, open("approve.json", "w"), indent=1)
for o in sorted(out, key=lambda o: -o["n"]): print(o["n"], o["name"])
