"""Per topic: group claims by POLICY (bge, cosine 0.3), keep issues with >=5 claims; 3 one-idea propositions per issue (cloud)."""
import re, json, pathlib, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
key = next(m[1].strip().strip("\"'") for l in (ROOT / ".env").read_text().splitlines() if (m := re.match(r"\s*(?:export\s+)?DEEPINFRA_API_KEY\s*=\s*(.*)", l)))
c = pd.read_json("all_claims.jsonl", lines=True)
c = c[c.position.notna() & ~c.position.fillna("").str.upper().str.contains("^NONE|POSITION: NONE") & c.policy.notna()].copy()
m = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True, device="mps")
c["emb"] = list(m.encode(c.policy.tolist(), normalize_embeddings=True))
parts = []
for topic, g in c.groupby("topic"):
    import numpy as np
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=0.3, metric="cosine", linkage="average").fit_predict(np.vstack(g.emb))
    g = g.assign(issue=[f"{topic}|{x}" for x in lab])
    sz = g.issue.value_counts(); parts.append(g[g.issue.isin(sz[sz >= 5].index)])
c = pd.concat(parts).drop(columns="emb")
print(len(c), "claims in", c.issue.nunique(), "issues")
SYS = ("Below are positions taken by different UK MPs on one policy issue. Write THREE different propositions, one per line, about the main "
       "policy choice these MPs argue over. Rules: each is ONE idea only, of the form '<policy> should <verb> ...', at most 10 words; "
       "no 'with', 'to', 'and', 'while' or 'in order to' add-ons; something UK MPs genuinely split on, not a goal everyone shares. "
       "No numbering, no extra text.")
S = requests.Session(); S.headers.update({"Authorization": f"Bearer {key}"}); tok = [0, 0]
def props(item):
    iss, g = item
    lst = "\n".join("- " + x for x in g.position.sample(min(30, len(g)), random_state=0))
    body = {"model": "Qwen/Qwen3.5-4B", "temperature": 0, "max_tokens": 90, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": lst}]}
    for _ in range(5):
        x = S.post("https://api.deepinfra.com/v1/openai/chat/completions", json=body, timeout=60)
        if x.status_code == 200:
            d = x.json(); tok[0] += d["usage"]["prompt_tokens"]; tok[1] += d["usage"]["completion_tokens"]
            return iss, [re.sub(r"^[\-\d\.\)\s]+", "", y).strip(' "') for y in (d["choices"][0]["message"]["content"] or "").split("\n") if y.strip()][:3]
    return iss, []
with ThreadPoolExecutor(32) as ex: cands = dict(ex.map(props, c.groupby("issue")))
c.to_json("all_issues.jsonl", orient="records", lines=True); json.dump(cands, open("all_cands.json", "w"), indent=1)
print(f"proposition tokens in {tok[0]} out {tok[1]}  ${(tok[0]*0.03+tok[1]*0.15)/1e6:.4f}")
for k in list(cands)[:8]: print(k, cands[k])
