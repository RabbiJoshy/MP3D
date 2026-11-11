"""Step 1: relabel every fight with the extraction model (one call per fight, fixed generic prompt)."""
import re, json, random, pathlib, requests, collections, pandas as pd
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D"); R = ROOT / "prototypes/research_2026_11_10/results"
key = next(m[1].strip().strip("\"'") for l in (ROOT / ".env").read_text().splitlines() if (m := re.match(r"\s*(?:export\s+)?DEEPINFRA_API_KEY\s*=\s*(.*)", l)))
MODEL = "Qwen/Qwen3.5-4B"
F = json.load(open(R / "fp_fights_final2.json"))["fights"]
p = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "major", "minor", "text"]).set_index("passage_id")
SYS = ("You analyse the main policy fight in a set of UK House of Commons speeches. You get the current label of a policy fight, labels of the individual "
       "measures MPs spoke about, the debate headings, and short excerpts from speakers on both sides. Write a clean label for the fight.\n"
       "Rules for `label`: the specific policy change at stake, named concretely (name the Bill, Act, scheme or institution), phrased as ONE proposal that someone can be for or against, "
       "at most 14 words, neutral wording, no 'or', no brackets, no vague words like 'the Bill' or 'the treaty' without a name.\n"
       "Answer with JSON only: {\"label\": str, \"description\": one plain sentence saying what the proposal is, \"old_label_clear\": true/false "
       "(was the old label already a clear, single, specific proposal?), \"same_direction\": true/false (does supporting the new label mean the same side as supporting the old label?)}")
def build(f):
    cl = collections.defaultdict(list)
    for k in ("mps_for", "mps_against", "mps_both_sides"):
        for m in f[k]:
            for c in m["claims"]: cl[c["side"] if c["side"] in ("FOR", "AGAINST") else "FOR"].append(c["passage_id"])
    ids = [x for s in cl.values() for x in s if x in p.index]
    heads = collections.Counter(p.major[ids].dropna()); mins = collections.Counter(p.minor[ids].dropna())
    rnd = random.Random(20261110); sel = []
    A, B = sorted(set(cl["FOR"]) & set(p.index)), sorted(set(cl["AGAINST"]) & set(p.index)); rnd.shuffle(A); rnd.shuffle(B)
    while len(sel) < 8 and (A or B):
        if A and len(sel) % 2 == 0 or not B: sel.append(("FOR", A.pop()))
        else: sel.append(("AGAINST", B.pop()))
    ml = list(dict.fromkeys(g["label"] for g in f["groups"]))[:8]
    u = (f"Current label: {f['label']}\nMeasure labels: " + " | ".join(ml) + "\nMain debates: " + "; ".join(h for h, _ in heads.most_common(3)) +
         "\nSub-headings: " + "; ".join(h for h, _ in mins.most_common(4)) + "\nExcerpts:\n" +
         "\n".join(f"- [{s}] " + " ".join(p.text[i].split())[:300] for s, i in sel))
    return u
S = requests.Session(); S.headers.update({"Authorization": f"Bearer {key}"}); tok = [0, 0]
def one(f):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 250, "chat_template_kwargs": {"enable_thinking": False}, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": build(f)}]}
    for _ in range(5):
        try:
            x = S.post("https://api.deepinfra.com/v1/openai/chat/completions", json=body, timeout=90)
            if x.status_code == 200:
                d = x.json(); tok[0] += d["usage"]["prompt_tokens"]; tok[1] += d["usage"]["completion_tokens"]
                return json.loads(d["choices"][0]["message"]["content"])
        except (requests.RequestException, ValueError): pass
    return None
with ThreadPoolExecutor(8) as ex: res = list(ex.map(one, F))
out = []
for f, r in zip(F, res):
    r = r or {}
    out.append({"fight_id": f["fight_id"], "topic": f["topic"], "old": f["label"], "new": (r.get("label") or f["label"]).strip(), "description": r.get("description", ""),
                "old_label_clear": r.get("old_label_clear") is not False and r.get("old_label_clear") != "false", "same_direction": r.get("same_direction") is not False and r.get("same_direction") != "false",
                "failed": not r})
cost = (tok[0] * 0.03 + tok[1] * 0.15) / 1e6
json.dump({"model": MODEL, "tokens": tok, "usd": round(cost, 5), "labels": out}, open(R / "fp_clean_labels.json", "w"), indent=1)
print("tokens", tok, "usd", round(cost, 4), "failed", sum(o["failed"] for o in out), "unclear", sum(not o["old_label_clear"] for o in out), "flipped", sum(not o["same_direction"] for o in out))
