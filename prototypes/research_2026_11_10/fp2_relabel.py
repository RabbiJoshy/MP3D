"""fp2 step 5: relabel every fight with a stronger model (one call per fight, fixed generic prompt)."""
import re, json, random, pathlib, requests, collections, sys, pandas as pd
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D"); D = ROOT / "data/quick/research_2026_11_10"; R = ROOT / "prototypes/research_2026_11_10/results"
key = next(m[1].strip().strip("\"'") for l in (ROOT / ".env").read_text().splitlines() if (m := re.match(r"\s*(?:export\s+)?DEEPINFRA_API_KEY\s*=\s*(.*)", l)))
MODEL = "deepseek-ai/DeepSeek-V3.2"; PRICE = (0.26, 0.38)  # first tried Qwen3-235B-A22B-Instruct-2507: too slow under load (~2 calls/min), abandoned after ~$0.03
FALLBACK = "Qwen/Qwen3.5-4B"; FPRICE = (0.03, 0.15)
F = json.load(open(D / "fp2/fights_pre.json"))
p = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "text"]).set_index("passage_id")
SYS = ("You analyse the main policy fight in a set of UK House of Commons speeches. You get the current (machine-made, possibly wrong) label of the fight, labels of the individual "
       "measures MPs spoke about, the debate headings with dates, and short excerpts from speakers on both sides (the [FOR]/[AGAINST] tags are rough machine guesses).\n"
       "Write a clean label for the fight. Rules for `label`: the specific policy change at stake, named concretely and ACCURATELY: name the Bill, Act, scheme or institution exactly as given in the debate headings; "
       "do not invent what a Bill does beyond what the excerpts and headings say. Phrase it as ONE proposal an MP can be for or against, at most 14 words, neutral wording, no 'or', no brackets. "
       "If the fight is about a Bill, orient the label so that supporting it means supporting the change the government or the Bill's sponsor is proposing.\n"
       "Answer with JSON only: {\"label\": str, \"description\": one plain sentence saying what the proposal is, \"for_means\": what an MP who is FOR the label wants (at most 12 words), "
       "\"against_means\": what an MP who is AGAINST the label wants (at most 12 words)}")
def build(f):
    cl = collections.defaultdict(list)
    for c in f["claims"]: cl[c["side"]].append(c["passage_id"])
    rnd = random.Random(20261110); A = sorted(set(cl["FOR"])); B = sorted(set(cl["AGAINST"])); rnd.shuffle(A); rnd.shuffle(B); sel = []
    while len(sel) < 10 and (A or B):
        if (A and len(sel) % 2 == 0) or not B: sel.append(("FOR", A.pop()))
        else: sel.append(("AGAINST", B.pop()))
    ml = list(dict.fromkeys(g["label"] for g in f["groups"]))[:8]
    heads = collections.Counter(); 
    for a, b, c, n in f["minors"]: heads[(a, b)] += n
    hd = "; ".join(f"{b} ({a})" for (a, b), _ in heads.most_common(3))
    mins = "; ".join(dict.fromkeys(c for a, b, c, n in f["minors"] if c))[:300]
    return (f"Current label: {f['label']}\nMeasure labels: " + " | ".join(ml) + "\nMain debates: " + hd + "\nSub-headings: " + mins + "\nExcerpts:\n" +
            "\n".join(f"- [{s}] " + " ".join(p.text[i].split())[:400] for s, i in sel))
S = requests.Session(); S.headers.update({"Authorization": f"Bearer {key}"}); tok = {"big": [0, 0], "small": [0, 0]}; fb = [0]
def call(model, body, tk):
    for a in range(5):
        try:
            x = S.post("https://api.deepinfra.com/v1/openai/chat/completions", json=body, timeout=120)
            if x.status_code == 200:
                d = x.json(); tok[tk][0] += d["usage"]["prompt_tokens"]; tok[tk][1] += d["usage"]["completion_tokens"]
                return json.loads(d["choices"][0]["message"]["content"])
            if x.status_code in (401, 402, 403): raise SystemExit(f"auth/quota error {x.status_code}")
        except (requests.RequestException, ValueError): pass
    return None
def one(f):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": build(f)}]
    r = call(MODEL, {"model": MODEL, "temperature": 0, "max_tokens": 300, "response_format": {"type": "json_object"}, "messages": msgs}, "big")
    if not r or not r.get("label"):
        fb[0] += 1; r = call(FALLBACK, {"model": FALLBACK, "temperature": 0, "max_tokens": 300, "chat_template_kwargs": {"enable_thinking": False}, "response_format": {"type": "json_object"}, "messages": msgs}, "small")
    return r
with ThreadPoolExecutor(16) as ex: res = list(ex.map(one, F))
out = []
for f, r in zip(F, res):
    r = r or {}
    out.append({"fight_id": f["fight_id"], "topic": f["topic"], "old": f["label"], "new": (r.get("label") or f["label"]).strip(), "description": r.get("description", ""),
                "for_means": r.get("for_means", ""), "against_means": r.get("against_means", ""), "failed": not r})
cost = (tok["big"][0] * PRICE[0] + tok["big"][1] * PRICE[1] + tok["small"][0] * FPRICE[0] + tok["small"][1] * FPRICE[1]) / 1e6
json.dump({"model": MODEL, "price_per_M_in_out": PRICE, "tokens": tok, "fallback_calls": fb[0], "usd": round(cost, 5), "labels": out}, open(R / "fp2_labels.json", "w"), indent=1)
print("fights", len(F), "tokens", tok, "usd", round(cost, 4), "fallback", fb[0], "failed", sum(o["failed"] for o in out))
