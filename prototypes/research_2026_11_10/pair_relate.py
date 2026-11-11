"""Classify relation for frozen pairs. python pair_relate.py"""
import re, json, time, pathlib, requests, collections
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D"); R = ROOT / "prototypes/research_2026_11_10/results"
key = next(m[1].strip().strip("\"'") for l in (ROOT / ".env").read_text().splitlines() if (m := re.match(r"\s*(?:export\s+)?DEEPINFRA_API_KEY\s*=\s*(.*)", l)))
MODEL = "Qwen/Qwen3.5-4B"
P = """Two MPs spoke in the same House of Commons debate.

Speaker A ({name_a}): {text_a}

Speaker B ({name_b}): {text_b}

1. In one sentence each, state the policy position each speaker personally argues for (what they want to happen). If a speaker takes no clear position of their own, write NONE.
2. Decide the relation between the two positions, judged on the same policy decision, scope and conditions:
EQUIVALENT - on the same decision they want the same outcome.
INCOMPATIBLE - on the same decision they want outcomes that cannot both happen.
UNDETERMINED - anything else: different decisions, different priorities, proposals that could both happen, or either speaker has no clear position.
3. Give one verbatim quote of at most 25 words from each speech that supports your reading.

Reply with JSON only: {{"position_a": "...", "position_b": "...", "decision": "the shared policy decision, or NONE", "relation": "EQUIVALENT|INCOMPATIBLE|UNDETERMINED", "quote_a": "...", "quote_b": "..."}}"""
S = requests.Session(); S.headers.update({"Authorization": f"Bearer {key}"}); tok = {"in": 0, "out": 0}
def parse(o):
    m = re.search(r"\{.*\}", o, re.S)
    try: return json.loads(m[0]) if m else None
    except Exception: return None
def one(r):
    body = {"model": MODEL, "temperature": 0.1, "max_tokens": 500, "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": P.format(name_a=r["a"]["mp"], text_a=r["a"]["text"], name_b=r["b"]["mp"], text_b=r["b"]["text"])}]}
    for a in range(6):
        try:
            x = S.post("https://api.deepinfra.com/v1/openai/chat/completions", json=body, timeout=90)
            if x.status_code == 200:
                d = x.json(); tok["in"] += d["usage"]["prompt_tokens"]; tok["out"] += d["usage"]["completion_tokens"]
                o = d["choices"][0]["message"]["content"] or ""
                pj = parse(o)
                if pj is None and a < 2: continue
                return {"pair_id": r["pair_id"], "output": pj, "raw": o}
            time.sleep(2 ** a)
        except requests.RequestException: time.sleep(2 ** a)
    return {"pair_id": r["pair_id"], "output": None, "raw": None}
fr = json.load(open(R / "pairs_frozen.json"))
with ThreadPoolExecutor(10) as ex: res = list(ex.map(one, fr))
norm = lambda s: re.sub(r"\s+", " ", s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')).strip().lower().rstrip(".")
byid = {r["pair_id"]: r for r in fr}; ok = 0
for r in res:
    r["quote_check"] = {}
    for s in "ab":
        q = (r["output"] or {}).get("quote_" + s)
        v = bool(q) and norm(q) in norm(byid[r["pair_id"]][s]["text"]); r["quote_check"][s] = v; ok += v
json.dump(res, open(R / "pairs_labels.json", "w"), indent=1, ensure_ascii=False)
key_ = {k["pair_id"]: k["stratum"] for k in json.load(open(R / "pairs_key.json"))}
cnt = collections.Counter(); st = collections.defaultdict(collections.Counter)
for r in res:
    rel = (r["output"] or {}).get("relation", "PARSE_FAIL"); cnt[rel] += 1; st[key_[r["pair_id"]]][rel] += 1
print("relations", dict(cnt)); print({k: dict(v) for k, v in st.items()})
print("quotes verbatim", ok, "/60; parse fails", sum(r["output"] is None for r in res))
print(f"tokens in {tok['in']} out {tok['out']} cost ${(tok['in']*0.03+tok['out']*0.15)/1e6:.4f}")
