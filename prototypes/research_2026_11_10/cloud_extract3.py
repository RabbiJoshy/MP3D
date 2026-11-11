"""Cloud claim extraction (DeepInfra, OpenAI-compatible), parallel. python cloud_extract.py IN.jsonl OUT.jsonl [n] [chars] [model]"""
import sys, re, json, time, pathlib, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
key = next(m[1].strip().strip("\"'") for l in (ROOT / ".env").read_text().splitlines() if (m := re.match(r"\s*(?:export\s+)?DEEPINFRA_API_KEY\s*=\s*(.*)", l)))
src, dst = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 0
CH = int(sys.argv[4]) if len(sys.argv) > 4 else 1200; MODEL = sys.argv[5] if len(sys.argv) > 5 else "Qwen/Qwen3.5-4B"
t = pd.read_json(src, lines=True) if src.endswith("jsonl") else pd.read_parquet(src)
if n: t = t.head(n)
SYS = ("You analyse excerpts of UK House of Commons speeches. Identify the main concrete MEASURE being argued about: a specific policy, "
       "bill, clause or proposed change, phrased as an action someone wants to take, e.g. 'Require academies to follow the national curriculum', "
       "'Charge VAT on private school fees', 'Abolish no-fault evictions'. Phrase it as the change being proposed, never as keeping the status quo "
       "(write 'Abolish X', not 'Keep X'; write 'Cut Y', not 'Protect Y from cuts'). Then decide whether THE SPEAKER is FOR or AGAINST that measure. "
       "Careful: speakers often describe a measure in order to attack it; criticism and sarcasm mean AGAINST. "
       "Answer in exactly this format:\nMEASURE: ...\nSTANCE: FOR or AGAINST\nIf the speaker argues no position on a concrete measure, answer MEASURE: NONE.")
S = requests.Session(); S.headers.update({"Authorization": f"Bearer {key}"})
tok = {"in": 0, "out": 0}
def one(r):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 80, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": r["text"][:CH]}]}
    for a in range(6):
        try:
            x = S.post("https://api.deepinfra.com/v1/openai/chat/completions", json=body, timeout=60)
            if x.status_code == 200:
                d = x.json(); tok["in"] += d["usage"]["prompt_tokens"]; tok["out"] += d["usage"]["completion_tokens"]
                o = d["choices"][0]["message"]["content"] or ""
                g = lambda k: (m := re.search(k + r":\s*(.*)", o)) and m[1].strip()
                return {**{k: r[k] for k in ("passage_id", "member_id", "mp", "party", "role", "debate", "topic", "text") if k in r},
                        "measure": g("MEASURE"), "stance": (g("STANCE") or "").upper()[:7]}
            time.sleep(2 ** a)
        except requests.RequestException: time.sleep(2 ** a)
    return None
t0 = time.time()
with ThreadPoolExecutor(64) as ex: res = [x for x in ex.map(one, t.to_dict("records")) if x]
pd.DataFrame(res).to_json(dst, orient="records", lines=True)
dt = time.time() - t0
pi, po = {"Qwen/Qwen3.5-4B": (0.03, 0.15)}.get(MODEL, (0.03, 0.15))
print(f"{len(res)}/{len(t)} in {dt:.0f}s = {len(res)/dt:.1f}/s | tokens in {tok['in']} out {tok['out']} | ${(tok['in']*pi+tok['out']*po)/1e6:.4f}")
