import re, json, pathlib, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
key = next(m[1].strip().strip("\"'") for l in (ROOT / ".env").read_text().splitlines() if (m := re.match(r"\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)", l)))
S = requests.Session(); S.headers.update({"Authorization": f"Bearer {key}"})
gold = {int(k): v for k, v in json.load(open("gold_housing.json")).items()}
c = pd.read_json("Housing_labelled.jsonl", lines=True).loc[list(gold)]; c["gold"] = c.index.map(gold)
used = [0]
def ask(r, state):
    qs = {"agree": {"type": "noul", "instructions": f"Would the speaker of this passage agree with this statement: \"{r.proposition}\"? Judge what the speaker argues, not what they describe; criticising a policy means disagreeing with it."},
          "relevant": {"type": "noul", "instructions": f"Does this passage take any position on this statement: \"{r.proposition}\"?"}}
    for _ in range(5):
        x = S.post("https://api.typesafe.ai/v1/systemone", json={"model": "jev-1.14.0", "state": state, "questions": qs}, timeout=60)
        if x.status_code == 200:
            d = x.json(); used[0] += d["usage"]["input_tokens"]; return d["answers"]["agree"]["noul"], d["answers"]["relevant"]["noul"]
    raise RuntimeError(x.text[:200])
for name, fn in [("speech", lambda r: r.text[:3000]), ("summary", lambda r: "The speaker: " + r.position), ("both", lambda r: f"Speaker's view (summary): {r.position}\n\nSpeech: {r.text[:2500]}")]:
    with ThreadPoolExecutor(6) as ex: res = list(ex.map(lambda r: ask(r, fn(r)), c.itertuples()))
    c[name] = ["A" if a >= .5 else "D" for a, _ in res]
    side = c[c.gold != "U"]
    print(f"Jev on {name:8}: sides right {(side[name]==side.gold).mean():.0%}  flipped {(side[name]!=side.gold).sum()} of {len(side)}")
c["llm"] = c.label.str[0]; side = c[c.gold != "U"]
print(f"small model    : sides right {(side.llm==side.gold).mean():.0%}")
print(f"tokens used {used[0]}  ~${used[0]*0.042/1e6:.4f}")
c[["mp","position","proposition","gold","llm","speech","summary","both"]].to_json("jev_eval.jsonl", orient="records", lines=True)
