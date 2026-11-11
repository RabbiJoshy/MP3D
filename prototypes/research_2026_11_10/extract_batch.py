"""Generic batched claim extraction for a whole topic; resumable. python extract_batch.py Housing [limit] [batch]"""
import sys, re, json, time, pathlib, pandas as pd
from mlx_lm import load, batch_generate
tag = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "0" else None; B = int(sys.argv[3]) if len(sys.argv) > 3 else 16
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
p = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "member_id", "major", "minor", "speech_type", "role", "text", "words"])
p["home"] = pd.read_parquet("homes.parquet").home.values
mem = pd.read_parquet(ROOT / "data/clean/members.parquet").set_index("member_id")
t = p[(p.home == tag.replace("_", " ")) & ~p.speech_type.str.contains("Chair", na=False) & (p.words >= 60)].join(mem[["name", "party_abbr"]], on="member_id")
out = pathlib.Path(f"{tag}_claims_all.jsonl")
done = set()
for f in (out, pathlib.Path(f"{tag}_claims.jsonl")):
    if f.exists(): done |= {json.loads(l)["passage_id"] for l in f.read_text().splitlines() if l.strip()}
t = t[~t.passage_id.isin(done)]
if limit: t = t.head(limit)
t = t.assign(L=t.text.str.len()).sort_values("L")
model, tok = load("mlx-community/Qwen3-4B-Instruct-2507-4bit")
SYS = ("You analyse excerpts of UK House of Commons speeches. Step 1: name the main specific policy the speaker discusses. "
       "Step 2: decide whether the SPEAKER THEMSELVES wants that policy, or opposes it. Careful: speakers often describe a policy, "
       "or another party's plan, in order to attack it. Sarcasm and criticism of opponents mean the speaker is AGAINST that thing. "
       "Step 3: write the speaker's own position as one plain sentence of at most 15 words. "
       "Answer in exactly this format:\nPOLICY: ...\nSPEAKER: FOR or AGAINST\nPOSITION: ...\nIf there is no policy position, answer POSITION: NONE.")
print(f"{len(t)} passages to do, batch {B}", flush=True)
t0 = time.time(); n = 0
with out.open("a") as f:
    for i in range(0, len(t), B):
        ch = t.iloc[i:i + B]
        prompts = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": x[:1800]}], add_generation_prompt=True) for x in ch.text]
        res = batch_generate(model, tok, prompts, max_tokens=80).texts
        for r, o in zip(ch.itertuples(), res):
            g = lambda k: (m := re.search(k + r":\s*(.*)", o)) and m[1].strip()
            f.write(json.dumps({"passage_id": r.passage_id, "member_id": int(r.member_id), "mp": r.name, "party": r.party_abbr, "role": r.role,
                                "debate": r.major or r.minor, "policy": g("POLICY"), "side": g("SPEAKER"), "position": g("POSITION"), "text": r.text}) + "\n")
        f.flush(); n += len(ch)
        if (i // B) % 10 == 0: print(f"{n}/{len(t)} {n/(time.time()-t0):.2f}/s | {g('POSITION')}", flush=True)
print("DONE", n, f"{time.time()-t0:.0f}s", f"{n/(time.time()-t0):.2f}/s")
