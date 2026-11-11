import json, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
M = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
tok = AutoTokenizer.from_pretrained(M); mod = AutoModelForSequenceClassification.from_pretrained(M).to("mps").eval()
id2 = mod.config.id2label; print(id2)
gold = {int(k): v for k, v in json.load(open("gold_housing.json")).items()}
c = pd.read_json("Housing_labelled.jsonl", lines=True).loc[list(gold)]
c["gold"] = c.index.map(gold); c["llm"] = c.label.str[0]
def nli(prem, hyp):
    with torch.no_grad():
        e = tok(prem, hyp, return_tensors="pt", truncation=True, padding=True).to("mps")
        p = torch.softmax(mod(**e).logits, -1).cpu().numpy()
    return p
def lab(p):
    names = [id2[i].lower() for i in range(len(id2))]
    out = []
    for row in p:
        d = dict(zip(names, row))
        out.append("A" if d["entailment"] > max(d["contradiction"], d["neutral"]) else "D" if d["contradiction"] > max(d["entailment"], d["neutral"]) else "U")
    return out
prem = ("The speaker " + c.position.str[0].str.lower() + c.position.str[1:]).tolist()
hyp = ("The speaker believes that " + c.proposition.str[0].str.lower() + c.proposition.str[1:]).tolist()
c["nli"] = lab(nli(prem, hyp))
def score(col):
    side = c[c.gold != "U"]
    return f"all {(c[col]==c.gold).mean():.0%} | sides right (gold A/D) {(side[col]==side.gold).mean():.0%} | flipped {((side[col]!=side.gold)&(side[col]!='U')).sum()} | missed {(side[col]=='U').sum()} of {len(side)}"
print("small model (v2):", score("llm")); print("NLI:            ", score("nli"))
c[["mp", "party", "position", "proposition", "gold", "llm", "nli"]].to_json("nli_eval.jsonl", orient="records", lines=True)
print(c[c.nli != c.gold][["mp", "position", "gold", "nli"]].to_string(max_colwidth=70))
P = nli(prem, hyp)
c["nli2"] = ["A" if r[0] > r[2] else "D" for r in P]
side = c[c.gold != "U"]
print("NLI forced A/D: sides right", f"{(side.nli2==side.gold).mean():.0%}", "flipped", (side.nli2 != side.gold).sum(), "of", len(side))
both = side[side.nli2 == side.llm]
print("where small model and forced NLI agree:", len(both), "claims,", f"{(both.llm==both.gold).mean():.0%} right")
