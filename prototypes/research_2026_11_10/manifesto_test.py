"""ManifestoBERTa (context model) on the same 200 passages. Each sentence is classified with the passage as context;
sentence probabilities are averaged into a passage label. Output: manifesto_test.jsonl"""
import json, pathlib, re
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
OUT = pathlib.Path(__file__).with_name("manifesto_test.jsonl")
M = "manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2025-1-1"
DOMAINS = {"1": "External relations", "2": "Freedom & democracy", "3": "Political system", "4": "Economy",
           "5": "Welfare & quality of life", "6": "Fabric of society", "7": "Social groups"}

cur = pd.read_parquet(ROOT / "data/clean/passages.parquet", columns=["passage_id", "date", "major", "minor", "role", "text", "words"]).sample(100, random_state=11)
hist = pd.read_parquet(ROOT / "data/quick/passages_hist_2022_2024.parquet").sample(100, random_state=11)
cur["parliament"], hist["parliament"] = "2024-", "2022-24"
df = pd.concat([cur, hist], ignore_index=True)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("xlm-roberta-large")
model = AutoModelForSequenceClassification.from_pretrained(M, trust_remote_code=True).to(dev).eval()
labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

rows = []
for r in df.itertuples():
    sents = [s for s in re.split(r"(?<=[.!?])\s+", r.text.strip()) if len(s.split()) >= 3] or [r.text]
    with torch.no_grad():
        enc = tok(sents, [r.text] * len(sents), return_tensors="pt", max_length=400, truncation=True, padding=True).to(dev)
        p = torch.softmax(model(**enc).logits, -1).cpu().numpy()
    avg = p.mean(0)
    top = np.argsort(-avg)[:3]
    lab = labels[top[0]]
    rows.append({"id": r.passage_id, "parliament": r.parliament, "major": r.major, "minor": r.minor, "words": int(r.words),
                 "text": r.text, "label": lab, "p": round(float(avg[top[0]]), 3), "domain": DOMAINS.get(lab[0], "?"),
                 "top3": [[labels[i], round(float(avg[i]), 3)] for i in top]})
    print(len(rows), lab, round(float(avg[top[0]]), 2), "|", r.text[:90].replace("\n", " "), flush=True)
OUT.write_text("\n".join(json.dumps(x) for x in rows))
