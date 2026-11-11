"""CAP major topics on the same 200 passages two ways:
(a) bgonzalezbustamante/ft-XLM-RoBERTa-policy-agenda-English (trained on UK Acts)
(b) embedding match: bge-base passage vs CAP topic descriptions (free, no training)
Output: cap_test.jsonl"""
import json, pathlib
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
HERE = pathlib.Path(__file__).parent
M = "bgonzalezbustamante/ft-XLM-RoBERTa-policy-agenda-English"
CAP = {"1": ("Macroeconomics", "general domestic macroeconomic policy: taxes, budget, public spending, borrowing, inflation, interest rates, cost of living"),
       "2": ("Civil rights", "civil rights, minority rights, discrimination, privacy, free speech, voting rights, abortion, assisted dying"),
       "3": ("Health", "health care, the NHS, hospitals, GPs, waiting lists, mental health, public health, smoking, drugs"),
       "4": ("Agriculture", "agriculture, farming, farmers, food production, fisheries, animal welfare"),
       "5": ("Labour", "labour, employment, jobs, wages, workers' rights, trade unions, pensions"),
       "6": ("Education", "education, schools, universities, colleges, teachers, skills, childcare"),
       "7": ("Environment", "environmental policy, pollution, sewage, climate change, nature, flooding, waste, fly-tipping"),
       "8": ("Energy", "energy policy, electricity, gas, oil, nuclear, renewables, energy bills, net zero"),
       "9": ("Immigration", "immigration, refugees, asylum seekers, small boats, borders, citizenship, visas"),
       "10": ("Transport", "transportation, roads, rail, buses, aviation, shipping, fuel duty for drivers"),
       "12": ("Law and crime", "law, crime, policing, courts, prisons, sentencing, victims, family law"),
       "13": ("Social welfare", "social welfare, benefits, universal credit, poverty, carers, disability support, social care"),
       "14": ("Housing", "housing, renting, landlords, social housing, homelessness, planning, local communities and urban affairs"),
       "15": ("Domestic commerce", "domestic commerce, business, banking, financial services, consumer protection, small businesses, regulation"),
       "16": ("Defence", "defence, the armed forces, military, veterans, weapons, national security"),
       "17": ("Technology", "space, science, technology, communications, the internet, artificial intelligence, broadband, media"),
       "18": ("Foreign trade", "foreign trade, trade deals, tariffs, exports, imports, Brexit trade arrangements"),
       "19": ("International affairs", "international affairs, foreign policy, foreign aid, diplomacy, wars and conflicts abroad, the EU"),
       "20": ("Government operations", "government operations, the civil service, Parliament, elections, local government, devolution, public procurement"),
       "21": ("Public lands", "public lands, water management, national parks, territorial issues, overseas territories"),
       "23": ("Culture", "cultural policy, arts, sport, museums, heritage, broadcasting, the BBC")}

base = pd.read_json(HERE / "manifesto_test.jsonl", lines=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForSequenceClassification.from_pretrained(M).to(dev).eval()
codes = [model.config.id2label[i] for i in range(model.config.num_labels)]
cls = []
for i in range(0, len(base), 16):
    enc = tok(list(base.text[i:i + 16]), return_tensors="pt", truncation=True, max_length=512, padding=True).to(dev)
    with torch.no_grad():
        cls.append(torch.softmax(model(**enc).logits, -1).cpu().numpy())
cls = np.vstack(cls)

bge = SentenceTransformer("BAAI/bge-base-en-v1.5", local_files_only=True, device=dev)
keys = list(CAP)
q = bge.encode(["Represent this sentence for searching relevant passages: " + CAP[k][1] for k in keys], normalize_embeddings=True)
d = bge.encode(list(base.text), normalize_embeddings=True)
sim = d @ q.T

rows = []
for j, r in base.iterrows():
    c = codes[int(cls[j].argmax())]
    e = keys[int(sim[j].argmax())]
    rows.append({"id": r.id, "text": r.text, "major": r.major, "minor": r.minor, "manifesto": r.label,
                 "cap_model": CAP[c][0], "cap_p": round(float(cls[j].max()), 3),
                 "cap_embed": CAP[e][0], "embed_sim": round(float(sim[j].max()), 3)})
out = pd.DataFrame(rows)
out.to_json(HERE / "cap_test.jsonl", orient="records", lines=True)
print(out.cap_model.value_counts(), "\n", out.cap_embed.value_counts())
print("model p>=.5:", (out.cap_p >= .5).mean().round(2), "| model and embed agree:", (out.cap_model == out.cap_embed).mean().round(2))
