"""Step 4: merge same-fight groups per topic. python fp_merge.py [--dry] [maxpairs]"""
from pathlib import Path
import sys, re, json, numpy as np, pandas as pd, requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
R="/Users/joshuathomasamar/PycharmProjects/MP3D/"; D=R+"data/quick/research_2026_11_10/"
o=pd.read_json(D+"fp/measures.jsonl",lines=True)
G=o.groupby("gid").agg(topic=("topic","first"),label=("label","first"),mps=("member_id","nunique"),n=("passage_id","size")).reset_index()
from sentence_transformers import SentenceTransformer
m=SentenceTransformer("BAAI/bge-base-en-v1.5",local_files_only=True,device="mps")
pairs={}
for t,x in G.groupby("topic"):
    x=x.reset_index(drop=True); E=m.encode(x.label.tolist(),normalize_embeddings=True); S=E@E.T; np.fill_diagonal(S,-1)
    for i in np.where(x.mps.values>=2)[0]:
        for j in np.argsort(-S[i])[:5]:
            a,b=sorted((int(x.gid[i]),int(x.gid[j]))); pairs[(a,b)]=float(S[i,j])
lab=G.set_index("gid").label
Q1="Measure 1: {a}. Measure 2: {b}. Are these two measures about the same specific policy decision, so that MPs arguing about one are taking part in the same political fight as MPs arguing about the other? Answer yes even if one is phrased as the opposite of the other or uses different words for the same policy."
Q2="Measure 1: {a}. Measure 2: {b}. Would an MP who wants Measure 1 to happen also want Measure 2 to happen?"
L=list(pairs); print("candidate pairs",len(L))
if "--dry" in sys.argv: sys.exit()
key = next(m[1].strip().strip("\"'") for l in (Path(R) / ".env").read_text().splitlines() if (m := re.match(r"\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)", l)))
S = requests.Session(); S.headers.update({"Authorization": f"Bearer {key}"}); used = [0]
def ask(pr):
    a,b=pr; res=[]
    for x,y in ((a,b),(b,a)):
        q={"q1":{"type":"noul","instructions":Q1.format(a=lab[x],b=lab[y])},"q2":{"type":"noul","instructions":Q2.format(a=lab[x],b=lab[y])}}
        r=None
        for _ in range(6):
            try:
                z=S.post("https://api.typesafe.ai/v1/systemone",json={"model":"jev-1.14.0","state":"Two policy measures are compared.","questions":q},timeout=60)
                if z.status_code==200:
                    d=z.json(); used[0]+=d["usage"]["input_tokens"]; r=(d["answers"]["q1"]["noul"],d["answers"]["q2"]["noul"]); break
            except requests.RequestException: pass
        res.append(r)
    return res
MAXP=int(sys.argv[sys.argv.index("--max")+1]) if "--max" in sys.argv else len(L)
L=sorted(L,key=lambda k:-pairs[k])[:MAXP]; print("judging pairs",len(L))
with ThreadPoolExecutor(8) as ex: res=list(ex.map(ask,L))
rows=[]
for (a,b),r in zip(L,res):
    if None in r: continue
    rows.append(dict(a=a,b=b,label_a=lab[a],label_b=lab[b],cos=pairs[(a,b)],p1=(r[0][0]+r[1][0])/2,p2=(r[0][1]+r[1][1])/2,p1_ab=r[0][0],p1_ba=r[1][0],p2_ab=r[0][1],p2_ba=r[1][1]))
pd.DataFrame(rows).to_json(D+"fp/merge_pairs.jsonl",orient="records",lines=True)
print("tokens",used[0],"cost $",used[0]*0.042/1e6)
json.dump({"tokens":used[0]},open(R+"prototypes/research_2026_11_10/results/fp_merge_tokens.json","w"))
