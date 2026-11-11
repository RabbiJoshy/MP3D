"""Step 2b/3: combine claims (new + edu2), group per topic with measures.py's method (bge, cosine 0.25, average linkage)."""
import json, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
D="/Users/joshuathomasamar/PycharmProjects/MP3D/data/quick/research_2026_11_10/"
new=pd.read_json(D+"fp/claims_new.jsonl",lines=True); edu=pd.read_json(D+"edu2_claims.jsonl",lines=True)
c=pd.concat([new,edu],ignore_index=True); c.to_json(D+"fp/claims.jsonl",orient="records",lines=True)
tot=c.groupby("topic").size()
c=c[c.measure.notna()&~c.measure.str.upper().str.startswith("NONE")&c.stance.isin(["FOR","AGAINST"])].copy()
m=SentenceTransformer("BAAI/bge-base-en-v1.5",local_files_only=True,device="mps")
out=[];gid=0
for t,x in c.groupby("topic"):
    x=x.copy(); E=m.encode(x.measure.tolist(),normalize_embeddings=True,batch_size=64)
    x["g"]=AgglomerativeClustering(n_clusters=None,distance_threshold=0.25,metric="cosine",linkage="average").fit_predict(E)
    lab={}
    for g,idx in x.groupby("g").indices.items():
        cen=E[idx].mean(0); lab[g]=x.measure.iloc[idx[np.argmax(E[idx]@cen)]]
    x["label"]=x.g.map(lab); x["gid"]=x.g+gid; gid+=x.g.max()+1; out.append(x)
o=pd.concat(out); o.to_json(D+"fp/measures.jsonl",orient="records",lines=True)
print(len(c),"claims;",o.gid.nunique(),"groups"); print(pd.DataFrame({"passages":tot,"claims":o.groupby("topic").size(),"groups":o.groupby("topic").gid.nunique()}).to_string())
