import json, random, pandas as pd
R="/Users/joshuathomasamar/PycharmProjects/MP3D/"
D=R+"data/quick/research_2026_11_10/"; H=R+"prototypes/research_2026_11_10/results/"
j=pd.read_json(D+"edu_jev2_sides.jsonl",lines=True)
done={c["passage_id"] for g in json.load(open(H+"edu_handcheck.json"))["groups"] for c in g["claims"]}
j=j[~j.passage_id.isin(done)].reset_index(drop=True)
CUR="Require academies to follow the national curriculum"
rng=random.Random(20261110)
A=j[j.new_verdict=="NEITHER"]; A=A.loc[rng.sample(sorted(A.index),30)]; A=A.assign(stratum="A")
B=j[(j.label==CUR)&(j.new_verdict==j.stance)]; B=B.loc[rng.sample(sorted(B.index),min(16,len(B)))]; B=B.assign(stratum="B")
Hn=j[(j.label==CUR)&(j.mp=="Damian Hinds")].assign(stratum="H")
allr=pd.concat([A,B,Hn]); allr=allr.groupby(level=0).first() if False else allr
# dedupe on (passage,label), keep first stratum
allr=allr.drop_duplicates(["passage_id","label"]).reset_index(drop=True)
p=pd.read_parquet(R+"data/clean/passages.parquet").reset_index(drop=True)
idx={pid:i for i,pid in enumerate(p.passage_id)}
def nb(pid,d):
    i=idx.get(pid)
    if i is None: return None
    k=i+d
    if 0<=k<len(p) and p.major[k]==p.major[i] and p.minor[k]==p.minor[i] and p.date[k]==p.date[i]: return p.text[k]
rows=allr.to_dict("records"); rng.shuffle(rows)
blind=[];key={}
for n,r in enumerate(rows,1):
    blind.append({"row_id":n,"passage_id":r["passage_id"],"mp":r["mp"],"party":r["party"],"group_label":r["label"],"passage_text":r["text"],"prev_passage":nb(r["passage_id"],-1),"next_passage":nb(r["passage_id"],1)})
    key[n]={"stratum":r["stratum"],"measure":r["measure"],"extractor_side":r["stance"],"judge_verdict":r["new_verdict"]}
json.dump(blind,open(H+"edu_handcheck2_blind.json","w"),indent=1)
json.dump(key,open(H+"edu_handcheck2_key.json","w"),indent=1)
print(len(blind),"rows; strata sizes",allr.stratum.value_counts().to_dict())
