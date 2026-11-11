import json, random, pandas as pd
ROOT="/Users/joshuathomasamar/PycharmProjects/MP3D"; D=ROOT+"/data/quick/research_2026_11_10"
F=json.load(open(D+"/fp/dense_input_fights.json"))["fights"]
p=pd.read_parquet(ROOT+"/data/clean/passages.parquet"); p["home"]=pd.read_parquet(D+"/homes.parquet")["home"].values
p["date"]=p["date"].astype(str)
rows=[]; rep=[]
CAP=int(__import__("sys").argv[1]) if len(__import__("sys").argv)>1 else 300
pid2=p.set_index("passage_id")
for f in F:
    if f["status"]!="fight": continue
    acc={c["passage_id"] for l in("mps_for","mps_against") for m in f[l] for c in m["claims"]}
    ad=pid2.loc[[x for x in acc if x in pid2.index]]
    deb=set(zip(ad.date,ad.major))
    q=p[(p.home==f["topic"])&(p.words>=60)&p.member_id.notna()&pd.Series(list(zip(p.date,p.major)),index=p.index).isin(deb)]
    ids=sorted(set(q.passage_id)|(acc&set(pid2.index)))
    keep=sorted(acc&set(ids)); rest=[i for i in ids if i not in acc]
    r=random.Random(20261110); r.shuffle(rest)
    sel=keep+rest[:max(0,CAP-len(keep))]
    rep.append((f["fight_id"],f["label"][:40],len(ids),len(sel)))
    for i in sel: rows.append({"passage_id":i,"fight":f["fight_id"],"label":f["label"]})
c=pd.DataFrame(rows); c.to_json(D+"/fp/dense_cands.jsonl",orient="records",lines=True)
for x in rep: print(x)
t=pd.merge(c,pid2[["text"]],left_on="passage_id",right_index=True)
est=(t.text.str.len().clip(upper=3000).sum()/4*2+len(t)*2*120)
print("fights",len(rep),"cands",len(c),"est tokens",int(est),"est cents",est*0.042/1e6*100)
