import json, random, sys, pandas as pd
ROOT="/Users/joshuathomasamar/PycharmProjects/MP3D"; D=ROOT+"/data/quick/research_2026_11_10"; R=ROOT+"/prototypes/research_2026_11_10/results"
F=[f for f in json.load(open(R+"/fp_fights_final.json"))["fights"] if not f.get("dense")]
CAP=int(sys.argv[1]) if len(sys.argv)>1 else 150
p=pd.read_parquet(ROOT+"/data/clean/passages.parquet"); p["home"]=pd.read_parquet(D+"/homes.parquet")["home"].values
p["date"]=p["date"].astype(str); pid2=p.set_index("passage_id")
deb=pd.Series(list(zip(p.date,p.major)),index=p.index)
rows=[]; rep=[]
for f in F:
    acc={c["passage_id"] for l in("mps_for","mps_against","mps_both_sides") for m in f[l] for c in m["claims"]}
    ad=pid2.loc[[x for x in acc if x in pid2.index]]
    ds=set(zip(ad.date,ad.major))
    q=p[(p.home==f["topic"])&(p.words>=60)&p.member_id.notna()&deb.isin(ds)]
    ids=sorted(set(q.passage_id)|(acc&set(pid2.index)))
    keep=sorted(acc&set(ids)); rest=[i for i in ids if i not in acc]
    r=random.Random(20261110); r.shuffle(rest)
    sel=keep+rest[:max(0,CAP-len(keep))]
    rep.append((f["fight_id"],len(ids),len(sel)))
    for i in sel: rows.append({"passage_id":i,"fight":f["fight_id"],"label":f["label"]})
c=pd.DataFrame(rows); c.to_json(D+"/fp/dense2_cands.jsonl",orient="records",lines=True)
t=pd.merge(c,pid2[["text"]],left_on="passage_id",right_index=True)
est=(t.text.str.len().clip(upper=3000).sum()/4*2+len(t)*240)
print("fights",len(rep),"cands",len(c),"uncapped",sum(x[1] for x in rep),"est cents (no reuse)",est*0.042/1e6*100)
