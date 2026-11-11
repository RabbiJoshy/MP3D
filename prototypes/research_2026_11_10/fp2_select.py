"""fp2 step 1: same eligibility/ranking as fp_select.py, debates in rank order up to CAP passages per topic (incl. already-extracted). New = not in fp/sample.jsonl."""
import random, json, sys, pandas as pd
R="/Users/joshuathomasamar/PycharmProjects/MP3D/"; D=R+"data/quick/research_2026_11_10/"
CAP=int(sys.argv[1]) if len(sys.argv)>1 else 1000
rng=random.Random(20261110)
p=pd.read_parquet(R+"data/clean/passages.parquet"); m=pd.read_parquet(R+"data/clean/members.parquet").drop_duplicates("member_id").set_index("member_id")
p["home"]=pd.read_parquet(D+"homes.parquet").home.values
p["mp"]=p.member_id.map(m.name); p["party"]=p.member_id.map(m.party_abbr)
p["heading"]=p.major.fillna("")+" "+p.minor.fillna("")
ok=(p.heading.str.contains(r"Bill|Amendment|Opposition Day|New Clause|Clause|Motion|Lords"))&(p.words>=60)&~p.speech_type.str.contains("Chair",na=False)&p.member_id.notna()&(p.party!="Speaker")
old=set(pd.read_json(D+"fp/sample.jsonl",lines=True).passage_id)
rows=[];tab=[]
for t in sorted(set(p.home)-{"Education","No policy"}):
    e=p[ok&(p.home==t)].copy(); e["debate"]=e.major
    g=e.groupby(["date","major"]).agg(n=("passage_id","size"),mps=("member_id","nunique"),parties=("party","nunique")).reset_index()
    g=g[g.parties>=2].sort_values(["mps","n"],ascending=False)
    take=[];tot=0
    for _,d in g.iterrows():
        if tot>=CAP:break
        ids=e[(e.date==d.date)&(e.major==d.major)]
        if tot+len(ids)>CAP: ids=ids.sample(CAP-tot,random_state=rng.randrange(10**6))
        take.append(ids);tot+=len(ids)
    s=pd.concat(take); s["topic"]=t; s["isnew"]=~s.passage_id.isin(old)
    rows.append(s); tab.append(dict(topic=t,debates=len(take),selected=len(s),already=int((~s.isnew).sum()),new=int(s.isnew.sum()),mps=s.member_id.nunique()))
s=pd.concat(rows)[["passage_id","member_id","mp","party","role","debate","topic","text","isnew"]]
n=s[s.isnew].drop(columns="isnew"); n.to_json(D+"fp2/sample_new.jsonl",orient="records",lines=True)
s[["passage_id","topic","isnew"]].to_json(D+"fp2/selected_all.jsonl",orient="records",lines=True)
t=pd.DataFrame(tab); t.to_csv(D+"fp2/sample_table.csv",index=False); print(t.to_string()); print("new",len(n),"est chars",n.text.str.len().clip(upper=1200).sum())
