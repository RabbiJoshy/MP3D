import random, json, re, pandas as pd
R="/Users/joshuathomasamar/PycharmProjects/MP3D/"; D=R+"data/quick/research_2026_11_10/"
rng=random.Random(20261110)
p=pd.read_parquet(R+"data/clean/passages.parquet"); m=pd.read_parquet(R+"data/clean/members.parquet").set_index("member_id")
p["home"]=pd.read_parquet(D+"homes.parquet").home.values
p["mp"]=p.member_id.map(m.name); p["party"]=p.member_id.map(m.party_abbr)
p["heading"]=p.major.fillna("")+" "+p.minor.fillna("")
ok=(p.heading.str.contains(r"Bill|Amendment|Opposition Day|New Clause|Clause|Motion|Lords"))&(p.words>=60)&~p.speech_type.str.contains("Chair",na=False)&p.member_id.notna()&(p.party!="Speaker")
rows=[];tab=[]
for t in sorted(set(p.home)-{"Education","No policy"}):
    e=p[ok&(p.home==t)].copy(); e["debate"]=e.major
    g=e.groupby(["date","major"]).agg(n=("passage_id","size"),mps=("member_id","nunique"),parties=("party","nunique")).reset_index()
    g=g[g.parties>=2].sort_values(["mps","n"],ascending=False)
    take=[];tot=0
    for _,d in g.iterrows():
        if tot>=250:break
        ids=e[(e.date==d.date)&(e.major==d.major)]
        if tot+len(ids)>250: ids=ids.sample(250-tot,random_state=rng.randrange(10**6))
        take.append(ids);tot+=len(ids)
    if not take: continue
    s=pd.concat(take); s["topic"]=t
    rows.append(s); tab.append(dict(topic=t,debates=len(take),passages=len(s),mps=s.member_id.nunique()))
s=pd.concat(rows)[["passage_id","member_id","mp","party","role","debate","topic","text"]]
s["date"]=0
s=s.drop(columns="date"); s.to_json(D+"fp/sample.jsonl",orient="records",lines=True)
pd.DataFrame(tab).to_csv(D+"fp/sample_table.csv",index=False); print(pd.DataFrame(tab).to_string(),len(s))
