"""fp2 step 3c/4: merge groups into fights (same rule as fp_remerge.py), keep fights with >= MINMP distinct extractor MPs.
Writes data/.../fp2/fights_pre.json (fights with claim lists + debates) and results/fp2_fights_pre_summary.json."""
import json, sys, collections, pandas as pd
R="/Users/joshuathomasamar/PycharmProjects/MP3D/"; D=R+"data/quick/research_2026_11_10/"; H=R+"prototypes/research_2026_11_10/"
MINMP=4
o=pd.read_json(D+"fp2/measures.jsonl",lines=True)
pa=pd.read_parquet(R+"data/clean/passages.parquet",columns=["passage_id","date","major","minor"]).drop_duplicates("passage_id").set_index("passage_id")
o["date"]=o.passage_id.map(lambda p:str(pa.date.get(p))); o["major"]=o.passage_id.map(lambda p:pa.major.get(p)); o["minor"]=o.passage_id.map(lambda p:pa.minor.get(p))
o["deb"]=list(zip(o.date,o.major))
debs=o.groupby("gid").deb.agg(set)
G=o.groupby("gid").agg(topic=("topic","first"),label=("label","first"),mps=("member_id","nunique")).reset_index().set_index("gid")
P=pd.read_json(D+"fp2/merge_pairs.jsonl",lines=True)
P["shared"]=[bool(debs.get(a,set())&debs.get(b,set())) for a,b in zip(P.a,P.b)]
P=P[(P.p1>=0.5)|((P.p1>=0.3)&P.shared)].sort_values("p1",ascending=False)
par={g:g for g in G.index}; sgn={g:1 for g in G.index}; members={g:[g] for g in G.index}
def find(g):
    s=1
    while par[g]!=g: s*=sgn[g]; g=par[g]
    return g,s
acc=0;rej=0
for r in P.itertuples():
    d=1 if r.p2>=0.5 else -1
    ra,sa=find(r.a); rb,sb=find(r.b)
    if ra==rb:
        if sa*sb!=d: rej+=1
        else: acc+=1
        continue
    if len(members[ra])+len(members[rb])>6: rej+=1; continue
    par[rb]=ra; sgn[rb]=sa*d*sb; members[ra]+=members[rb]; acc+=1
roots={}
for g in G.index: roots.setdefault(find(g)[0],[]).append(g)
fights=[];k=0
for root,gs in roots.items():
    anchor=max(gs,key=lambda g:G.mps[g]); sa=find(anchor)[1]
    dirs={g:int(find(g)[1]*sa) for g in gs}
    x=o[o.gid.isin(gs)].copy()
    if x.member_id.nunique()<MINMP: continue
    x["side"]=[s if dirs[g]==1 else ("AGAINST" if s=="FOR" else "FOR") for s,g in zip(x.stance,x.gid)]
    fights.append(dict(fight_id=k,topic=G.topic[root],label=G.label[anchor],n_mps=int(x.member_id.nunique()),
        groups=[dict(gid=int(g),label=G.label[g],direction=dirs[g]) for g in gs],
        claims=[dict(passage_id=r.passage_id,member_id=int(r.member_id),mp=r.mp,party=r.party,measure=r.measure,side=r.side,gid=int(r.gid)) for r in x.itertuples()],
        debates=[list(d) for d in sorted(set(x.deb),key=str)],
        minors=[[m,int(n)] for m,n in collections.Counter(zip(x.date,x.major,x.minor.fillna(""))).most_common(6)] if False else
               [[a,b,c,int(n)] for (a,b,c),n in collections.Counter(zip(x.date,x.major,x.minor.fillna(""))).most_common(6)]))
    k+=1
json.dump(fights,open(D+"fp2/fights_pre.json","w"))
st=pd.DataFrame([dict(topic=f["topic"],n=f["n_mps"]) for f in fights])
allf=len(roots)
summ=dict(groups=int(len(G)),pairs_judged=len(pd.read_json(D+"fp2/merge_pairs.jsonl",lines=True)),pairs_passing=len(P),accepted=acc,rejected_or_capped=rej,
   fights_total=allf,multi_group=sum(len(v)>1 for v in roots.values()),fights_ge4=len(fights),per_topic_ge4=st.groupby("topic").size().to_dict(),mps_dist=st.n.describe().to_dict())
json.dump(summ,open(H+"results/fp2_fights_pre_summary.json","w"),indent=1,default=str); print(json.dumps(summ,default=str))
