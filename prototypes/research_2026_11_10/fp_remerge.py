"""Re-merge groups from saved pair scores (no API). Rule: merge if mean p1>=0.5, OR p1>=0.3 and groups share a debate (date+major).
Then recount as fp_recount.py. Writes data/.../fp/fights_v2.jsonl and results/fp_fights_v2.json."""
import json, pandas as pd
R="/Users/joshuathomasamar/PycharmProjects/MP3D/"; D=R+"data/quick/research_2026_11_10/"; H=R+"prototypes/research_2026_11_10/"
o=pd.read_json(D+"fp/measures.jsonl",lines=True)
pa=pd.read_parquet(R+"data/clean/passages.parquet",columns=["passage_id","date","major"]).drop_duplicates("passage_id").set_index("passage_id")
o["deb"]=[(str(pa.date.get(p)),pa.major.get(p)) if p in pa.index else None for p in o.passage_id]
print("passages without debate info",int(o.deb.isna().sum()))
debs=o.dropna(subset=["deb"]).groupby("gid").deb.agg(set)
G=o.groupby("gid").agg(topic=("topic","first"),label=("label","first"),mps=("member_id","nunique")).reset_index().set_index("gid")
P=pd.read_json(D+"fp/merge_pairs.jsonl",lines=True)
share=lambda a,b: bool(debs.get(a,set()) & debs.get(b,set()))
P["shared"]=[share(a,b) for a,b in zip(P.a,P.b)]
P=P[(P.p1>=0.5)|((P.p1>=0.3)&P.shared)].sort_values("p1",ascending=False)
par={g:g for g in G.index}; sgn={g:1 for g in G.index}; members={g:[g] for g in G.index}
def find(g):
    s=1
    while par[g]!=g: s*=sgn[g]; g=par[g]
    return g,s
acc=[];rej=0
for r in P.itertuples():
    d=1 if r.p2>=0.5 else -1
    ra,sa=find(r.a); rb,sb=find(r.b)
    if ra==rb:
        if sa*sb!=d: rej+=1
        else: acc.append(r)
        continue
    if len(members[ra])+len(members[rb])>6: rej+=1; continue
    par[rb]=ra; sgn[rb]=sa*d*sb
    members[ra]+=members[rb]; acc.append(r)
fights=[]; roots={}
for g in G.index: roots.setdefault(find(g)[0],[]).append(g)
for k,(root,gs) in enumerate(roots.items()):
    anchor=max(gs,key=lambda g:G.mps[g]); sa=find(anchor)[1]
    fights.append(dict(fight_id=k,topic=G.topic[root],label=G.label[anchor],groups=[dict(gid=int(g),label=G.label[g],direction=int(find(g)[1]*sa)) for g in gs]))
pd.DataFrame(fights).to_json(D+"fp/fights_v2.jsonl",orient="records",lines=True)
print("candidate pairs passing",len(P),"accepted",len(acc),"rejected/capped",rej,"multi-group fights",sum(len(f['groups'])>1 for f in fights))
# ---- recount (as fp_recount.py)
F=pd.DataFrame(fights).set_index("fight_id"); j=pd.read_json(D+"fp/jev_sides.jsonl",lines=True)
gd={g["gid"]:(f,g["direction"]) for f,gs in F.groups.items() for g in gs}
o["fight"]=o.gid.map(lambda g:gd[g][0]); o["dir"]=o.gid.map(lambda g:gd[g][1])
vj=j.set_index("passage_id").new_verdict; nm=o.groupby("fight").member_id.nunique()
big=o[o.fight.map(nm)>=3].copy(); big["verdict"]=big.passage_id.map(vj).fillna("NOT_JUDGED")
M=pd.read_parquet(R+"data/clean/members.parquet").set_index("member_id")
out=[]; npairs=pd.read_json(D+"fp/merge_pairs.jsonl",lines=True)
for f,x in big.groupby("fight"):
    acc_=x[x.verdict==x.stance]; drop=x[x.verdict!=x.stance]
    acc_=acc_.assign(side=[s if d==1 else ("AGAINST" if s=="FOR" else "FOR") for s,d in zip(acc_.stance,acc_.dir)])
    sides=acc_.groupby("member_id").side.agg(set); both=[int(m) for m,s in sides.items() if len(s)>1]
    def mp(m,side=None):
        a=acc_[acc_.member_id==m]; a=a[a.side==side] if side else a
        return dict(member_id=int(m),name=M.name.get(m),party=M.party_abbr.get(m),claims=[dict(passage_id=r.passage_id,measure=r.measure,side=r.side) for r in a.itertuples()])
    fo=[mp(m,"FOR") for m,s in sides.items() if s=={"FOR"}]; ag=[mp(m,"AGAINST") for m,s in sides.items() if s=={"AGAINST"}]
    if len(fo)+len(ag)<3: continue
    st="fight" if len(fo)>=2 and len(ag)>=2 else "one_short" if fo and ag else "one_sided"
    reason=lambda r:"NEITHER" if r.verdict=="NEITHER" else "NOT_JUDGED(budget)" if r.verdict=="NOT_JUDGED" else "ERR" if r.verdict=="ERR" else "opposite"
    out.append(dict(fight_id=int(f),topic=F.topic[f],label=F.label[f],status=st,groups=F.groups[f],mps_for=fo,mps_against=ag,
      mps_both_sides=[mp(m) for m in both],n_dropped=len(drop),dropped=[dict(passage_id=r.passage_id,reason=reason(r)) for r in drop.itertuples()]))
tab={t:dict(passages=0,claims=0,groups=0,fights_by_status={}) for t in o.topic.unique()}
cl=pd.read_json(D+"fp/claims.jsonl",lines=True)
for t,x in o.groupby("topic"): tab[t].update(passages=int((cl.topic==t).sum()),claims=len(x),groups=int(x.gid.nunique()))
for f in out: d=tab[f["topic"]]["fights_by_status"]; d[f["status"]]=d.get(f["status"],0)+1
cost=json.load(open(H+"results/fp_fights.json"))["cost"]
tot=dict(passages_extracted=int(len(cl)),new_passages=4965,claims=len(o),groups=int(o.gid.nunique()),merge_pairs_judged=len(npairs),merge_pairs_p1_ge_half=int((npairs.p1>=0.5).sum()),
  fights_total=len(F),multi_group_fights=int((F.groups.map(len)>1).sum()),fights_reported=len(out),by_status={s:sum(f["status"]==s for f in out) for s in("fight","one_short","one_sided")},claims_judged=len(j))
json.dump(dict(totals=tot,per_topic=tab,cost=cost,fights=out),open(H+"results/fp_fights_v2.json","w"),indent=1,default=str)
print(json.dumps(tot))
