"""v3 re-merge: same components as fp_remerge.py, but pair direction from MP overlap > party profile > p(Q2). No API. Writes results/fp_fights_v3.json (v2 schema + per-group 'rule')."""
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

import numpy as np
j0=pd.read_json(D+"fp/jev_sides.jsonl",lines=True); vj0=j0.set_index("passage_id").new_verdict
Mb=pd.read_parquet(R+"data/clean/members.parquet").set_index("member_id")
o["_v"]=o.passage_id.map(vj0); ac=o[o._v==o.stance]
gm={g:{int(m):s for m,s in zip(x.member_id,x.stance)} for g,x in ac.groupby("gid")}  # accepted positions per group
# an MP with both stances inside one group is ambiguous -> drop
for g,x in ac.groupby("gid"):
    c=x.groupby("member_id").stance.nunique(); gm[g]={m:s for m,s in gm[g].items() if c[m]==1}
parties=sorted(Mb.party_abbr.dropna().unique())
def vec(g,side): 
    v=np.zeros(len(parties))
    for m,s in gm.get(g,{}).items():
        if s==side and Mb.party_abbr.get(m) in parties: v[parties.index(Mb.party_abbr.get(m))]+=1
    return v
cos=lambda a,b: float(a@b/np.linalg.norm(a)/np.linalg.norm(b)) if a.any() and b.any() else 0.0
def decide(r):
    A,B=gm.get(r.a,{}),gm.get(r.b,{}); com=set(A)&set(B)
    if len(com)>=2:
        same=sum(A[m]==B[m] for m in com); opp=len(com)-same
        if same!=opp: return (1 if same>opp else -1),1
    if len(A)>=3 and len(B)>=3:
        af,aa,bf,ba=vec(r.a,"FOR"),vec(r.a,"AGAINST"),vec(r.b,"FOR"),vec(r.b,"AGAINST")
        ss=cos(af,bf)+cos(aa,ba); so=cos(af,ba)+cos(aa,bf)
        if abs(ss-so)>=0.3: return (1 if ss>so else -1),2
    return (1 if r.p2>=0.5 else -1),3
# step 1: components exactly as v2 (order and 6-cap unchanged; direction irrelevant to connectivity)
par={g:g for g in G.index}; members={g:[g] for g in G.index}
def f1(g):
    while par[g]!=g: g=par[g]
    return g
for r in P.itertuples():
    ra,rb=f1(r.a),f1(r.b)
    if ra==rb: continue
    if len(members[ra])+len(members[rb])>6: continue
    par[rb]=ra; members[ra]+=members[rb]
comp={g:f1(g) for g in G.index}
# step 2: directed edges within components, Kruskal by (rule, -p1) with sign union-find
E=[]
for r in P.itertuples():
    if comp[r.a]==comp[r.b]:
        d,ru=decide(r); E.append((ru,-r.p1,r.a,r.b,d))
E.sort(key=lambda e:(e[0],e[1]))
p2={g:g for g in G.index}; sg={g:1 for g in G.index}
def find(g):
    s=1
    while p2[g]!=g: s*=sg[g]; g=p2[g]
    return g,s
adj={g:[] for g in G.index}; ruledist={1:0,2:0,3:0}; dropped_e=0
for ru,_,a,b,d in E:
    (ra,sa),(rb,sb)=find(a),find(b)
    if ra==rb:
        if sa*sb!=d: dropped_e+=1
        continue
    p2[rb]=ra; sg[rb]=sa*d*sb; adj[a].append((b,d,ru)); adj[b].append((a,d,ru)); ruledist[ru]+=1
print("tree edges by rule",ruledist,"contradicting edges dropped",dropped_e)
roots={}
for g in G.index: roots.setdefault(comp[g],[]).append(g)
fights=[]
for k,(root,gs) in enumerate(roots.items()):
    anchor=max(gs,key=lambda g:G.mps[g]); dr={anchor:(1,"anchor")}; q=[anchor]
    while q:
        x=q.pop(0)
        for y,d,ru in adj[x]:
            if y not in dr: dr[y]=(dr[x][0]*d,{1:"mp_overlap",2:"party_profile",3:"p_q2"}[ru]); q.append(y)
    fights.append(dict(fight_id=k,topic=G.topic[root],label=G.label[anchor],groups=[dict(gid=int(g),label=G.label[g],direction=int(dr[g][0]),rule=dr[g][1]) for g in gs]))
pd.DataFrame(fights).to_json(D+"fp/fights_v3.jsonl",orient="records",lines=True)
print("multi-group fights",sum(len(f['groups'])>1 for f in fights))
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
json.dump(dict(totals=tot,per_topic=tab,cost=cost,fights=out),open(H+"results/fp_fights_v3.json","w"),indent=1,default=str)
print(json.dumps(tot))
