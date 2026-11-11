"""Step 4b: build fights from merge_pairs; Step 5 judge selection + run (--run); Step 6 recount."""
import sys, json, pandas as pd, numpy as np
sys.path.insert(0,"/Users/joshuathomasamar/PycharmProjects/MP3D/prototypes/research_2026_11_10")
R="/Users/joshuathomasamar/PycharmProjects/MP3D/"; D=R+"data/quick/research_2026_11_10/"; H=R+"prototypes/research_2026_11_10/"
o=pd.read_json(D+"fp/measures.jsonl",lines=True)
G=o.groupby("gid").agg(topic=("topic","first"),label=("label","first"),mps=("member_id","nunique")).reset_index().set_index("gid")
P=pd.read_json(D+"fp/merge_pairs.jsonl",lines=True)
P=P[P.p1>=0.5].sort_values("p1",ascending=False)
par={g:g for g in G.index}; sgn={g:1 for g in G.index}  # sgn: direction relative to parent
members={g:[g] for g in G.index}
def find(g):
    s=1
    while par[g]!=g: s*=sgn[g]; g=par[g]
    return g,s
acc=[];rej=0
for r in P.itertuples():
    d=1 if r.p2>=0.5 else -1
    ra,sa=find(r.a); rb,sb=find(r.b)
    if ra==rb:
        if sa*sb!=d: rej+=1   # contradictory, weaker link dropped
        else: acc.append(r)
        continue
    if len(members[ra])+len(members[rb])>6: rej+=1; continue
    par[rb]=ra; sgn[rb]=sa*d*sb  # want dir(b rel ra)= sa*d ; sgn[rb]*sb = sa*d
    members[ra]+=members[rb]; acc.append(r)
fights=[]
roots={}
for g in G.index: roots.setdefault(find(g)[0],[]).append(g)
for k,(root,gs) in enumerate(roots.items()):
    anchor=max(gs,key=lambda g:G.mps[g]); sa=find(anchor)[1]
    fights.append(dict(fight_id=k,topic=G.topic[root],label=G.label[anchor],groups=[dict(gid=int(g),label=G.label[g],direction=int(find(g)[1]*sa)) for g in gs]))
json.dump(fights,open(D+"fp/fights.jsonl","w")) if False else pd.DataFrame(fights).to_json(D+"fp/fights.jsonl",orient="records",lines=True)
gd={g["gid"]:(f["fight_id"],g["direction"]) for f in fights for g in f["groups"]}
o["fight"]=o.gid.map(lambda g:gd[g][0]); o["dir"]=o.gid.map(lambda g:gd[g][1])
nm=o.groupby("fight").member_id.nunique()
print("pairs judged",len(P)+ (0),"accepted merges(joined or consistent)",len(acc),"rejected/capped",rej,"| multi-group fights",sum(len(f['groups'])>1 for f in fights))
big=o[o.fight.map(nm)>=3].copy()
print("claims in fights >=3 MPs",len(big),"fights",big.fight.nunique())
# reuse education verdicts
ev=pd.read_json(D+"edu2_jev2_sides.jsonl",lines=True)[["passage_id","label","p_for","p_against","new_verdict"]]
big=big.merge(ev,on=["passage_id","label"],how="left",suffixes=("","_old")); big["reused"]=big.new_verdict.notna()
todo=big[~big.reused]; print("reused",int(big.reused.sum()),"to judge",len(todo))
est=(sum(min(len(t),3000) for t in todo.text)/4*2+len(todo)*2*120)*0.042/1e6; print("est cost $",round(est,4))
BUDGET=float(sys.argv[sys.argv.index("--budget")+1]) if "--budget" in sys.argv else 0.03
if "--run" in sys.argv:
    from jev_sides2_run import ask, verdict, used
    from concurrent.futures import ThreadPoolExecutor
    # shrink: judge fights in descending distinct-MP order until budget
    todo=todo.assign(nmp=todo.fight.map(nm)).sort_values(["nmp","fight"],ascending=False)
    c=(todo.text.str.len().clip(upper=3000)/4*2+240)*0.042/1e6; todo=todo[c.cumsum()<=BUDGET]; print("judging",len(todo),"claims (budget",BUDGET,")")
    rows=todo.to_dict("records")
    with ThreadPoolExecutor(8) as ex: res=list(ex.map(ask,rows))
    for r,(f,a) in zip(rows,res): r["p_for"],r["p_against"],r["new_verdict"]=f,a,verdict(f,a)
    out=pd.concat([big[big.reused],pd.DataFrame(rows)]); out["reused"]=out["reused"].fillna(False)
    out.drop(columns=["text"]).to_json(D+"fp/jev_sides.jsonl",orient="records",lines=True)
    print("tokens",used[0],"cost $",used[0]*0.042/1e6); json.dump({"tokens":used[0]},open(H+"results/fp_jev_tokens.json","w"))
