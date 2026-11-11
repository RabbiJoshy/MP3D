import json, pandas as pd
R="/Users/joshuathomasamar/PycharmProjects/MP3D/"; D=R+"data/quick/research_2026_11_10/"; H=R+"prototypes/research_2026_11_10/"
o=pd.read_json(D+"fp/measures.jsonl",lines=True); F=pd.read_json(D+"fp/fights.jsonl",lines=True).set_index("fight_id")
j=pd.read_json(D+"fp/jev_sides.jsonl",lines=True)
gd={g["gid"]:(f,g["direction"]) for f,gs in F.groups.items() for g in gs}
o["fight"]=o.gid.map(lambda g:gd[g][0]); o["dir"]=o.gid.map(lambda g:gd[g][1])
vj=j.set_index("passage_id").new_verdict
nm=o.groupby("fight").member_id.nunique()
big=o[o.fight.map(nm)>=3].copy()
big["verdict"]=big.passage_id.map(vj).fillna("NOT_JUDGED")
M=pd.read_parquet(R+"data/clean/members.parquet").set_index("member_id")
out=[];npairs=pd.read_json(D+"fp/merge_pairs.jsonl",lines=True)
for f,x in big.groupby("fight"):
    acc=x[x.verdict==x.stance]; drop=x[x.verdict!=x.stance]
    acc=acc.assign(side=[s if d==1 else ("AGAINST" if s=="FOR" else "FOR") for s,d in zip(acc.stance,acc.dir)])
    sides=acc.groupby("member_id").side.agg(set); both=[int(m) for m,s in sides.items() if len(s)>1]
    def mp(m,side=None):
        a=acc[acc.member_id==m]; a=a[a.side==side] if side else a
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
tk=lambda n:json.load(open(H+f"results/{n}.json"))["tokens"]
cost=dict(extract=dict(tokens_in=1886175,tokens_out=104530,usd=0.0723),merge=dict(tokens=tk("fp_merge_tokens"),usd=round(tk("fp_merge_tokens")*0.042/1e6,4)),judge=dict(tokens=tk("fp_jev_tokens"),usd=round(tk("fp_jev_tokens")*0.042/1e6,4)))
cost["total_usd"]=round(sum(v["usd"] for v in cost.values()),4)
tot=dict(passages_extracted=int(len(cl)),new_passages=4965,claims=len(o),groups=int(o.gid.nunique()),merge_pairs_judged=len(npairs),merge_pairs_p1_ge_half=int((npairs.p1>=0.5).sum()),
  fights_total=len(F),multi_group_fights=int((F.groups.map(len)>1).sum()),fights_reported=len(out),by_status={s:sum(f["status"]==s for f in out) for s in("fight","one_short","one_sided")},claims_judged=len(j))
json.dump(dict(totals=tot,per_topic=tab,cost=cost,fights=out),open(H+"results/fp_fights.json","w"),indent=1,default=str)
print(json.dumps(tot)); print(json.dumps(cost))
print(pd.DataFrame({t:{s:v["fights_by_status"].get(s,0) for s in("fight","one_short","one_sided")} for t,v in tab.items()}).T.to_string())
for f in sorted([f for f in out if f["status"]=="fight"],key=lambda f:-(len(f["mps_for"])+len(f["mps_against"])))[:8]: print(f["topic"],"|",f["label"],"|",len(f["mps_for"]),len(f["mps_against"]))
mp_=npairs[npairs.p1>=0.5].sort_values("p1",ascending=False)
print(mp_[["label_a","label_b","p1","p2"]].sample(8,random_state=1).to_string())
print(npairs[npairs.label_a.str.contains("VAT|tax",case=False)&npairs.label_b.str.contains("VAT|tax|private",case=False)&(npairs.cos>0)][["label_a","label_b","p1","p2"]].head(12).to_string())
