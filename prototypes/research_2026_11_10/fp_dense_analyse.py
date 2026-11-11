import json, random, collections, pandas as pd
ROOT="/Users/joshuathomasamar/PycharmProjects/MP3D"; D=ROOT+"/data/quick/research_2026_11_10"; R=ROOT+"/prototypes/research_2026_11_10/results"
inp=json.load(open(D+"/fp/dense_input_fights.json"))
s=pd.read_json(D+"/fp/dense_sides.jsonl",lines=True)
p=pd.read_parquet(ROOT+"/data/clean/passages.parquet",columns=["passage_id","member_id"])
s=s.merge(p,on="passage_id"); s=s[s.verdict.isin(["FOR","AGAINST"])]
mem=pd.read_parquet(ROOT+"/data/clean/members.parquet").drop_duplicates("member_id").set_index("member_id")
out=json.loads(json.dumps(inp)); stats=[]; newpos=[]; flips=[]; splits=[]
for f in out["fights"]:
    if f["status"]!="fight": continue
    before={}
    for side,l in(("FOR","mps_for"),("AGAINST","mps_against")):
        for m in f[l]: before[m["member_id"]]=side
    for m in f["mps_both_sides"]: before[m["member_id"]]="mixed"
    meta={m["member_id"]:(m["name"],m["party"]) for l in("mps_for","mps_against","mps_both_sides") for m in f[l]}
    g=s[s.fight==f["fight_id"]]; new={}; F,A,M=[],[],[]
    for mid,gg in g.groupby("member_id"):
        mid=int(mid); v=[1 if x=="FOR" else -1 for x in gg.verdict]; st=sum(v)/len(v)
        name,party=meta.get(mid,(mem.name.get(mid,"?"),mem.party_abbr.get(mid,"?")))
        e={"member_id":mid,"name":name,"party":party,"claims":[{"passage_id":a,"measure":f["label"],"side":b} for a,b in zip(gg.passage_id,gg.verdict)],"stance":round(st,3),"focus":len(v)}
        side="FOR" if st>0 else "AGAINST" if st<0 else "mixed"; new[mid]=side
        {"FOR":F,"AGAINST":A,"mixed":M}[side].append(e)
    fid=f["fight_id"]
    bF=sum(v=="FOR" for v in before.values()); bA=sum(v=="AGAINST" for v in before.values())
    flipped=[m for m in new if m in before and before[m]!=new[m]]
    added=[m for m in new if m not in before]; lost=[m for m in before if m not in new]
    for m in added: newpos.append((fid,m,new[m]))
    for m in flipped: flips.append((fid,m,before[m],new[m]))
    # party splits
    byp=collections.defaultdict(lambda:{"FOR":[],"AGAINST":[]})
    for e in F: byp[e["party"]]["FOR"].append(e["name"])
    for e in A: byp[e["party"]]["AGAINST"].append(e["name"])
    for pt,d in byp.items():
        if d["FOR"] and d["AGAINST"]: splits.append((fid,f["label"],pt,d["FOR"],d["AGAINST"]))
    f["mps_for"],f["mps_against"],f["mps_both_sides"]=F,A,M; f["dense"]=True
    f["status"]="fight" if len(F)>=2 and len(A)>=2 else "one_short" if (len(F)>=2 or len(A)>=2) and len(F)>=1 and len(A)>=1 else "one_sided"
    f["n_dense_candidates"]=int((s.fight==fid).sum())
    stats.append({"fight":fid,"label":f["label"],"topic":f["topic"],"before_for":bF,"before_against":bA,"before":bF+bA,"after_for":len(F),"after_against":len(A),"after_mixed":len(M),"after":len(F)+len(A),"flipped":len(flipped),"added":len(added),"lost":len(lost),"status":f["status"]})
out["totals"]["dense_by_status"]=dict(collections.Counter(f["status"] for f in out["fights"]))
json.dump(out,open(R+"/fp_fights_dense.json","w"),indent=1)
json.dump({"stats":stats,"flips":flips,"splits":splits,"newpos":newpos},open(R+"/fp_dense_stats.json","w"),indent=1,default=int)
for x in sorted(stats,key=lambda x:-x["after"]): print(x)
print("total before",sum(x["before"] for x in stats),"after",sum(x["after"] for x in stats),"flips",len(flips),"new",len(newpos),"lost",sum(x["lost"] for x in stats))
for x in splits: print(x)
