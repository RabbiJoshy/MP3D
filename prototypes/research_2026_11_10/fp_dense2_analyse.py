import json, collections, pandas as pd
ROOT="/Users/joshuathomasamar/PycharmProjects/MP3D"; D=ROOT+"/data/quick/research_2026_11_10"; R=ROOT+"/prototypes/research_2026_11_10/results"
inp=json.load(open(R+"/fp_fights_final.json"))
s=pd.read_json(D+"/fp/dense_sides2.jsonl",lines=True)
p=pd.read_parquet(ROOT+"/data/clean/passages.parquet",columns=["passage_id","member_id"])
s=s.merge(p,on="passage_id"); s=s[s.verdict.isin(["FOR","AGAINST"])]
mem=pd.read_parquet(ROOT+"/data/clean/members.parquet").drop_duplicates("member_id").set_index("member_id")
out=json.loads(json.dumps(inp)); keep=[]; before_mps=set(); after_mps=set(); newpos=[]
def status(F,A): return "fight" if F>=2 and A>=2 else "one_short" if (F>=2 or A>=2) and F>=1 and A>=1 else "one_sided"
old={}
for f in out["fights"]:
    for l in("mps_for","mps_against","mps_both_sides"):
        for m in f[l]: before_mps.add(m["member_id"])
    old[f["fight_id"]]=f["status"]
    if f.get("dense"):
        for l in("mps_for","mps_against"):
            for m in f[l]: after_mps.add(m["member_id"])
        keep.append(f); continue
    meta={m["member_id"]:(m["name"],m["party"]) for l in("mps_for","mps_against","mps_both_sides") for m in f[l]}
    before={m["member_id"] for l in("mps_for","mps_against","mps_both_sides") for m in f[l]}
    g=s[s.fight==f["fight_id"]]; F,A,M=[],[],[]
    for mid,gg in g.groupby("member_id"):
        mid=int(mid); v=[1 if x=="FOR" else -1 for x in gg.verdict]; st=sum(v)/len(v)
        name,party=meta.get(mid,(mem.name.get(mid,"?"),mem.party_abbr.get(mid,"?")))
        e={"member_id":mid,"name":name,"party":party,"claims":[{"passage_id":a,"measure":f["label"],"side":b} for a,b in zip(gg.passage_id,gg.verdict)],"stance":round(st,3),"focus":len(v)}
        {"FOR":F,"AGAINST":A,"mixed":M}["FOR" if st>0 else "AGAINST" if st<0 else "mixed"].append(e)
        if mid not in before: newpos.append({"fight":f["fight_id"],"member_id":mid,"name":name,"party":party,"stance":e["stance"],"focus":e["focus"],"claims":e["claims"]})
    f["mps_for"],f["mps_against"],f["mps_both_sides"]=F,A,M; f["dense"]=True
    f["status"]=status(len(F),len(A)); f["n_dense_candidates"]=int((s.fight==f["fight_id"]).sum())
    if len(F)+len(A)>=3:
        keep.append(f)
        for e in F+A: after_mps.add(e["member_id"])
out["fights"]=keep
C=collections.Counter(f["status"] for f in keep)
out["totals"]["by_status"]=dict(C); out["totals"]["dense_by_status"]=dict(C); out["totals"]["fights_reported"]=len(keep)
pt={}
for f in keep: pt.setdefault(f["topic"],collections.Counter())[f["status"]]+=1
for t,v in out["per_topic"].items(): v["fights_by_status"]=dict(pt.get(t,{}))
json.dump(out,open(R+"/fp_fights_final2.json","w"),indent=1)
json.dump(newpos,open(R+"/fp_dense2_newpos.json","w"),indent=1)
print("before",dict(collections.Counter(old.values())),"after",dict(C),"fights",len(keep))
print("MPs before",len(before_mps),"after",len(after_mps),"new positions",len(newpos))
rt=collections.Counter(f["topic"] for f in keep if f["status"]=="fight"); print(rt.most_common())
