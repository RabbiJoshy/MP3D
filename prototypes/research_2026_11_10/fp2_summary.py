import json, collections, sqlite3, pandas as pd
ROOT="/Users/joshuathomasamar/PycharmProjects/MP3D/"; R=ROOT+"prototypes/research_2026_11_10/results/"; D=ROOT+"data/quick/research_2026_11_10/fp2/"
d=json.load(open(R+"fp2_fights.json")); F=d["fights"]
sel=pd.read_json(D+"selected_all.jsonl",lines=True); newp=sel[sel.isnew].groupby("topic").size().to_dict()
pre=json.load(open(R+"fp2_fights_pre_summary.json")); lab=json.load(open(R+"fp2_labels.json")); xm=json.load(open(R+"fp2_xmerge.json")); dt=json.load(open(R+"fp2_dense_tokens.json"))
two={}; per={}
for t,v in d["per_topic"].items(): per[t]=dict(new_passages=newp.get(t,0),**v,fights_ge4_extractor_mps=pre["per_topic_ge4"].get(t,0))
for f in F:
    if f["status"]=="fight": two[f["topic"]]=two.get(f["topic"],0)+1
mp=collections.defaultdict(set)
for f in F:
    for k in("mps_for","mps_against"):
        for m in f[k]: mp[m["member_id"]].add(f["topic"])
cov={n:sum(len(v)>=n for v in mp.values()) for n in (1,2,3,4)}
db=sqlite3.connect(D+"budget.sqlite"); led=db.execute("select sum(usd) from spend").fetchone()[0]
cost=dict(extract_new=0.1813,merge_pairs_10234=0.3710,relabel_deepseek_v3_2=lab["usd"],relabel_abandoned_qwen_try=0.0735,xmerge=xm["usd"],dense=dt["usd"])
cost["total_usd"]=round(sum(cost.values()),4)
S=dict(totals=d["totals"],new_passages_total=int(sum(newp.values())),new_passages_per_topic=newp,pre=pre,per_topic=per,two_sided_per_topic=two,
  mps_with_position=len(mp),mps_in_topics_ge=cov,labelling_model=lab["model"],labelling_price_per_M_in_out=lab["price_per_M_in_out"],labelling_tokens=lab["tokens"],
  xmerge=dict(units=xm["n_units"],merged_units=xm["n_merged_units"]),dense_tokens=dt,cost=cost,ledger_usd=round(led,4),skipped_units=len(json.load(open(R+"fp2_skipped_units.json"))))
json.dump(S,open(R+"fp2_summary.json","w"),indent=1,default=str)
print(json.dumps({k:S[k] for k in("new_passages_total","mps_with_position","mps_in_topics_ge","two_sided_per_topic","cost","ledger_usd","skipped_units","xmerge")}))
