import json, pandas as pd
D="/Users/joshuathomasamar/PycharmProjects/MP3D/data/quick/research_2026_11_10/"
R="/Users/joshuathomasamar/PycharmProjects/MP3D/prototypes/research_2026_11_10/results/"
j=pd.read_json(D+"edu2_jev2_sides.jsonl",lines=True)
m=pd.read_json(D+"edu2_measures.jsonl",lines=True); allc=pd.read_json(D+"edu2_claims.jsonl",lines=True)
samp=pd.read_json(D+"edu2_sample.jsonl",lines=True)
j["acc"]=j.new_verdict==j.stance
groups=[]
for g,x in j.groupby("g"):
    a=x[x.acc]; F=a[a.stance=="FOR"]; A=a[a.stance=="AGAINST"]
    f,ag=set(F.member_id),set(A.member_id); both=f&ag
    fs,as_=f-both,ag-both
    pf=sorted(set(F.party)); pa=sorted(set(A.party))
    # same-party splits (strict sides)
    pm=a.drop_duplicates("member_id").set_index("member_id").party.to_dict()
    nm=a.drop_duplicates("member_id").set_index("member_id").mp.to_dict()
    sp=[]
    for p in set(pm[i] for i in fs)&set(pm[i] for i in as_):
        sp.append({"party":p,"for":sorted(nm[i] for i in fs if pm[i]==p),"against":sorted(nm[i] for i in as_ if pm[i]==p)})
    ext=x.drop_duplicates("member_id")
    groups.append({"g":int(g),"label":x.label.iloc[0],"n_claims":len(x),
      "extractor_for":int(x[x.stance=="FOR"].member_id.nunique()),"extractor_against":int(x[x.stance=="AGAINST"].member_id.nunique()),
      "mps_for":len(f),"mps_against":len(ag),"parties_for":pf,"parties_against":pa,
      "both_sides_mps":sorted(nm.get(i,str(i)) for i in both),"strict_for":len(fs),"strict_against":len(as_),
      "contested":len(f)>=2 and len(ag)>=2,"strict_contested":len(fs)>=2 and len(as_)>=2,"same_party_splits":sp})
sh=j.new_verdict.value_counts(normalize=True).round(3).to_dict()
tj=json.load(open(R+"edu2_jev_tokens.json"))["tokens"]
ex_cost=(1423362*0.03+72246*0.15)/1e6; j_cost=tj*0.042/1e6
out={"sample":{"education_passages_total":5226,"in_edu_sample_jsonl":1097,"remaining":4129,
  "eligible_bill_amend_oppday_not_in_old_sample":0,"note":"Heading regex (Bill|Amendment|Opposition Day|New Clause|Clause|Motion|Lords) + words>=60 + non-Chair: all 924 such passages already in edu_sample.jsonl (which has 1097 rows, not 970; 173 extra come from committee/other debates). Under looser rule (any length) 180 of the remaining match. So 0 eligible; topped up by random from 4129 remaining (all taken, <5000).",
  "n_sampled_new":len(samp),"split":{"eligible":0,"topup":len(samp)}},
 "n_passages_total":len(allc),"n_claims_with_measure":len(m),"n_groups":int(m.g.nunique()),
 "n_contested_by_extractor":int(j.g.nunique()),"n_judged_claims":len(j),"n_reused":int(j.reused.sum()),"verdict_shares":sh,
 "n_contested_after_judge":sum(g["contested"] for g in groups),"n_contested_strict":sum(g["strict_contested"] for g in groups),
 "groups":groups,"tokens":{"extract_in":1423362,"extract_out":72246,"judge_input":tj},
 "cost_usd":{"extract":round(ex_cost,5),"judge":round(j_cost,5),"total":round(ex_cost+j_cost,5)}}
json.dump(out,open(R+"edu2_full.json","w"),indent=1)
print({k:out[k] for k in ["n_passages_total","n_claims_with_measure","n_groups","n_contested_by_extractor","n_judged_claims","n_reused","verdict_shares","n_contested_after_judge","n_contested_strict","cost_usd"]})
for g in sorted([g for g in groups if g["contested"]],key=lambda g:-min(g["strict_for"],g["strict_against"]))[:12]:
    print(g["label"][:60],g["mps_for"],g["mps_against"],g["parties_for"],g["parties_against"],"both",len(g["both_sides_mps"]),"strict",g["strict_for"],g["strict_against"],[(s["party"],len(s["for"]),len(s["against"])) for s in g["same_party_splits"]])
