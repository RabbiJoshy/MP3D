import json, random, pandas as pd
R="/Users/joshuathomasamar/PycharmProjects/MP3D/"
D=R+"data/quick/research_2026_11_10/"
j=pd.read_json(D+"edu_jev_sides.jsonl",lines=True)
orig=json.load(open(R+"prototypes/research_2026_11_10/results/edu_contested.json"))
n_orig=len(orig["label"])
acc=j[j.jev==j.stance]
surv={}
for g,x in acc.groupby("g"):
    if x[x.stance=="FOR"].member_id.nunique()>=2 and x[x.stance=="AGAINST"].member_id.nunique()>=2: surv[g]=x
ids=sorted(surv, key=lambda g:str(g))
print("orig",n_orig,"surv",len(ids),[orig["label"].get(str(g)) for g in ids])
rng=random.Random(20261110)
pick=rng.sample(ids,5)
out=[]
for g in pick:
    cl=[]
    for side in ("FOR","AGAINST"):
        x=surv[g][surv[g].stance==side].sort_values("passage_id")
        recs=x.to_dict("records")
        if len(recs)>4: recs=rng.sample(recs,4)
        cl+=recs
    out.append({"g":int(g),"label":orig["label"][str(g)],"claims":cl})
json.dump({"n_orig":n_orig,"surv":[orig["label"].get(str(g)) for g in ids],"picked":out},open("/private/tmp/claude-501/-Users-joshuathomasamar--claude-spares-spare-JizpXL/ae3693f7-e27a-4399-b69d-d7f8212770bd/scratchpad/pick.json","w"),default=str)
for o in out:
    print("\n#####",o["g"],o["label"])
    for c in o["claims"]:
        print("\n--",c["passage_id"],c["mp"],c["party"],c["stance"],"| measure:",c["measure"],"\n",c["text"])
