"""sk21: cold-start analysis. Free/local. Reuses sk20_run.py code (exec of its text up to '# ---- metrics')."""
import json, random
src=open("/Users/joshuathomasamar/PycharmProjects/MP3D/prototypes/research_2026_11_10/sk20_run.py").read()
exec(src[:src.index("# ---- metrics")])
OUT=R/"data/quick/research_2026_11_10/sk21"; OUT.mkdir(parents=True,exist_ok=True)
def auc(s,l):
    r=rankdata(s); n1=l.sum(); n0=len(l)-n1
    return (r[l].sum()-n1*(n1+1)/2)/(n1*n0)
res={}
# ---- step 1 groups
hc=(Sg&elig)[:,tr].sum(1)
grp=np.where(hc==0,"ZERO",np.where(hc<10,"LOW","HIGH"))
mains=["Lab","Con","LD"]
inte=cand[:,te].any(1)  # MPs who are candidates for at least one TEST EDM
res["step1_mps_by_group_party_all"]={g:{p:int(((grp==g)&(party==p)).sum()) for p in sorted(set(party))} for g in ["ZERO","LOW","HIGH"]}
res["step1_mps_by_group_party_in_test"]={g:{p:int(((grp==g)&(party==p)&inte).sum()) for p in sorted(set(party))} for g in ["ZERO","LOW","HIGH"]}
# ---- step 2
names2=["B0","H","S_pre","B0+H","B0+H+S"]
def cells_for(gm,SCd,names,js=te):
    out={n:[] for n in names}; meta=[]
    for j in js:
        base=cand[:,j]&gm
        for p in sorted(set(party[base])):
            mm=base&(party==p); l=Sg[:,j]; ns=(mm&l).sum(); nn=(mm&~l).sum()
            if ns>=3 and nn>=3:
                meta.append((j,p))
                for n in names: out[n].append(auc(SCd[n][j][mm],l[mm]))
    return {n:np.array(v) for n,v in out.items()},meta
def pooled(gm,SCd,names,js=te):
    out={n:[] for n in names}
    for j in js:
        mm=cand[:,j]&gm; l=Sg[:,j]
        if (mm&l).sum()<3 or (mm&~l).sum()<3: continue
        for n in names:
            s=SCd[n][j].copy()
            for p in set(party[mm]): k=mm&(party==p); s[k]-=s[k].mean()
            out[n].append(auc(s[mm],l[mm]))
    return {n:(float(np.mean(v)) if len(v) else None) for n,v in out.items()},len(out[names[0]])
s2={}
for g in ["ZERO","LOW","HIGH"]:
    gm=grp==g; o,meta=cells_for(gm,SC,names2); mp=np.array([m[1] for m in meta])
    s2[g]={"ncells":len(meta),"mean":{n:(float(o[n].mean()) if len(meta) else None) for n in names2}}
    s2[g]["by_party"]={p:{"ncells":int((mp==p).sum()),"mean":{n:float(o[n][mp==p].mean()) for n in names2}} for p in sorted(set(mp)) if (mp==p).sum()>=20}
    s2[g]["by_party_all_counts"]={p:int((mp==p).sum()) for p in sorted(set(mp))}
    pm,nc=pooled(gm,SC,names2); s2[g]["pooled_party_centred"]=dict(auc=pm,nedm=nc)
res["step2"]=s2
# ---- step 3: simulated cold start
# party+S refit (and +M) on TRAIN fit rows: need party-rate feature
rng2=random.Random(20261114); Xs=[];ys=[]
for j in half:
    hist=trs[date[trs]<date[j]]; Rt_,prv_=rate(hist); H_,Mm_=feats(j,hist,Rt_); msk=cand[:,j]
    f=np.c_[logit(prv_),centre(S_pre[:,j],msk),centre(Mm_,msk)]
    pos=np.where(msk&Sg[:,j])[0]; neg=np.where(msk&~Sg[:,j])[0]
    neg=np.array(rng2.sample(list(neg),min(len(neg),20*len(pos)))) if len(pos) else np.array([],int)
    ix=np.r_[pos,neg].astype(int); Xs.append(f[ix]); ys.append(np.r_[np.ones(len(pos)),np.zeros(len(neg))])
X2=np.vstack(Xs); y2=np.concatenate(ys); sc2=StandardScaler().fit(X2)
mPS=LogisticRegression(max_iter=1000).fit(sc2.transform(X2)[:,[0,1]],y2)
mPSM=LogisticRegression(max_iter=1000).fit(sc2.transform(X2),y2)
res["step3_refit_coefs"]={"party+S":mPS.coef_[0].round(3).tolist(),"party+S+M":mPSM.coef_[0].round(3).tolist()}
HIGH=np.where(grp=="HIGH")[0]; HIGHc=np.array([i for i in HIGH if inte[i]])
teA=np.array(te); posn=lambda js:np.searchsorted(teA,js)
B=1000; brng=np.random.default_rng(20261114)
W=np.stack([np.bincount(brng.integers(0,len(te),len(te)),minlength=len(te)) for _ in range(B)])
ci=lambda a:[float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]
sim_names=["real B0+H","real B0+H+S","party only (B0+H masked)","S alone","B0+H+S (masked hist)","party+S refit","M alone","party+S+M refit"]
draws=[]; seeds=[20261114+k for k in range(5)]
for sd in seeds:
    r3=random.Random(sd); chosen=np.zeros(nM,bool); pick=r3.sample(list(HIGHc),int(round(0.3*len(HIGHc)))); chosen[pick]=True
    SCs={n:{} for n in sim_names}
    for j in te:
        msk=cand[:,j]; H,Mm=feats(j,tr,Rt); sp=centre(S_pre[:,j],msk); mc=centre(Mm,msk)
        B0m=Rt.copy(); Hm=H.copy()
        for p in set(party[msk]):
            pp=msk&(party==p); Hm[chosen&(party==p)]=H[pp].mean(); B0m[chosen&(party==p)]=prv[chosen&(party==p)]
        fr=sc.transform(np.c_[logit(Rt),H,sp,mc]); fm=sc.transform(np.c_[logit(B0m),Hm,sp,mc])
        SCs["real B0+H"][j]=models["B0+H"][1].decision_function(fr[:,[0,1]])
        SCs["real B0+H+S"][j]=models["B0+H+S"][1].decision_function(fr[:,[0,1,2]])
        SCs["party only (B0+H masked)"][j]=models["B0+H"][1].decision_function(fm[:,[0,1]])
        SCs["S alone"][j]=sp
        SCs["B0+H+S (masked hist)"][j]=models["B0+H+S"][1].decision_function(fm[:,[0,1,2]])
        f2=sc2.transform(np.c_[logit(prv),sp,mc])
        SCs["party+S refit"][j]=mPS.decision_function(f2[:,[0,1]]); SCs["M alone"][j]=mc; SCs["party+S+M refit"][j]=mPSM.decision_function(f2)
    o,meta=cells_for(chosen,SCs,sim_names); cj=np.array([m[0] for m in meta]); pn=posn(cj)
    den=W@np.bincount(pn,minlength=len(te)); bsd={}
    d={"seed":sd,"n_chosen":int(chosen.sum()),"ncells":len(meta),"auc":{},"ci":{}}
    for n in sim_names:
        d["auc"][n]=float(o[n].mean()); bsd[n]=(W@np.bincount(pn,weights=o[n],minlength=len(te)))/den; d["ci"][n]=ci(bsd[n])
    dd_=bsd["real B0+H"]-bsd["party+S refit"]; d["real_minus_partyS"]={"point":d["auc"]["real B0+H"]-d["auc"]["party+S refit"],"ci":ci(dd_)}
    d["gap_recovered"]=(d["auc"]["party+S refit"]-0.5)/(d["auc"]["real B0+H"]-0.5)
    d["gap_recovered_ci"]=ci((bsd["party+S refit"]-0.5)/(bsd["real B0+H"]-0.5))
    d["map_add"]={"point":d["auc"]["party+S+M refit"]-d["auc"]["party+S refit"],"ci":ci(bsd["party+S+M refit"]-bsd["party+S refit"])}
    draws.append(d); print(sd,d["ncells"],{k:round(v,3) for k,v in d["auc"].items()})
summ3={n:dict(mean=float(np.mean([d["auc"][n] for d in draws])),min=float(min(d["auc"][n] for d in draws)),max=float(max(d["auc"][n] for d in draws)),mean_ci_lo=float(np.mean([d["ci"][n][0] for d in draws])),mean_ci_hi=float(np.mean([d["ci"][n][1] for d in draws]))) for n in sim_names}
gr=[d["gap_recovered"] for d in draws]
res["step3"]=dict(draws=draws,summary=summ3,gap_recovered=dict(mean=float(np.mean(gr)),min=float(min(gr)),max=float(max(gr))),
  passes_reading=bool(summ3["party+S refit"]["mean_ci_lo"]>0.55 and np.mean(gr)>=1/3),
  note="B0+H masked => constant within party => AUC 0.5; weights of masked B0+H+S therefore equal S ranking")
# ---- step 4 ex-ministers
last=pd.Timestamp(E.date.max()); ex=np.zeros(nM,bool); exinfo=[]
for i,r in M.iterrows():
    posts=json.loads(r.gov_posts) if isinstance(r.gov_posts,str) and r.gov_posts else []
    for p in posts:
        s=pd.to_datetime(p["startDate"]) if p.get("startDate") else pd.Timestamp("1900-01-01")
        e=pd.to_datetime(p["endDate"]) if p.get("endDate") else None
        if s<pd.Timestamp(cutd) and e is not None and e<last and (cand[i,te].any()): ex[i]=True
n_ex=int(ex.sum()); n_ex_sig=int((Sg[:,te]&cand[:,te]&ex[:,None]).any(1).sum()) if n_ex else 0
res["step4"]=dict(n_exministers=n_ex,n_with_test_signatures=n_ex_sig,by_party={p:int((ex&(party==p)).sum()) for p in sorted(set(party[ex]))},
   mean_train_sigs=float(hc[ex].mean()) if n_ex else None)
if n_ex_sig>=10:
    pm,nc=pooled(ex,SC,names2); o,meta=cells_for(ex,SC,names2)
    res["step4"]["pooled_party_centred"]=dict(auc=pm,nedm=nc); res["step4"]["cells"]=dict(n=len(meta),mean={n:(float(o[n].mean()) if len(meta) else None) for n in names2})
else: res["step4"]["verdict"]="too few"
# ---- step 2 bootstrap CIs on group means (for chart)
for g in ["ZERO","LOW","HIGH"]:
    o,meta=cells_for(grp==g,SC,names2)
    if len(meta):
        cj=np.array([m[0] for m in meta]); pn=posn(cj); den=W@np.bincount(pn,minlength=len(te))
        res["step2"][g]["ci"]={n:ci((W@np.bincount(pn,weights=o[n],minlength=len(te)))/np.maximum(den,1)) for n in names2}
        for a,b in [("B0+H+S","B0+H")]:
            dd_=(W@np.bincount(pn,weights=o[a]-o[b],minlength=len(te)))/np.maximum(den,1); res["step2"][g]["diff_S_over_B0H"]=dict(point=float((o[a]-o[b]).mean()),ci=ci(dd_))
json.dump(res,open(RES/"sk21_summary.json","w"),indent=1,default=float)
# ---- chart
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":12})
fig,ax=plt.subplots(1,2,figsize=(17,6.5))
lab={"B0":"B0 activity","H":"H history","S_pre":"S speeches","B0+H":"B0+H","B0+H+S":"B0+H+S"}; cols=["#9aa5b1","#7b9acc","#e08a3c","#4c78a8","#2b8a5a"]
a=ax[0]; w=0.16
for k,n in enumerate(names2):
    for gi,g in enumerate(["ZERO","LOW","HIGH"]):
        v=res["step2"][g]["mean"][n]
        if v is None: continue
        x=gi+(k-2)*w; lo,hi=res["step2"][g]["ci"][n]; a.bar(x,v,w,color=cols[k],label=lab[n] if gi==0 else None); a.errorbar(x,v,yerr=[[v-lo],[hi-v]],fmt="none",ecolor="k",capsize=2)
        a.text(x,hi+0.005,f"{v:.2f}",ha="center",fontsize=8,rotation=90)
a.set_xticks(range(3)); a.set_xticklabels([f"{g}\n({res['step2'][g]['ncells']} cells)" for g in ["ZERO","LOW","HIGH"]]); a.axhline(0.5,ls="--",c="r"); a.set_ylim(0.4,1.05); a.set_ylabel("Within-party AUC"); a.set_title("AUC by signing-history group"); a.legend(fontsize=9,loc="lower right")
a=ax[1]; ns=["real B0+H","party+S refit","party+S+M refit"]; lb=["Real history\n(B0+H)","No history +\nspeeches","No history +\nspeeches + map"]
for k,n in enumerate(ns):
    v=summ3[n]["mean"]; a.bar(k,v,color=["#4c78a8","#e08a3c","#7a5aa6"][k]); a.errorbar(k,v,yerr=[[v-summ3[n]["min"]],[summ3[n]["max"]-v]],fmt="none",ecolor="k",capsize=5); a.text(k,summ3[n]["max"]+0.006,f"{v:.3f}",ha="center")
a.set_xticks(range(3)); a.set_xticklabels(lb); a.axhline(0.5,ls="--",c="r"); a.set_ylim(0.4,1.0); a.set_ylabel("Within-party AUC"); a.set_title("Simulated cold start (30% of HIGH MPs; range over 5 draws)")
plt.tight_layout(); plt.savefig(RES/"sk21_chart.png",dpi=130)
print(json.dumps({k:res[k] for k in ["step1_mps_by_group_party_in_test","step2","step4"]},indent=1,default=float)); print(json.dumps({k:res["step3"][k] for k in ["summary","gap_recovered","passes_reading"]},indent=1)); print(res["step3_refit_coefs"])
print([ (d["real_minus_partyS"],d["gap_recovered_ci"],d["map_add"]) for d in draws[:2]])
