"""sk20 analysis: EDM signing prediction. Free/local. All choices fixed in the task brief."""
import json, pathlib, random, re
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
R=pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D"); D=R/"data/quick/research_2026_11_10/sk20"; RES=R/"prototypes/research_2026_11_10/results"
SEED=20261113; rng=random.Random(SEED); nrg=np.random.default_rng(SEED)
E=pd.read_parquet(D/"edms.parquet"); M=pd.read_parquet(D/"members.parquet"); z=np.load(D/"mat.npz"); Sg,elig,prim=z["Sg"],z["elig"],z["prim"]
EE=np.load(D/"edm_emb.npy"); SS=np.load(D/"S.npz"); S_any,S_pre=SS["S_any"],SS["S_pre"]
prep=json.load(open(D/"prep_stats.json"))
nM,nE=Sg.shape; party=M.party_abbr.values
# map coords
h=open(R/"prototypes/app_hierarchy.html").read(); i=h.index('<script id="d" type="application/json">')+len('<script id="d" type="application/json">')
dd=json.loads(h[i:h.index("</script>",i)]); cid={m["id"]:k for k,m in enumerate(dd["mps"])}
XYZ=np.array(dd["xyz"],float); onmap=np.array(dd["onmap"],bool)
C=np.full((nM,3),np.nan)
for r,m in enumerate(M.member_id):
    if m in cid and onmap[cid[m]]: C[r]=XYZ[cid[m]]
has_map=~np.isnan(C[:,0]); Cn=C/np.linalg.norm(np.where(has_map[:,None],C,1),axis=1,keepdims=True)
date=pd.to_datetime(E.date).values
# candidate matrix: eligible and not primary sponsor of that EDM
cand=elig.copy()
for j,p in enumerate(prim):
    if p>=0: cand[p,j]=False
nsig=(Sg&cand).sum(0)
# ---- split
def split(cut):
    tr=np.where(date<np.datetime64(cut))[0]; te=np.where((date>=np.datetime64(cut))&(nsig>=5))[0]; return tr,te
cut=pd.Timestamp("2026-03-01"); cuts_tried=[]
while True:
    tr,te=split(cut.date().isoformat()); cuts_tried.append((cut.date().isoformat(),len(te)))
    if len(te)>=100: break
    cut-=pd.DateOffset(months=1)
print("cuts",cuts_tried,"train",len(tr),"test",len(te))
cutd=np.datetime64(cut.date().isoformat())
# ---- feature builders (history = a set of earlier TRAIN EDMs)
sim=EE@EE.T
def nbrs(j,hist,k=25):
    hs=hist[date[hist]<date[j]]
    if len(hs)==0: return hs
    s=sim[j,hs]; return hs[np.argsort(-s)[:k]]
def rate(hist_idx):
    # smoothed own signing rate; eligible-weighted; shrink to party rate with a=5 pseudo-EDMs
    n=elig[:,hist_idx].sum(1).astype(float); c=(Sg[:,hist_idx]&elig[:,hist_idx]).sum(1).astype(float)
    pr={p:(c[party==p].sum()/max(n[party==p].sum(),1)) for p in set(party)}
    prv=np.array([pr[p] for p in party])
    return (c+5*prv)/(n+5), prv
def feats(j,hist,Rt):
    nb=nbrs(j,hist); 
    H=Sg[:,nb].mean(1) if len(nb) else np.zeros(nM)
    if len(nb):
        cnt=Sg[:,nb].sum(); w=Sg[:,nb].sum(1)  # signer occurrences per MP
        ok=has_map[:,None]&Sg[:,nb]; wt=ok.sum(1).astype(float)
        mean=(np.nan_to_num(Cn)*wt[:,None]).sum(0)/max(wt.sum(),1); nm=np.linalg.norm(mean)
        Mm=np.where(has_map,np.nan_to_num(Cn)@(mean/nm),np.nan) if nm>0 else np.full(nM,np.nan)
    else: Mm=np.full(nM,np.nan)
    return H,Mm
def fill(x,mask):
    x=x.copy(); med=np.nanmedian(x[mask]) if np.isfinite(x[mask]).any() else 0; x[~np.isfinite(x)]=med; return x
def centre(x,mask): x=fill(x,mask); return x-x[mask].mean()
def logit(p): p=np.clip(p,1e-3,1-1e-3); return np.log(p/(1-p))
# ---- fit set: later half of TRAIN, history = strictly earlier TRAIN EDMs
trs=tr[np.argsort(date[tr],kind="stable")]; half=trs[len(trs)//2:]
Xs=[];ys=[]
Rcache={}
for j in half:
    hist=trs[date[trs]<date[j]]
    key=len(hist)
    Rt,prv=rate(hist)
    H,Mm=feats(j,hist,Rt)
    msk=cand[:,j]
    f=np.c_[logit(Rt),H,centre(S_pre[:,j],msk),centre(Mm,msk)]
    pos=np.where(msk&Sg[:,j])[0]; neg=np.where(msk&~Sg[:,j])[0]
    neg=np.array(rng.sample(list(neg),min(len(neg),20*len(pos)))) if len(pos) else np.array([],int)
    ix=np.r_[pos,neg].astype(int); Xs.append(f[ix]); ys.append(np.r_[np.ones(len(pos)),np.zeros(len(neg))])
X=np.vstack(Xs); y=np.concatenate(ys); print("fit rows",X.shape,y.mean())
sc=StandardScaler().fit(X); models={}
for name,cols in {"B0+H":[0,1],"B0+H+S":[0,1,2],"B0+H+S+M":[0,1,2,3]}.items():
    models[name]=(cols,LogisticRegression(C=1.0,max_iter=1000).fit(sc.transform(X)[:,cols],y))
coefs={n:dict(zip(["B0","H","S","M"],np.round(m.coef_[0],3).tolist())) for n,(c,m) in models.items()}
# ---- test scores
Rt,prv=rate(tr)
names=["P0","B0","H","S_pre","S_any","M","B0+H","B0+H+S","B0+H+S+M"]
SC={n:{} for n in names}
for j in te:
    msk=cand[:,j]; H,Mm=feats(j,tr,Rt)
    sp=centre(S_pre[:,j],msk); sa=centre(S_any[:,j],msk); mc=centre(Mm,msk)
    SC["P0"][j]=prv; SC["B0"][j]=Rt; SC["H"][j]=H; SC["S_pre"][j]=sp; SC["S_any"][j]=sa; SC["M"][j]=mc
    f=np.c_[logit(Rt),H,sp,mc]; fz=sc.transform(f)
    for n,(cols,m) in models.items(): SC[n][j]=m.decision_function(fz[:,cols])
# ---- metrics
def auc(s,l):
    r=rankdata(s); n1=l.sum(); n0=len(l)-n1
    return (r[l].sum()-n1*(n1+1)/2)/(n1*n0)
cells={n:[] for n in names}; cellmeta=[]; overall={n:[] for n in names}; p20={n:[] for n in names}
for j in te:
    msk=cand[:,j]; lab=Sg[:,j]
    for p in sorted(set(party[msk])):
        mm=msk&(party==p); ns=(mm&lab).sum(); nn=(mm&~lab).sum()
        if ns>=3 and nn>=3:
            cellmeta.append((j,p))
            for n in names: cells[n].append(auc(SC[n][j][mm],lab[mm]))
    ix=np.where(msk)[0]
    for n in names:
        s=SC[n][j][ix]; overall[n].append(auc(s,lab[ix]))
        rk=nrg.random(len(ix)); o=np.lexsort((rk,-s))[:20]; p20[n].append(lab[ix][o].mean())
cellj=np.array([c[0] for c in cellmeta]); cellp=np.array([c[1] for c in cellmeta])
cells={n:np.array(v) for n,v in cells.items()}
def agg(v,sel): return float(v[sel].mean()) if sel.sum() else float("nan")
groups={"all":np.ones(len(cellj),bool),"Lab":cellp=="Lab","Con":cellp=="Con","LD":cellp=="LD"}
tab={g:{n:agg(cells[n],s) for n in names} for g,s in groups.items()}
tab["ncells"]={g:int(s.sum()) for g,s in groups.items()}
tab["overall_auc"]={n:float(np.mean(overall[n])) for n in names}; tab["p20"]={n:float(np.mean(p20[n])) for n in names}
tab["p20_base_rate"]=float(np.mean([Sg[cand[:,j],j].mean() for j in te]))
# ---- bootstrap over TEST EDMs
teA=np.array(te); pos_in=np.searchsorted(teA,cellj)  # te sorted ascending
B=1000; brng=np.random.default_rng(SEED)
W=np.stack([np.bincount(brng.integers(0,len(te),len(te)),minlength=len(te)) for _ in range(B)])  # B x nte
def boot(vec,sel):
    num=W@np.bincount(pos_in[sel],weights=vec[sel],minlength=len(te)); den=W@np.bincount(pos_in[sel],minlength=len(te)); return num/den
bs={}
for g,s in groups.items():
    for n in names: bs[(g,n)]=boot(cells[n],s)
ci=lambda a:[float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]
tab["boot_ci"]={g:{n:ci(bs[(g,n)]) for n in names} for g in groups}
diffs={}
for g in groups:
    for a,b in [("B0+H+S","B0+H"),("B0+H+S+M","B0+H+S"),("B0+H+S","B0"),("B0+H","B0")]:
        d=bs[(g,a)]-bs[(g,b)]; diffs[f"{g}: {a} - {b}"]=dict(point=tab[g][a]-tab[g][b],ci=ci(d),passes=bool(np.percentile(d,2.5)>0))
tab["diffs"]=diffs
# precision@20 bootstrap CI
p20a={n:np.array(p20[n]) for n in names}; tab["p20_ci"]={n:ci(W@p20a[n]/len(te)) for n in names}
tab["p20_diffs"]={f"{a}-{b}":dict(point=float(p20a[a].mean()-p20a[b].mean()),ci=ci(W@(p20a[a]-p20a[b])/len(te))) for a,b in [("B0+H+S","B0+H"),("B0+H+S+M","B0+H+S")]}
# ---- Step 4: per-EDM effect of S
perE={}
for j,k in zip(cellj,range(len(cellj))): perE.setdefault(j,[]).append(k)
dS={j:np.mean([cells["B0+H+S"][k]-cells["B0+H"][k] for k in ks]) for j,ks in perE.items()}
order=sorted(dS,key=dS.get); worst=order[:5]; best=order[::-1][:5]
def show(j):
    t=" ".join(E.title[j].split()[:12]); msk=cand[:,j]; s=SC["B0+H+S"][j]; lab=Sg[:,j]; ix=np.where(msk)[0]; o=ix[np.argsort(-s[ix])]
    f=lambda L:[f"{M.name[i]} ({party[i]})" for i in L[:3]]
    return dict(edm=int(E.id[j]),date=E.date[j],title=t,n_signers=int(nsig[j]),delta_auc=round(float(dS[j]),3),top_ranked_signed=f([i for i in o if lab[i]]),top_ranked_not_signed=f([i for i in o if not lab[i]]))
tab["step4_S_helps"]=[show(j) for j in best]; tab["step4_S_hurts"]=[show(j) for j in worst]
tab["coefs"]=coefs
summ=dict(prep=prep,cut=cut.date().isoformat(),cuts_tried=cuts_tried,n_train_edms=int(len(tr)),n_test_edms=int(len(te)),n_fit_rows=int(len(y)),
          note_fit="fit on later half of TRAIN; features from strictly earlier TRAIN EDMs only; negatives 20:1 sampled; S and M centred per EDM; S uses pre-date passages",
          results=tab)
json.dump(summ,open(RES/"sk20_summary.json","w"),indent=1,default=float)
# ---- chart
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":12})
lab_n={"P0":"P0 party","B0":"B0 activity","H":"H history","S_pre":"S speeches","M":"M map","B0+H":"B0+H","B0+H+S":"B0+H+S","B0+H+S+M":"B0+H+S+M"}
ms=list(lab_n)
fig,ax=plt.subplots(1,3,figsize=(21,6.5))
def bar(a,vals,los,his,title,xl,base):
    x=np.arange(len(ms)); a.bar(x,vals,color="#4c78a8"); a.errorbar(x,vals,yerr=[np.array(vals)-np.array(los),np.array(his)-np.array(vals)],fmt="none",ecolor="k",capsize=3)
    for xi,v in zip(x,vals): a.text(xi,v+0.004,f"{v:.3f}",ha="center",fontsize=10)
    a.axhline(base,ls="--",c="r"); a.set_xticks(x); a.set_xticklabels([lab_n[m] for m in ms],rotation=35,ha="right"); a.set_title(title); a.set_ylabel(xl)
for a,g,t in [(ax[0],"all","Within-party AUC, all parties"),(ax[1],"Lab","Within-party AUC, Labour only")]:
    v=[tab[g][m] for m in ms]; l=[tab["boot_ci"][g][m][0] for m in ms]; hi=[tab["boot_ci"][g][m][1] for m in ms]
    bar(a,v,l,hi,t,"AUC",0.5); a.set_ylim(0.45,max(hi)+0.03)
v=[tab["p20"][m] for m in ms]; bar(ax[2],v,[tab["p20_ci"][m][0] for m in ms],[tab["p20_ci"][m][1] for m in ms],"Precision at 20 per EDM","precision",tab["p20_base_rate"]); ax[2].set_ylim(0,max(v)+0.1)
plt.tight_layout(); plt.savefig(RES/"sk20_chart.png",dpi=130)
print(json.dumps({k:tab[k] for k in ["all","Lab","Con","LD","ncells","overall_auc","p20","diffs","p20_diffs","coefs"]},indent=1,default=float))
