"""sk20 prep: build EDM table, signature matrix, EDM embeddings, S scores (any-date and pre-date). Free/local."""
import json, glob, pathlib, pickle
import numpy as np, pandas as pd
R=pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D"); D=R/"data/quick/research_2026_11_10/sk20"
lst={e["Id"]:e for e in json.load(open(D/"edm/list.json"))}
rows=[]; sig=[]
for f in glob.glob(str(D/"edm/det/*.json")):
    d=json.load(open(f))["Response"]
    rows.append(dict(id=d["Id"],status=d["Status"],amend=d["AmendmentToMotionId"] is not None,date=d["DateTabled"][:10],title=d["Title"] or "",text=d["MotionText"] or "",primary=d["MemberId"],n_raw=len(d["Sponsors"])))
    for s in d["Sponsors"]:
        sig.append((d["Id"],s["MemberId"],bool(s["IsWithdrawn"]),s["SponsoringOrder"]))
E=pd.DataFrame(rows); SG=pd.DataFrame(sig,columns=["edm","member","withdrawn","order"])
stats=dict(listed=len(lst),detail_fetched=len(E),withdrawn_edms=int((E.status!=0).sum()),amendments=int(E.amend.sum()),sig_rows=len(SG),withdrawn_sigs=int(SG.withdrawn.sum()))
E=E[(E.status==0)&(~E.amend)].sort_values(["date","id"]).reset_index(drop=True)
SG=SG[~SG.withdrawn&SG.edm.isin(E.id)]
stats["edms_kept"]=len(E)
# members
M=pd.read_parquet(R/"data/clean/members.parquet").drop_duplicates("member_id").reset_index(drop=True)
spk=M.party_abbr.eq("Spk")|M.other_posts.fillna("").str.contains("Speaker")
stats["speakers_excluded"]=M[spk].name.tolist()
M=M[~spk].reset_index(drop=True)
mid=M.member_id.values; mix={m:i for i,m in enumerate(mid)}
stats["members_project_nonspeaker"]=len(M)
sm=set(SG.member); stats["signer_ids_total"]=len(sm); stats["signer_ids_in_project"]=len(sm&set(mid))
stats["signature_rows_matched_frac"]=float(SG.member.isin(mid).mean())
# eligibility (n_mps x n_edm): in office on date, not a minister/whip on date
dt=pd.to_datetime(E.date)
elig=np.zeros((len(M),len(E)),bool)
for i,r in M.iterrows():
    st=pd.to_datetime(r.start); en=pd.to_datetime(r.end) if pd.notna(r.end) else pd.Timestamp("2100-01-01")
    ok=(dt>=st-pd.Timedelta(days=0))&(dt<=en)
    posts=json.loads(r.gov_posts) if isinstance(r.gov_posts,str) and r.gov_posts else []
    mn=np.zeros(len(E),bool)
    for p in posts:
        s=pd.to_datetime(p["startDate"]) if p.get("startDate") else pd.Timestamp("1900-01-01"); e=pd.to_datetime(p["endDate"]) if p.get("endDate") else pd.Timestamp("2100-01-01")
        mn|=((dt>=s)&(dt<=e)).values
    elig[i]=ok.values&~mn
stats["ministers_excluded"]="yes: MPs with a gov_posts entry active on the EDM tabling date are ineligible (includes whips; PPSs not identifiable)"
eix={e:j for j,e in enumerate(E.id)}
Sg=np.zeros((len(M),len(E)),bool)
for r in SG.itertuples():
    if r.member in mix: Sg[mix[r.member],eix[r.edm]]=True
prim=np.array([mix.get(p,-1) for p in E.primary])
# sponsors who signed while not eligible: ignore (they are not candidates); keep Sg anyway for history
np.savez_compressed(D/"mat.npz",Sg=Sg,elig=elig,prim=prim)
E.to_parquet(D/"edms.parquet"); M.to_parquet(D/"members.parquet")
# embeddings of EDMs
from sentence_transformers import SentenceTransformer
mdl=SentenceTransformer("BAAI/bge-base-en-v1.5",device="cpu",local_files_only=True)
txt=[(t+". "+x)[:2000] for t,x in zip(E.title,E.text)]
EE=mdl.encode(txt,normalize_embeddings=True,batch_size=64,show_progress_bar=False).astype(np.float32)
np.save(D/"edm_emb.npy",EE)
# S scores
P=pd.read_parquet(R/"data/clean/passages.parquet",columns=["passage_id","member_id","date","speech_type","words"])
Emb=np.load(R/"data/embed/passages.npy",mmap_mode="r")
keep=(P.words.astype(int)>=60)&~P.speech_type.fillna("").str.contains("Chair")&P.member_id.isin(mid)
stats["passages_used"]=int(keep.sum()); stats["mps_with_passages"]=int(P[keep].member_id.nunique())
edate=dt.values.astype("datetime64[D]").astype(np.int64)
S_any=np.full((len(M),len(E)),np.nan,np.float32); S_pre=S_any.copy()
pdt=pd.to_datetime(P.date).values.astype("datetime64[D]").astype(np.int64)
K=5
for m,g in P[keep].groupby("member_id"):
    ix=g.index.values; X=np.asarray(Emb[ix]); X=X/np.linalg.norm(X,axis=1,keepdims=True)
    sims=X@EE.T  # n x nE
    i=mix[m]
    k=min(K,len(ix)); S_any[i]=np.sort(np.partition(sims,-k,axis=0)[-k:],axis=0).mean(0) if len(ix)>=1 else np.nan
    pd_=pdt[ix][:,None]; mask=pd_<edate[None,:]
    s2=np.where(mask,sims,-np.inf); top=np.partition(s2,-k,axis=0)[-k:]
    cnt=np.isfinite(top).sum(0); tv=np.where(np.isfinite(top),top,0).sum(0)
    S_pre[i]=np.where(cnt>0,tv/np.maximum(cnt,1),np.nan)
np.savez_compressed(D/"S.npz",S_any=S_any,S_pre=S_pre)
stats["S_pre_nan_frac"]=float(np.isnan(S_pre).mean())
json.dump(stats,open(D/"prep_stats.json","w"),indent=1,default=str); print(json.dumps(stats,indent=1,default=str))
