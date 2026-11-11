"""Read-only analysis of source data; writes a separate review, never production data."""
import json, re, html
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
R=Path(__file__).resolve().parents[1]
s=(R/'prototypes/app.html').read_text()
d=json.JSONDecoder().raw_decode(s.split('const D=',1)[1])[0]
n=np.load(R/'data/quick/topic_space.npz',allow_pickle=True)
ks=np.flatnonzero(~n['is_chatter']); centres=n['centres'][ks]
passages=pd.read_parquet(R/'data/clean/passages.parquet')
assert len(passages)==len(n['topic']), 'Passage alignment mismatch'
labels=AgglomerativeClustering(n_clusters=12,linkage='average',metric='cosine').fit_predict(centres)
claims=[t for t in d['topics'] if t['has_claim']]
mat=np.full((len(d['mps']),len(claims)),np.nan)
for j,t in enumerate(claims):
 for m in d['mps']:
  v=dict(m['st']).get(t['k'])
  if v is not None: mat[m['i'],j]=v
rows=[]
for j,t in enumerate(claims):
 v=mat[:,j];v=v[np.isfinite(v)];lo=int(sum(v<0));hi=int(sum(v>0));p=hi/max(lo+hi,1)
 entropy=-(p*np.log2(p)+(1-p)*np.log2(1-p)) if 0<p<1 else 0
 rows.append(dict(title=t['title'],k=t['k'],coverage=len(v),low=lo,high=hi,neutral=int(sum(v==0)),balance_entropy=round(float(entropy),3),base=float(len(v)/len(mat)*entropy),question=t['question']))
# Diagnostic shortlist only: coverage * side entropy, discounted by strongest
# squared correlation with earlier items; correlations need 30 shared MPs.
selected=[]; remaining=list(range(len(rows)))
while remaining:
 def value(j):
  redundancy=0
  for k in selected:
   ok=np.isfinite(mat[:,j])&np.isfinite(mat[:,k])
   if sum(ok)>=30 and np.std(mat[ok,j])>0 and np.std(mat[ok,k])>0:
    redundancy=max(redundancy,float(np.corrcoef(mat[ok,j],mat[ok,k])[0,1]**2))
  return rows[j]['base']*(1-redundancy)
 j=max(remaining,key=lambda j:(value(j),-rows[j]['k']));rows[j]['selection_score']=round(value(j),4);selected.append(j);remaining.remove(j)
groups=[]
for g in range(12):
 ids=ks[labels==g];cnt=n['counts'][:,ids];breadth=int(sum(cnt.sum(axis=1)>0))
 topics=[]
 for k in ids:
  ix=np.flatnonzero(n['topic']==k); ix=ix[np.argsort(-n['sim'][ix])]
  examples=[]; seen=set()
  for idx in ix:
   row=passages.iloc[int(idx)]; debate=(str(row['date']),str(row['major']),str(row['minor']))
   if debate in seen: continue
   seen.add(debate);examples.append(dict(text=str(row['text']),date=str(row['date']),debate=str(row['minor']),member=str(row['member_id'])))
   if len(examples)==2:break
  topics.append(dict(examples=examples,id=int(k),terms=[str(x) for x in n['top_terms'][k][:6]],speakers=int(sum(n['counts'][:,k]>0)),words=int(n['counts'][:,k].sum()),scored=str(int(k)) in json.loads((R/'data/quick/claim_topics.json').read_text())))
 topics.sort(key=lambda t:-t['words']);groups.append(dict(label=' / '.join(topics[0]['terms'][:3]),speakers=breadth,words=sum(t['words'] for t in topics),topics=topics))
groups.sort(key=lambda g:-g['words'])
report=dict(method='Exploratory hierarchy of existing 90-cluster centres; 12 display groups is a presentation choice. Not a fresh validated discovery. Ranking = observed coverage × binary side entropy × (1 − largest squared correlation with previously selected questions); minimum overlap 30. Not predictive information gain or public importance.',groups=groups,questions=[rows[j] for j in selected])
(R/'prototypes/topic_review.json').write_text(json.dumps(report,indent=2))
e=html.escape
cards=''.join(f'<details><summary>{e(g["label"])} <small>{len(g["topics"])} clusters · {g["speakers"]} speakers</small></summary>'+''.join(f'<p><b>{e(" · ".join(t["terms"][:3]))}</b><br>{t["speakers"]} speakers · {t["words"]:,} words · {"scored disagreement available" if t["scored"] else "attention only; disagreement not established"}</p>'+''.join('<blockquote>'+e(x['text'][:1000])+('…' if len(x['text'])>1000 else '')+'<small>'+e(x['date']+' · '+x['debate']+' · MP '+x['member'])+'</small></blockquote>' for x in t['examples']) for t in g['topics'])+'</details>' for g in groups)
questions=''.join(f'<tr><td>{i+1}</td><td>{e(r["title"])}</td><td>{r["coverage"]}</td><td>{r["low"]} / {r["high"]}</td><td>{r["selection_score"]}</td></tr>' for i,r in enumerate(report['questions']))
page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MP3D · Topic space review</title><style>body{background:#0b211d;color:#eee8d8;font:17px system-ui;margin:0;padding:5vw;max-width:1100px}h1{font-size:clamp(32px,5vw,60px);max-width:850px}p{line-height:1.6}small{color:#b6c9bf;display:block}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}details{background:#14352d;border:1px solid #36554a;border-radius:14px;padding:20px}summary{cursor:pointer;font-weight:600}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:12px 5px;border-bottom:1px solid #36554a}.note{color:#cfbb83}a{color:#cfbb83}</style><p>MP3D / WORKING REVIEW</p><h1>Explore Parliament’s conversations.</h1><p>A candidate landscape, before we decide which disagreements should place you on the map.</p><p class="note">These are automatic groups of existing topic clusters, with raw term labels. The largest group contains 33 clusters: this candidate hierarchy is too coarse for the public UI. Labels name the largest member, not the whole family. They need source inspection: grouping does not establish coherent opposing positions. Speaker counts overlap across groups.</p><section>'''+cards+'''</section><h2>Which questions deserve to come first?</h2><p>This diagnostic uses actual observed scores, rewards coverage and a mix of positions, then discounts redundant questions. It does not measure public importance or validated information gain. It only compares the existing 20; it cannot discover missing questions.</p><table><tr><th>Order</th><th>Question</th><th>Scored MPs</th><th>− / +</th><th>Diagnostic</th></tr>'''+questions+'''</table><h2>Before replacing the live map</h2><p>Inspect representative speeches and cluster stability; establish disagreement rather than assuming it; evaluate prediction on unseen MPs and debates against party-only and simple baselines. Keep attention separate from position, and leave missing positions unfilled.</p><p><a href="app.html">Return to the current map</a></p>'''
(R/'prototypes/topic_review.html').write_text(page)
print(json.dumps(report['questions'][:6],indent=2));print('Wrote separate topic_review.html and topic_review.json')
