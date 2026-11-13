/* Quiz logic for the hierarchy app (pure functions, no DOM). Adapted from topic_selection.js and visitor_placement.js.
   Questions are merged two-sided fights. Their subject (and group) come from the app's own hierarchy, not a hand-written list.
   Matching uses only observed, non-zero MP stances; no imputation. */
const HAQuiz=(()=>{
 const VERBS=new Set('pass support extend increase establish require remove introduce implement impose abolish amend exclude appoint enable transfer mandate reduce end raise lift return negotiate criminalise disapply repeal restrict devolve insert ban cut scrap give allow make set create adopt reject block delay keep protect provide publish reform replace stop strengthen limit legalise legalize cap freeze tax fund'.split(' '));
 /* "Should Parliament pass the X Bill?" when the fight label starts with a plain verb; otherwise a neutral fallback. */
 function question(label){
  const l=String(label||'').trim().replace(/[.?]+$/,''),w=(l.split(/\s+/)[0]||'').toLowerCase();
  return VERBS.has(w)?'Should Parliament '+l.charAt(0).toLowerCase()+l.slice(1)+'?':'Do you support this: '+l+'?';
 }
 const sideOf=v=>v>0?1:0;
 const entropy=p=>p<=0||p>=1?0:-p*Math.log2(p)-(1-p)*Math.log2(1-p);
 /* rows[i] = Map(fightIndex -> signed stance, non-zero only); qs = [{k,subj,grp}] ; minShared for the redundancy correlation */
 function create(rows,qs,minShared){
  minShared=minShared||15;
  const info=new Map(qs.map(q=>[q.k,q])),keys=qs.map(q=>q.k);
  const bin=rows.map(r=>new Map([...r].map(([k,v])=>[k,sideOf(v)])));
  const rates=new Map(keys.map(k=>{const obs=bin.filter(r=>r.has(k));return[k,(1+obs.reduce((s,r)=>s+r.get(k),0))/(2+obs.length)]}));
  const corr=new Map();
  for(const k of keys)for(const j of keys)if(j<k){
   const pr=bin.filter(r=>r.has(k)&&r.has(j)),n=pr.length;let r2=0;
   if(n>=minShared){const a=pr.map(r=>r.get(k)),b=pr.map(r=>r.get(j)),ma=a.reduce((s,v)=>s+v,0)/n,mb=b.reduce((s,v)=>s+v,0)/n;
    const va=a.reduce((s,v)=>s+(v-ma)**2,0),vb=b.reduce((s,v)=>s+(v-mb)**2,0);
    if(va*vb>0)r2=(a.reduce((s,v,i)=>s+(v-ma)*(b[i]-mb),0)**2)/(va*vb)}
   corr.set(j+':'+k,r2)}
  function weights(answers){
   let w=bin.map(()=>1);
   for(const[str,a]of Object.entries(answers)){
    const k=+str;if(a===0||!rates.has(k))continue;
    const side=a>0?1:0,p=rates.get(k),marginal=side?(.2+.6*p):(.8-.6*p);
    w=w.map((v,i)=>v*(bin[i].has(k)?(bin[i].get(k)===side?.8:.2):marginal))}
   const sum=w.reduce((s,v)=>s+v,0);return w.map(v=>sum?v/sum:1/bin.length)}
  /* opts: excluded (asked already), priorities (subject names the visitor chose), avoidSubj (Set of subjects already used in the opening questions) */
  function rank(answers,opts){
   opts=opts||{};const excl=new Set(opts.excluded||[]),pri=new Set(opts.priorities||[]),avoid=opts.avoidSubj||new Set();
   const w=weights(answers||{}),answered=Object.entries(answers||{}).filter(([,v])=>v!==0).map(([k])=>+k);
   let cand=keys.filter(k=>!excl.has(k));
   const fresh=cand.filter(k=>!avoid.has(info.get(k).subj));if(fresh.length)cand=fresh;       // opening questions come from different subjects
   const picked=cand.filter(k=>pri.has(info.get(k).subj));if(pri.size&&picked.length)cand=picked;
   return cand.map(k=>{
    let mass=0,yes=0,n=0;bin.forEach((r,i)=>{if(r.has(k)){mass+=w[i];yes+=w[i]*r.get(k);n++}});
    const p=mass?yes/mass:rates.get(k);
    const red=Math.max(0,...answered.map(j=>corr.get(Math.min(j,k)+':'+Math.max(j,k))||0));
    const sameSubj=answered.some(j=>info.get(j)&&info.get(j).subj===info.get(k).subj),sameGrp=answered.some(j=>info.get(j)&&info.get(j).grp!=null&&info.get(j).grp===info.get(k).grp);
    const repeat=(sameSubj?.65:1)*(sameGrp?.85:1);
    const qinfo=info.get(k),sal=Math.log2(1+n)*(0.35+(qinfo&&qinfo.r!=null?qinfo.r:0.5));
    const opener=(answered.length<5&&qinfo&&qinfo.subj==='Conscience (free votes)')?.35:1;
    return{k,score:mass*entropy(p)*(1-.75*red)*repeat*sal*opener,coverage:n}}).sort((a,b)=>b.score-a.score||a.k-b.k)}
  return{rank,rates}
 }
 /* Visitor position: when `model` ({v, b, ridge}) is supplied, uses the exact same 3D ridge projection
    as MPs: solve (V_obs^T W V_obs + ridge * I) p = V_obs^T W (y_obs - b_obs).
    Mixed answers (a === 0) pull toward 0 - b_obs with weight 1. Returns null with no answered fights. */
 function solve3(A,rhs){
  const M=A.map((r,i)=>[r[0],r[1],r[2],rhs[i]]);
  for(let col=0;col<3;col++){
   let piv=col;for(let r=col+1;r<3;r++)if(Math.abs(M[r][col])>Math.abs(M[piv][col]))piv=r;
   if(Math.abs(M[piv][col])<1e-12)return null;
   if(piv!==col){const t=M[col];M[col]=M[piv];M[piv]=t}
   const d=M[col][col];for(let c=col;c<4;c++)M[col][c]/=d;
   for(let r=0;r<3;r++)if(r!==col){const f=M[r][col];for(let c=col;c<4;c++)M[r][c]-=f*M[col][c]}
  }
  return[M[0][3],M[1][3],M[2][3]];
 }
 function place(ans,star,fights,model){
  if(model&&model.v&&model.b){
   const A=[[model.ridge,0,0],[0,model.ridge,0],[0,0,model.ridge]],rhs=[0,0,0];let n=0;
   for(const[ks,a]of Object.entries(ans)){
    const k=+ks,v=model.v[k];if(!v)continue;
    const wt=star&&star.has(k)?1.8:1,y=a-model.b[k];
    for(let i=0;i<3;i++){rhs[i]+=wt*v[i]*y;for(let j=0;j<3;j++)A[i][j]+=wt*v[i]*v[j]}
    n++;
   }
   if(!n)return null;
   let p=solve3(A,rhs);if(!p)return null;
   const nm=Math.hypot(...p);if(nm>1.5)p=p.map(v=>v*1.5/nm);
   return{p,n};
  }
  const b=[0,0,0];let wSum=0,n=0;
  for(const[ks,a]of Object.entries(ans)){const k=+ks,f=fights[k],wt=star&&star.has(k)?1.8:1;
   if(a===0){wSum+=wt*.35;continue}
   if(!f||!f.u){continue}
   const r=Math.max(.15,f.r||0);for(let j=0;j<3;j++)b[j]+=a*wt*f.u[j]*r;wSum+=wt*.65;n++}
  if(!n||wSum<=0)return null;
  let p=b.map(v=>v/wSum);const nm=Math.hypot(...p);if(nm>.96)p=p.map(v=>v*.96/nm);
  return{p,n}
 }
 /* Match the visitor to every MP on shared answered fights (MP stance sign; zero or missing never counts). Needs >=3 shared directional answers. */
 function match(ans,star,fights,nMps,dist){
  const out=[];
  for(let i=0;i<nMps;i++){
   let agree=0,tot=0,hit=0;const shared=[];
   for(const[ks,a]of Object.entries(ans)){if(a===0)continue;const k=+ks,sc=fights[k]&&fights[k].sc[i];
    if(sc==null||!Number.isFinite(sc)||sc===0)continue;
    const wt=star&&star.has(k)?1.6:1;tot+=wt;agree+=wt*a*Math.max(-1,Math.min(1,sc));
    const ok=a*sc>0;if(ok)hit++;shared.push({k,sc,ok})}
   if(shared.length>=3)out.push({i,score:agree/(tot+1),nShared:shared.length,hit,shared,dist:dist?dist(i):0})}
  return out.sort((a,b)=>b.score-a.score||b.nShared-a.nShared||a.dist-b.dist||a.i-b.i)
 }
 /* Party summary over MPs that have >=3 comparable answers: share of comparable answers that agree. */
 function parties(matches,partyOf,minMPs){
  const g={};matches.forEach(m=>{const p=partyOf(m.i),a=g[p]||(g[p]={p,mps:0,hit:0,tot:0});a.mps++;a.hit+=m.hit;a.tot+=m.nShared});
  return Object.values(g).filter(a=>a.mps>=(minMPs||3)).map(a=>({...a,pct:Math.round(100*a.hit/a.tot)})).sort((a,b)=>b.pct-a.pct||b.mps-a.mps)
 }
 /* Projected stances across propositions given 3D coordinate p.
    Uses linear factor model y_hat = v_k . p + b_k or directional projection. */
 function predictStances(p,fights,model,answeredMap){
  if(!p||!fights)return[];
  const answered=answeredMap||{};
  const out=[];
  for(let k=0;k<fights.length;k++){
   const f=fights[k];if(!f)continue;
   let score=0;
   if(model&&model.v&&model.b&&model.v[k]){
    const vk=model.v[k],bk=model.b[k];
    score=vk[0]*p[0]+vk[1]*p[1]+vk[2]*p[2]+bk;
   }else if(f.u){
    score=(p[0]*f.u[0]+p[1]*f.u[1]+p[2]*f.u[2])*(f.r||1);
   }
   const isAns=answered[k]!==undefined;
   const ans=answered[k];
   const conf=Math.min(1,Math.abs(score));
   const dir=score>=0.2?'for':(score<=-0.2?'against':'split');
   out.push({k,f,label:f.label,subj:f.subj,score,conf,dir,answered:isAns,ans});
  }
  return out;
 }
 return{question,create,place,match,parties,predictStances}
})();
if(typeof module!=='undefined')module.exports=HAQuiz;
