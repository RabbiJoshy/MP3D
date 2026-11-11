// MP-disjoint diagnostic; scores already extracted upstream, no claim of text-level holdout.
const fs=require('node:fs'),path=require('node:path'),S=require('./topic_selection.js');
const source=fs.readFileSync(path.join(__dirname,'app.html'),'utf8').split('const D=')[1];
const D=JSON.parse(source.slice(0,source.indexOf(';\nconst M=')));
// Preserve the original 20-question experiment; opening-pool review is evaluated separately.
const topics=D.topics.filter(t=>t.has_claim).map(({quiz_review,...t})=>t),keys=topics.map(t=>t.k);
let seed=917;function random(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;}
const shuffled=D.mps.map(m=>({m,r:random()})).sort((a,b)=>a.r-b.r).map(x=>x.m);
const methods=['legacy','coverage','adaptive','random'];const results=Object.fromEntries(methods.map(m=>[m,[]]));const party=[],prior=[];
for(let fold=0;fold<5;fold++){
 const train=shuffled.filter((m,i)=>i%5!==fold),test=shuffled.filter((m,i)=>i%5===fold),model=S.create(train,topics);
 const coverage=keys.slice().sort((a,b)=>train.filter(m=>new Map(m.st).has(b)).length-train.filter(m=>new Map(m.st).has(a)).length||a-b);
 for(const mp of test){
  const st=new Map(mp.st.filter(([,v])=>v!==0)),known=keys.filter(k=>st.has(k));if(known.length<4)continue;
  const target=known[Math.floor(random()*known.length)],truth=st.get(target)>0?1:0;
  const p=model.rates.get(target);prior.push((p-truth)**2);
  const peers=train.filter(m=>m.p===mp.p).map(m=>new Map(m.st).get(target)).filter(v=>v!==undefined&&v!==0);
  const pp=(peers.filter(v=>v>0).length+4*p)/(peers.length+4);party.push((pp-truth)**2);
  const randomOrder=keys.map(k=>({k,r:random()})).sort((a,b)=>a.r-b.r).map(x=>x.k);
  for(const method of methods){
   const answers={},attempted=[target];
   for(let step=0;step<5;step++){
    const order=method==='adaptive'?model.rank(answers,attempted).map(x=>x.k):method==='legacy'?keys:method==='coverage'?coverage:randomOrder;
    const k=order.find(k=>!attempted.includes(k));attempted.push(k);
    if(st.has(k))answers[k]=st.get(k)>0?1:-1;
   }
   const prediction=model.predict(answers,target);
   results[method].push({id:mp.id,target,brier:(prediction-truth)**2,answered:Object.keys(answers).length});
  }
 }
}
const mean=x=>x.reduce((a,b)=>a+b,0)/x.length;
const metrics=Object.fromEntries(methods.map(m=>[m,{n:results[m].length,brier:mean(results[m].map(x=>x.brier)),mean_answers:mean(results[m].map(x=>x.answered))}]));
const delta=results.adaptive.map((r,i)=>r.brier-results.legacy[i].brier),boots=[];
for(let b=0;b<2000;b++)boots.push(mean(delta.map(()=>delta[Math.floor(random()*delta.length)])));
boots.sort((a,b)=>a-b);
const report={protocol:'5 MP-disjoint folds, seed 917. MPs with >=4 non-neutral scores. One randomly chosen observed target per MP is withheld and excluded from all selectors. Five attempted questions; unavailable answers are skipped, not imputed. Same empirical neighbour predictor for all orders. Scores/topics were developed upstream: not debate-disjoint or visitor validation. Random is one seeded order per MP. No parameter tuning on these results.',metrics,baselines:{topic_prior_brier:mean(prior),party_prior_brier:mean(party)},adaptive_minus_legacy:{mean:mean(delta),bootstrap95:[boots[50],boots[1949]]},initial_order:S.create(D.mps,topics).rank().map(r=>({title:D.topics[r.k].title,...r}))};
fs.writeFileSync(path.join(__dirname,'topic_selection_evaluation.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
