const fs=require('node:fs'),P=require('./visitor_placement.js');
const text=fs.readFileSync(__dirname+'/app.html','utf8').split('const D=')[1],D=JSON.parse(text.slice(0,text.indexOf(';\nconst M=')));
const keys=D.topics.filter(t=>t.quiz_review).map(t=>t.k),distance=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i]));
const mean=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:null;
const examples=[];let tooFew=0;
for(const m of D.mps){
 const st=new Map(m.st),answers=Object.fromEntries(keys.filter(k=>st.has(k)&&st.get(k)!==0).map(k=>[k,Math.sign(st.get(k))]));
 if(Object.keys(answers).length<3){tooFew++;continue;}
 // Exclude the target MP from neighbours. Geometry itself was fitted upstream on all MPs.
 const point=P.compute(answers,new Set(),D.mps.filter(x=>x.i!==m.i),D.parties,D.topics,'bal');
 examples.push({mp:m.n,answers:Object.keys(answers).length,error:distance(point.b,m.b),origin_error:distance([0,0,0],m.b),matches:point.topMPs.length,top_shared:point.topMPs[0]?.nShared??0});
}
const combinations=[];
for(let mask=0;mask<2**keys.length;mask++){
 const a=Object.fromEntries(keys.map((k,i)=>[k,mask&(1<<i)?1:-1]));
 const p=P.compute(a,new Set(),D.mps,D.parties,D.topics,'bal');
 combinations.push({answers:a,radius:Math.hypot(...p.b),matches:p.topMPs.length,min_shared:Math.min(...p.topMPs.map(x=>x.nShared)),best_score:p.topMPs[0]?.score??null});
}
const report={scope:'Actual UI placement code, with only available non-neutral answers on the five excerpt-reviewed questions. Target MP excluded from neighbour candidates, but upstream geometry used all MPs: reconstruction diagnostic, NOT held-out validation or visitor accuracy. All scores remain unaudited beyond the ten illustrative excerpts.',reviewed_topics:keys.map(k=>D.topics[k].title),eligible_mps:examples.length,excluded_for_fewer_than_three:tooFew,with_all_five:examples.filter(x=>x.answers===5).length,mean_star_distance:mean(examples.map(x=>x.error)),mean_origin_distance:mean(examples.map(x=>x.origin_error)),fraction_better_than_origin:mean(examples.map(x=>Number(x.error<x.origin_error))),combinations_tested:combinations.length,min_shared_in_shown_matches:Math.min(...combinations.map(x=>x.min_shared)),finite_bounded_stars:combinations.every(x=>Number.isFinite(x.radius)&&x.radius<=.960001),examples,combinations};
fs.writeFileSync(__dirname+'/visitor_placement_evaluation.json',JSON.stringify(report,null,2));console.log(JSON.stringify({...report,examples:undefined,combinations:undefined},null,2));
