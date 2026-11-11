const assert=require('node:assert/strict'),fs=require('node:fs'),P=require('./visitor_placement.js');
const s=fs.readFileSync(__dirname+'/../docs/index.html','utf8').split('const D=')[1],D=JSON.parse(s.slice(0,s.indexOf(';\nconst M=')));
const compute=a=>P.compute(a,new Set(),D.mps,D.parties,D.topics,'bal');
assert.equal(compute({}),null);
assert.equal(compute({0:1}).topMPs.length,0);
assert.equal(compute({0:1,3:-1}).topMPs.length,0);
assert.equal(compute({0:0,3:0,6:0}).topMPs.length,0);
const keys=D.topics.filter(t=>t.quiz_review).map(t=>t.k);assert.equal(keys.length,5);
const answers=Object.fromEntries(keys.slice(0,4).map(k=>[k,1]));
const comparableStances=keys.slice(0,3).map(k=>[k,1]);
const partialMP={...D.mps[0],st:[...comparableStances.slice(0,2),[keys[2],0]]};
const comparableMP={...D.mps[1],st:[...comparableStances,[keys[3],0]]};
const {topMPs}=P.compute(answers,new Set(),[partialMP,comparableMP],[],D.topics,'bal');
assert.equal(topMPs.length,1);
assert.equal(topMPs[0].m,comparableMP);
assert.deepEqual(topMPs[0].shared.map(x=>x.k),keys.slice(0,3));
assert.equal(topMPs[0].score,0.75);
for(let mask=0;mask<32;mask++){
 const a=Object.fromEntries(keys.map((k,i)=>[k,mask&(1<<i)?1:-1])),u=compute(a);
 assert(u.b.every(Number.isFinite));assert(Math.hypot(...u.b)<=.960001);
 assert(u.topMPs.every(m=>m.nShared>=3));
 assert(u.topMPs.every(m=>m.shared.every(x=>Object.hasOwn(a,x.k)&&x.sc!==0)));
 assert(u.topMPs.every((m,i)=>!i||u.topMPs[i-1].score>=m.score));
}
for(const t of D.topics.filter(t=>t.quiz_review))for(const side of ['lo','hi']){
 const q=t.quiz_review[side+'_source'];assert(q.source_text.includes(q.t));assert(D.mps.some(m=>m.id===q.m));
}
console.log('Passed all 32 answer profiles, minimum evidence, empty/mixed answers, bounds, match ordering and exact source spans.');
