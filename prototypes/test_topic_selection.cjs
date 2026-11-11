const assert=require('node:assert/strict'),fs=require('node:fs'),S=require('./topic_selection.js');
const s=fs.readFileSync(__dirname+'/../docs/index.html','utf8').split('const D=')[1],D=JSON.parse(s.slice(0,s.indexOf(';\nconst M=')));
const before=JSON.stringify(D.mps),model=S.create(D.mps,D.topics),keys=D.topics.filter(t=>t.quiz_review).map(t=>t.k);
assert.equal(model.rank().length,keys.length);
assert(model.rank({1:-1,2:0},[3]).every(r=>![1,2,3].includes(r.k)));
assert.deepEqual(model.rank({1:0},[1]),model.rank({},[1]),'Neutral must not masquerade as directional evidence');
assert.notDeepEqual(model.rank({0:-1}).map(r=>r.score),model.rank({0:1}).map(r=>r.score),'Answer direction must affect selection');
assert.equal(model.rank({},[],[keys.at(-1)])[0].k,keys.at(-1),'Explicit priorities win');
const used=[],answers={};for(let i=0;i<keys.length;i++){const next=model.rank(answers,used)[0];assert(next);used.push(next.k);if(i%3)answers[next.k]=i%2?1:-1;}
assert.equal(new Set(used).size,keys.length);assert.equal(model.rank(answers,used).length,0);
assert.equal(before,JSON.stringify(D.mps),'Selection cannot change MP scores');
assert(D.topics.every(t=>S.areaFor(t.k,D.topics)!=='Other debates'));
const remapped=D.topics.map(t=>({...t,k:t.k+100}));assert.equal(S.areaFor(113,remapped),'Homes & communities','Area mapping must survive key reordering');
assert.equal(S.areaFor(999,[{k:999,title:'New subject'}]),'Other debates');
assert(S.create([{st:[]}],[{k:0,has_claim:true}]).rank().every(r=>Number.isFinite(r.score)));
console.log('Passed: exclusion, neutral/skip handling, answer sensitivity, priorities, full traversal, immutable scores, category remapping and sparse inputs.');
