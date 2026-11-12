// Verify the release artifacts and referential integrity after a local build.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const read = p => fs.readFileSync(path.join(root, p), 'utf8');
const page = read('docs/index.html');
for (const p of ['docs/stance.html', 'prototypes/app_hierarchy.html', 'prototypes/sk13_app_asks.html']) {
  assert.equal(read(p), page, `${p} must match the release`);
}
assert(!page.includes('__DATA__') && !page.includes('/*__HA_'), 'All build placeholders must be resolved');
const d = JSON.parse(page.match(/<script id="d" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const n = d.mps.length;
assert.equal(new Set([...d.mps, ...d.po].map(m => m.id)).size, n + d.po.length, 'Searchable member IDs must be unique');
assert.equal(d.xyz.length, n);
assert.equal(d.onmap.length, n);
assert(d.xyz.every(p => p.length === 3 && p.every(Number.isFinite)));
for (const f of d.fights) {
  assert.equal(f.sc.length, n);
  assert(f.sc.every(v => v === null || Number.isFinite(v)));
}
for (const g of d.groups) assert(g.subjects.every(k => d.subjects[k]), 'Group references must resolve');
for (const [i, claims] of Object.entries(d.mpclaims)) {
  assert(d.mps[i]);
  for (const [k, side, sources] of claims) {
    assert(d.fights[k]);
    assert(['for', 'against', 'both'].includes(side));
    for (const [pid] of sources) assert(d.P[pid], `Missing source ${pid}`);
  }
}
for (const i of Object.keys(d.asks)) assert(+i >= 0 && +i < n + d.po.length, 'Ask profiles must be searchable');
const Q = require('./ha_quiz.js');
assert.equal(Q.place({}, new Set(), d.fights), null);
assert.equal(Q.match({}, new Set(), d.fights, n).length, 0);
console.log(`Release copies identical; ${n + d.po.length} unique profiles, ${d.subjects.length} subjects, ${d.fights.length} fights; source references valid.`);
