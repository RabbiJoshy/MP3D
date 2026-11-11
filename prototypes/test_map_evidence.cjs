const assert = require('node:assert/strict');
const {indexPositions, renderPositions, axisRows, renderAxes} = require('./map_evidence.js');

const fight = {
  label: 'Build <homes>', topic: 'Housing',
  for: [{id: 7, claims: [{p: 'p1', m: 'Build homes', s: 'FOR'}]}],
  against: [{id: 8, claims: [{p: 'p2', m: 'Build homes', s: 'AGAINST'}]}],
  both: [{id: '9', claims: [{p: 'p1', m: 'Build homes', s: 'FOR'}, {p: 'p1', m: 'Build homes', s: 'AGAINST'}]}],
};
const index = indexPositions([fight]);
assert.equal(index.get(7)[0].side, 'for');
assert.equal(index.get(8)[0].side, 'against');
assert.equal(index.get(9)[0].side, 'both');
assert.equal(renderPositions(index.get(99), {}), '');
const positions = renderPositions(index.get(9), {
  p1: {date: '2026-07-01', debate: 'Housing debate', ex: 'An excerpt', full: 'Complete speech <script>', url: 'javascript:alert(1)'},
});
assert.match(positions, /Evidence on both sides/);
assert.match(positions, /Build &lt;homes&gt;/);
assert.match(positions, /Complete speech &lt;script&gt;/);
assert.match(positions, /Supports — Build homes/);
assert.match(positions, /Opposes — Build homes/);
assert.doesNotMatch(positions, /An excerpt|javascript:/);
assert.equal(positions.match(/class="position-speech"/g).length, 1);
assert.match(renderPositions(index.get(8), {}), /No speech text/);

const topics = [
  {k: 0, title: 'Housing', pole_hi: 'Build homes', has_claim: true, diam_bal: {div: [-0.9, 0.1, 0], r_div: 0.8}, diam_win: {div: [1, 0, 0], r_div: 0.2}},
  {k: 1, title: 'Transport', has_claim: true, diam_bal: {div: [0.4, 0.5, 0], r_div: 0.75}, diam_win: {div: [-1, 0, 0], r_div: 0.8}},
  {k: 2, title: 'Focus only', has_claim: false, diam_bal: {div: [1, 0, 0], r_div: 1}},
  {k: 3, title: 'Missing', has_claim: true},
];
const bal = axisRows(topics, 'diam_bal', 0);
assert.deepEqual(bal.map(r => r.topic.k), [0, 1]);
assert.ok(Math.abs(bal[0].value + 0.72) < 1e-10);
assert.ok(Math.abs(bal[1].value - 0.30) < 1e-10);
assert.deepEqual(axisRows(topics, 'diam_win', 0).map(r => r.topic.k), [1, 0]);
const charts = renderAxes(topics, 'diam_bal', '2024–26 Parliament');
assert.match(charts, /All Commons · 2024–26 Parliament/);
assert.match(charts, /−0\.72/);
assert.match(charts, /\+0\.30/);
assert.match(charts, /Positive pole · Build homes/);
assert.doesNotMatch(charts, /Focus only|Missing/);
console.log('map evidence and axis checks passed');
