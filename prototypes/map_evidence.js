const MP3DMapEvidence = (() => {
  const esc = value => String(value ?? '').replace(/[&<>"]/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;',
  }[c]));
  const sides = {for: 'Supports', against: 'Opposes', both: 'Evidence on both sides'};
  const claimSides = {FOR: 'Supports', AGAINST: 'Opposes'};

  function indexPositions(fights = []) {
    const byMP = new Map();
    for (const fight of fights) {
      for (const side of Object.keys(sides)) {
        for (const mp of fight[side] || []) {
          const id = Number(mp.id);
          if (!byMP.has(id)) byMP.set(id, []);
          byMP.get(id).push({label: fight.label, topic: fight.topic, side, claims: mp.claims});
        }
      }
    }
    return byMP;
  }

  function sourceLink(value) {
    if (typeof value !== 'string' || !/^https:\/\//i.test(value)) return '';
    return `<a href="${esc(value)}" target="_blank" rel="noopener noreferrer">Read the source debate ↗</a>`;
  }

  function speechSources(claims, passages) {
    const byPassage = new Map();
    for (const claim of claims || []) {
      if (!Object.hasOwn(passages, claim.p)) continue;
      if (!byPassage.has(claim.p)) byPassage.set(claim.p, new Set());
      const prefix = claimSides[claim.s];
      byPassage.get(claim.p).add(prefix ? `${prefix} — ${claim.m}` : claim.m);
    }
    if (!byPassage.size) return '<p class="meta">No speech text is available for this position.</p>';
    return [...byPassage].map(([id, statements]) => {
      const p = passages[id];
      const heading = [p.date, p.debate].filter(Boolean).join(' · ') || 'Commons speech';
      return `<details class="position-speech"><summary>${esc(heading)}</summary>
        <ul>${[...statements].map(s => `<li>${esc(s)}</li>`).join('')}</ul>
        <blockquote class="position-text">${esc(p.full || p.ex)}</blockquote>
        ${sourceLink(p.url)}</details>`;
    }).join('');
  }

  function renderPositions(positions = [], passages = {}, period) {
    if (!positions.length) return '';
    return `<section class="debate-positions" aria-label="Positions in debates">
      <h3>Positions in debates</h3>
      <p class="meta">These positions come from sampled Commons speeches${period ? ` (${esc(period)})` : ''}. The map and quiz use separate speech scales.</p>
      ${positions.map(({label, topic, side, claims}) => `<details class="debate-position">
        <summary><span class="position-side ${side}">${sides[side]}</span><strong>${esc(label)}</strong><small>${esc(topic)}</small></summary>
        ${speechSources(claims, passages)}
      </details>`).join('')}
    </section>`;
  }

  function axisRows(topics, diameterKey, axis) {
    return topics.filter(t => t.has_claim).map(topic => {
      const diameter = topic[diameterKey];
      return {topic, value: diameter?.div?.[axis] * diameter?.r_div};
    }).filter(({value}) => Number.isFinite(value) && value !== 0)
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value) || a.topic.title.localeCompare(b.topic.title));
  }

  function renderAxes(topics, diameterKey, period, activeTopicKey) {
    const mode = diameterKey === 'diam_win' ? 'Party Removed' : 'All Commons';
    return `<p class="axis-context">${mode} · ${esc(period)}</p>
      <p class="meta">Each bar shows where a topic’s positive pole points along an underlying map axis, before rotation or topic pull. The length combines that direction with the fit of the topic in 3D. These are descriptive alignments of the fitted map. They establish neither cause nor predictive accuracy.</p>
      <div class="axis-charts">${['X', 'Y', 'Z'].map((axis, i) => {
        const rows = axisRows(topics, diameterKey, i);
        const visible = rows.slice(0, 5);
        const active = rows.find(({topic}) => topic.k === activeTopicKey);
        if (active && !visible.includes(active)) visible.push(active);
        return `<section class="axis-chart" aria-label="${axis} axis">
          <h3>${axis} axis</h3>
          <div class="axis-ticks" aria-hidden="true"><span>−1</span><span>0</span><span>+1</span></div>
          ${visible.map(({topic, value}) => {
            const width = 50 * Math.min(1, Math.abs(value));
            const start = value < 0 ? 50 - width : 50;
            const signed = `${value < 0 ? '−' : '+'}${Math.abs(value).toFixed(2)}`;
            return `<button class="axis-row" type="button" data-axis-topic="${topic.k}">
              <span class="axis-row-label"><strong>${esc(topic.title)}</strong><span>${signed}</span></span>
              <span class="axis-pole">Positive pole · ${esc(topic.pole_hi)}</span>
              <span class="axis-track" aria-hidden="true"><i class="${value < 0 ? 'negative' : 'positive'}" style="left:${start}%;width:${width}%"></i></span>
            </button>`;
          }).join('') || '<p class="meta">No fitted topics for this axis.</p>'}
        </section>`;
      }).join('')}</div>`;
  }

  return {indexPositions, renderPositions, axisRows, renderAxes};
})();
if (typeof module === 'object' && module.exports) module.exports = MP3DMapEvidence;
