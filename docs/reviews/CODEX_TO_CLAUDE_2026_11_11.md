# Codex → Claude handover · 11 November 2026

## Scope and ownership

The user asked Codex to finish the existing main app, audit it, provide a local app to try, and leave a handover. On 11 November they explicitly chose: **finish the app; leave Claude’s paid data job alone**. Do not launch duplicate extraction/relabel/scoring jobs. Claude was running relabel work and had created `app2.html`, `app2_template.html`, `build_app2.py`; Codex did not edit those files.

No commit, push or public deployment was requested or performed. The workspace already contains extensive uncommitted work; preserve it.

## What Codex changed

- `prototypes/map_evidence.js`: MP profiles show clear Supports / Opposes / Evidence on both sides, proposal text, then expandable full speech contributions and source-debate links. Missing evidence produces no invented position. Text is escaped and source URLs restricted to HTTPS.
- `prototypes/unified_template.html`: integrated those profiles and an accessible “What the map axes pick up” dialog. Axis bars are computed from the **current main map’s** topic diameters (`div[axis] * r_div`), with period and All Commons / Party Removed context. They describe alignment, not causal influence or prediction. Removed the old hand-written axis labels.
- `prototypes/build_unified_space.py`: canonical main app builder. Embeds completed `fp2_mp_evidence.json` if present, otherwise `fp_mp_evidence.json`. Validates all accepted claims have sources, strips unreachable/dropped source text and records the source file and actual date span. The current build uses the completed pilot unless the full export exists when building.
- `prototypes/visitor_placement.js`: zero and non-finite MP scores no longer count as shared directional answers or dilute agreement. Matches still require three observed directional answers; geometry only breaks ties.
- `prototypes/test_map_evidence.cjs`, `test_visitor_placement.cjs`: regression checks for side/source rendering, escaping, signed axes and zero-score eligibility.
- `prototypes/build_poc_page.py`: older builder defaults to isolated `poc.html` / `poc_lab.html`; it requires explicit `--publish` before overwriting public artifacts. Use `build_unified_space.py` for this main app.

The fuller research pipeline scaffolding also exists from Codex’s earlier work: `fp2_budget.py`, `fp2_dense_run.py`, `fp2_analyse.py`, and the evidence export in `fp2_build_space_page.py`. The budget ledger recorded $0.1813 extraction + $0.37105 merging before Claude took over. This is **not** a reconciliation of subsequent Claude requests or final spend. The dense runner checkpoints; analysis checks candidate coverage. These downstream scripts have not been certified by a completed full run here.

## Rebuild and run

From the repository root:

```sh
.venv/bin/python prototypes/build_unified_space.py
node prototypes/test_map_evidence.cjs
node prototypes/test_visitor_placement.cjs
node prototypes/test_topic_selection.cjs
node prototypes/test_topic_relationships.cjs
```

The builder produces `prototypes/app.html`, `prototypes/unified.html`, `docs/index.html`, `docs/stance.html`, and `prototypes/lab.html`. The current local server serves prototypes at `http://127.0.0.1:8776/app.html`. If absent, start `python3 -m http.server 8776 --bind 127.0.0.1 --directory prototypes`.

## Audit: what is good

The evidence explorer is the strongest product: recognisable MPs, concrete policy proposals, explicit positions, then the actual speech. The user specifically likes this and the explanation of what the axes pick up. Keep these visible and simple. The existing full-screen map, Explore, quiz and topic relationship modes provide a usable shell. Observed-answer match thresholds and honest source disclosures are useful safeguards.

## Audit: readiness and remaining limitations

Ready to try as a local exploratory portfolio demo, **not validated as an accurate political-placement product**.

1. The main map still uses the earlier model: `fit_whitened_3d` replaces missing stance entries with zero and combines stance with speech focus. It is not the proposed observed-only latent model in HANDOVER.md. New fight evidence is integrated into profiles, **not silently substituted into the map or quiz**. The UI states that distinction. Migrating the geometry/questions is separate work requiring an explicit schema and validation.
2. `visitor_placement_evaluation.json` is a reconstruction diagnostic, not held-out accuracy: only 83/643 MPs eligible, four with all five reviewed questions; 48.2% beat the map centre despite a better mean distance. Upstream geometry used all MPs. Do not sell the star as calibrated accuracy.
3. `vote_check.json` reports 85% held-out vote agreement, but links by debate headings, not verified policy/amendment identity. Rebel/free-vote agreement is dominated by assisted dying (538/629 triples). It is a useful diagnostic, not a release gate or proof labels are correct. Validate exact motion identity and direction on an independently checked sample.
4. Pilot fight labels/placements remain machine-derived and can be wrong. Inspect full passages on both sides, especially unusual within-party positions; relabelling alone is not verification. Coverage remains uneven.
5. The app embeds substantial source text and photos. Profile initial load and quiz rendering on mobile before public launch. Consider source-text lazy loading and caching computation after measuring.

## Highest-value next product work

- Make the result about exact shared policies: agreement, difference and missing coverage first; provisional map position second.
- After the five opening questions, offer an evidence-supported next question that increases overlap with candidate MPs. Explain the choice in plain language. Do not add lots of unreviewed questions just because the larger run produces them.
- Use one coherent schema for visitor questions, MP sides, sources and map input; record version and dataset provenance. Integrate the fuller run only after source/polarity checks.
- Test with real users: can they start without explanation, understand what “mixed” versus “skip” does, explain one match, and find its source? Improve those points before adding more map controls.
- Reconcile `app2` with the canonical main app deliberately. Avoid two builders competing for the same published outputs.


## Final quiz changes and verification

The final template simplifies entry to **Start the five questions**, with an optional chooser limited to eligible questions. It hides the visitor marker until a directional answer, hides the quiz guide outside active questions, updates answers in place to retain keyboard focus and expanded speeches, and shows every shared match question with the visitor side, MP side and agreement/difference counts. Every result evidence button opens that MP on the corresponding topic.

Rebuilt successfully against `fp_mp_evidence.json`. Four Node suites passed after rebuilding; production `app.html`, `docs/index.html`, and `docs/stance.html` are identical and contain no unresolved template markers. Browser tested the five-question flow (three directional answers, one mixed, one skipped), results, exact topic evidence handoff, full research speech expansion, and axis dialog. Verified an answer change preserved focus and an open source disclosure. Mobile/touch was not revalidated in this final pass.

A concrete source-quality issue appeared during browser testing: Allison Gardner’s Water Industry Ownership profile also presents a North Staffordshire housing/roads excerpt (2025-09-09) among its cached topic evidence. This comes from the old map-scale source selection, not the new fight-profile renderer. Audit topic relevance and evidence selection before describing every map position as proven. Likewise, broad devolution speeches require checking before inferring support for one specific transfer of planning powers.

The local deliverable is the existing `app.html`, not Claude’s separate `app2.html`. Nothing has been publicly deployed. Preserve both until the user chooses how to reconcile them.
