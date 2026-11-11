# MP3D handover (current as of 11 November 2026)

**Read this first, then `CLAUDE.md`.**
- App-side handover (what changed in the app, how to rebuild and test): `docs/reviews/CODEX_TO_CLAUDE_2026_11_11.md`.
- Full history of what was tried and why it failed: `AUDIT_BRIEF.md` (long; read only what you need).
- Audit log: `.codex-audit/digest.md` (local, not committed). Auditor details: `docs/CODEX_AUDITOR.md`.

## What the user wants
- **Purpose:** a portfolio piece for a data-analyst job at a political strategy consultancy.
- **Steer on 10 Nov:** it is for a job application. A usable, mostly-right app now beats more validation. Improve iteratively afterwards.
- **What they like most:** click an MP, see clear and obvious positions, click through to the speech. Also the "what the map axes pick up" explanation.
- **Framing:** "Where do you live in this Parliament's space, on Parliament's topics." Narrow scope is accepted for now.
- **Core idea:** per-topic spaces with only MPs who have evidence (nobody is neutral by default); structure beyond party; separate talk (focus) from stance (side).
- **Supervision:** minimal and visible. No hand-written proposition lists, no cherry-picked topics, no per-topic tuning. Prove methods on a fresh random topic.
- **Working style:**
  - Short, plain answers. Propose one small step, get a yes, then do it.
  - Show, don't tell (charts, screenshots).
  - Never open or print `.env`. Commit or push only when asked.
- **Cost rules:** tests cost cents and take minutes. Ask before anything over about 10 cents or 10 minutes.
- **Delegation:** most work goes to Sonnet subagents with explicit cost caps. Evidence audits via `scripts/codex_audit.py` (at most two per experiment; two were used for the strict-judge experiment).

## Why the old unit was dropped
- The old unit was a "contested measure": at least 2 MPs arguing each side of the same named measure, after a strict judge.
- On all 5,226 Education passages it left 3 groups, 2 of them real fights. Too thin.
- Related findings:
  - Hand-check of 38 claims: 82% same measure, 79% right side.
  - The strict FOR/AGAINST/NEITHER judge (`prototypes/research_2026_11_10/jev_sides2.py`, prompts in `results/edu_jev2_test.json`) keeps about two thirds of good claims and almost no wrong ones. Changing its 0.5 cut-off does not help.
  - A pairwise agree/clash test on 30 same-debate pairs found 3 real clashes, with poor model precision (3 of 7). Parked.

## Current pipeline: "fights" plus "dense scoring"
Scripts: `prototypes/research_2026_11_10/` (`fp_*.py` first pass, `fp2_*.py` fuller run). Data: `data/quick/research_2026_11_10/fp/` and `fp2/` (gitignored). Results: `prototypes/research_2026_11_10/results/`.
1. **Select whole debates per topic** (bill, amendment and opposition-day headings), ranked by distinct MPs. (`fp2_select.py`)
2. **Extract MEASURE plus side per speech** with `cloud_extract3.py` (small Qwen on DeepInfra, about 1.45 cents per 1,000 speeches).
3. **Group measures per topic, merge same-fight groups** with pairwise yes/no judge questions: merge if p >= 0.5, or p >= 0.3 and the groups share a debate. (`fp2_group.py`, `fp2_merge.py`)
4. **Relabel each fight** with a larger model (deepseek-ai/DeepSeek-V3.2) into a concrete proposal, a description and what each side wants. (`fp2_relabel.py`)
5. **Cross-topic duplicate merge.** (`fp2_xmerge.py`)
6. **Dense scoring:** for every fight, ask the strict judge about every speech in the fight's own debates (does the speaker want it, oppose it, or neither). An MP's stance is the mean of +1/-1 over their side-taking speeches. Focus is the number of such speeches, which separates talk from stance. (`fp2_dense_cands.py`, `fp2_dense_run.py`, `fp2_analyse.py`)
7. **Free votes** are detected by rule (both largest parties internally split) and treated as a "Conscience" pseudo-topic. Per-topic scales and a topic-level map are in `prototypes/topic_space.py`.

## Fuller run result (`results/fp2_fights.json`, `results/fp2_summary.json`)
- 12,338 new speeches extracted on top of the roughly 10,000-speech first pass.
- 242 fights: 82 two-sided, 42 one MP short, 118 one-sided.
- 577 MPs have a position; 475 in 2 or more topics, 354 in 3 or more, 250 in 4 or more.
- 18 topics have a two-sided fight. None in Environment, Technology or International affairs.
- Cost $1.91 against a $2 cap (dense scoring $0.97). 167 of the smallest fights were not dense-scored, for budget (`results/fp2_skipped_units.json`).
- Total API spend across 10-11 Nov: about $2.50.

## Quality checks
- **Reading checks** of random positions against the passages: 23/30, 17/20 and 26/30 correct (latest: `results/fp2_readcheck.json`, 26 correct, 2 no real position, 1 different measure, 1 wrong side).
- **Vote check** (`vote_check.py`, `results/vote_check.json`):
  - Speech positions match the MP's actual division vote about 85% of the time (85.0% on held-out pairs; 41 of 59 first-pass fights linked to divisions by debate heading).
  - On rebel votes, excluding the assisted-dying free vote, 81% (n=91).
  - Most large disagreements were the check comparing a fight with a division on a different question.
  - It is a diagnostic, not proof.
  - When a speech goes against the party's usual line, the MP also voted against the party only about 48% of the time. So "against most of their party" means how they spoke, not how they voted.
- **The map does not beat the party line.** On held-out topic scores, party average plus individual deviation gets about 84% sign accuracy; cross-validation chose zero extra dimensions, even on the fuller data. So the app is a description (who stands where, with speeches). The claim "the space predicts beyond party" cannot be made yet.
- Fitting the map on individual fights let one big debate (assisted dying) dominate the axes; fitting on topic-level scores fixed that.

## The app
- **The one app:** `prototypes/app.html`, built by `prototypes/build_unified_space.py` from `prototypes/unified_template.html`, plus `map_evidence.js`, `visitor_placement.js`, `topic_selection.js` and `topic_relationships.js`. Four Node test suites: `test_map_evidence.cjs`, `test_visitor_placement.cjs`, `test_topic_selection.cjs`, `test_topic_relationships.cjs`.
- It embeds `results/fp2_mp_evidence.json` (497 MPs, 1,831 positions on the 82 two-sided fights), exported by `fp2_build_space_page.py`.
- The builder also writes `docs/index.html` and `docs/stance.html`. **Pushing those publishes them.**
- The main map and quiz still use the older model (missing treated as zero). The new fights feed the MP profiles only.
- Research pages kept for reference: `prototypes/space.html` (first pass), `prototypes/space_full.html` (fuller run).
- `prototypes/app2.html` (with `build_app2.py`, `app2_template.html`) was an alternative build and is now redundant. Not deleted.
- Rebuild and test commands: see the Codex handover linked above.

## Known flaws still open
- Duplicate fights: Employment Rights Bill x5, Water (Special Measures) Bill x3, private-school VAT x2.
- Yvette Cooper is wrongly shown against repealing the Safety of Rwanda Act (one speech).
- A few vague labels.
- 108 positions have no source link (standing committee speeches).
- 167 small fights are unscored.
- The "education tax" nickname groups have a backwards direction tag (cosmetic).

## Sensible next steps (the user decides)
1. Migrate the quiz and map to the fights data with one schema.
2. Dedupe fights and re-check suspicious within-party positions.
3. Score the skipped fights (roughly $0.50 to $1).
4. Early Day Motion signatures hold-out (free Parliament API) as an external check.
5. Retire app2.

## Dead ends (don't repeat; details in `AUDIT_BRIEF.md`)
- **Speech embeddings alone, or signed claim embeddings:** give subject and tone, not sides.
- **Votes as the source of sides:** they cover too little; use them only as a check.
- **ManifestoBERTa and open CAP classifiers as topic labellers:** too inaccurate. The best CAP model is gated.
- **Off-the-shelf NLI as the stance judge:** too cautious.
- **Choosing the "most evenly split" proposition:** rewards noise.
- **Random 300-per-topic or 5-per-MP samples for the 3D map:** MPs don't overlap, so the axes are noise.
- **"Who backs what" co-occurrence (PPMI + SVD):** mixes focus with stance and treats missing as zero.
- **Old "contested measure" unit with the strict judge:** too few fights (see above).
