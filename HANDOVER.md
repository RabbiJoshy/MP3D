# Current release handover — 13 November 2026

**What the app is.** A descriptive tool: what MPs said in the Commons, with a link to the speech, on a 3D map that separates parties where they actually disagree. It is **not** a predictive model. Two pre-registered tests (`results/sk20_summary.json`, `results/sk21_summary.json`) showed speeches add little to predicting which motions an MP signs, and the map adds nothing beyond party. Say so if asked.

**Live:** https://rabbijoshy.github.io/MP3D/ (GitHub Pages publishes `main` → `/docs`; pushing `docs/` publishes). Local preview: `http://127.0.0.1:8765/prototypes/map_companion/index.html`, refreshed automatically at the end of every build.

## What changed on 13 November
- **Contested proposals from debates (`research_2026_11_10/sk22_wedge.py`).** One generic rule: every debate with ≥5 Labour and ≥5 non-Labour speakers (260 debates; a second round at ≥3 added 157 more and yielded one extra proposal, so the method is exhausted). DeepSeek writes ≤2 proposals the debate is split on; the strict judge (jev) reads each speaker's 2 best-matching passages; proposals with ≥3 MPs on each side are kept → 32 proposals in `results/sk22_fights.json`. Cost ≈ 63¢.
- **Overrides layer (`results/overrides/`).** All 1,049 sk22 positions were re-read by Gemini Flash (a second machine opinion, not a human): 990 confirmed, 56 dropped, 3 flipped; 13 labels merged, 2 retired. `positions.jsonl` / `labels.jsonl` are append-only and applied in memory by the builder; `--no-overrides` builds without them, `--no-sk22` without the proposals. `audit.json` has agreement by confidence band: 89% at 0.7–0.8, 94% at 0.8–0.9, 99% above 0.9; AGAINST 99%, FOR 92%.
- **Threshold lesson.** For a new, unreviewed run use 0.8/0.2 on FOR verdicts. Do **not** apply it to the reviewed set: tested, it removed 15 of 32 proposals and 146 of 262 AGAINST positions that had already been checked.
- **Axes** are named by the argument pulling hardest at each end (loading × √speakers; one argument names at most one end). "Explore axes" is its own tab.
- **Earlier the same day (Codex/Antigravity):** 1,487 "bridge" positions (`data/quick/research_2026_11_12/bridge/accepted.json`, 95% FOR, only 20 read-checked), confidence-weighted 3D fit, Varimax rotation (changes axis naming, not distances between MPs), Map companion interface.

## Map numbers (rotation-independent; pass marks fixed beforehand in `results/sk22_passmarks.json`)
| | 12 Nov | 13 Nov |
|---|---|---|
| Two-sided arguments / MPs on map | 77 / 552 | 89 / 561 |
| Party share of map variance | 25% | 31% |
| Green–Reform / Labour–Green distance | 3.1 / 1.4 | 3.3 / 1.7 |
| Held-out argument accuracy (mean r) | 0.88 | 0.89 |
| Within-party split votes, AUC | 0.66 | 0.70 |

## Still open
- The 1,487 bridge positions and the older fp2 positions have not had the full second read; the override prompt used for sk22 works on them unchanged.
- Labour is still one cluster and Reform sits with the Conservatives; more of the same data will not change that.
- Phone-width pass on a real device. Hover preview exists only on the topic list (not axis or MP mode).
- Uncommitted edits that predate this work: `README.md`, `build_poc_page.py`, `build_unified_space.py`, two `test_*.cjs` files.

## Build and verify (current)
    .venv/bin/python prototypes/build_hierarchy_app.py     # writes app_hierarchy.html, docs/, map_companion preview
    node prototypes/test_release.cjs

---

# Previous release handover — 12 November 2026

This section supersedes the older app paths below. Consolidation is complete locally; no public deployment has been performed.

- Canonical source: `prototypes/build_hierarchy_app.py` and `prototypes/hierarchy_app_template.html`, promoted from the improved sk13 copy. Shared quiz and relationship modules: `ha_quiz.js`, `ha_relationships.js`.
- The sphere stays. The user asked specifically about retaining it; consolidation leaves its geometry unchanged. A future local-topic redesign must preserve the sphere as the overview.
- Preserve the agreed topic tree and current geometry for this release. The atlas redesign is deferred; no new paid extraction or scoring is needed.
- Tonight’s additions: checked asks with linked excerpts; 82 profile-only MPs searchable by name; collapsed profile sections with remembered state; clickable axis explanations; small-party averages; restored place names; duplicate asks removed.
- Existing saved build counts: 551 map/profile MPs plus 82 profile-only = 633 searchable; 3,591 additional asks; 614 profiles with at least three entries. The asks read-check was 18/20, not a guarantee of correctness.
- Focus remains a separate, unintegrated research result. Show speech share and counts if integrated later; broad subject coverage alone is not discriminating.
- Defer the 44 sk19 candidate causes: their saved read-check found 21/40 real matches.
- Known geometry limitation: topics are compressed to rank one before the global rank-three map. Quiz placement uses a different weighted-direction formula. Consolidating the release does not fix these limitations.
- Pre-consolidation source copies are preserved under `prototypes/archive/pre_consolidation_2026_11_12/`. Research scripts and results remain in place. No commit or push has been performed by this takeover.

## Targeted reading experiment — 12 November

User wants to preserve the current UI and try additional evidence before deciding whether to keep the sphere. They are open to a UMAP/t-SNE-style clustered 2D alternative, but no switch was made.

- Experiment source and reports: `prototypes/research_2026_11_12/bridge_{select,judge,verify,evaluate,plot}.py`, `bridge_report.json`, `bridge_readcheck.json`, `bridge_result.png`.
- Local checkpoints and candidate passages: `data/quick/research_2026_11_12/bridge/` (gitignored).
- Selected 480 missing MP/argument pairs across all 72 existing arguments and 213 MPs; 450 candidates were for MPs with only 1–2 existing positions. Same-debate-heading BGE retrieval, no party selection, no new propositions.
- First judge: 148 directional verdicts, 81 at p(side)>=0.8 and p(other)<=0.2. Random 20-item Codex reading check: 17 supported, 2 partial scope/deadline, 1 wrong bill. Not independent human validation.
- Rechecked all 81 with debate heading and exact-proposal scope rules. Final 60 additions across 49 MPs and 35 arguments passed both checks. The stricter rule independently rejected all 3 reading-check faults at the acceptance threshold; it also rejected 2 initially supported sampled claims. Remaining additions are provisional machine labels.
- Coverage: 516→518 MPs with any scored argument; 281→301 with 3+; 199→216 eligible for a 3-shared-argument neighbour comparison; eligible pairs 591→627.
- Frozen original holdout: correct side unchanged at 79.15% (259 nonzero cells); RMSE 0.75346→0.74974 (260 cells). One new passage excluded from evaluation because its speech also supplied test evidence. No claim of improved prediction or beyond-party structure is supported.
- 561 paid requests total (480 initial +81 verification), 375,504 input tokens, USD 0.015771168 at official $0.042/million. No more paid work queued.
- Added-data model saved locally, not installed in release or experiment UI. Shared JS/Python placement replays all 518 placed MPs within 5e-16. Current app UI, docs pages and sphere remain unchanged.
- Interpretation: targeted reading can fill coverage gaps cheaply, but this batch did not resolve the geometry. Do not scale blindly or claim a clustered 2D view would cure sparse evidence. Inspect selection, graph overlap, argument definition/context and representation before another large batch.

## Build and verify

```sh
.venv/bin/python prototypes/build_hierarchy_app.py
node prototypes/test_release.cjs
node --test prototypes/test_map_evidence.cjs prototypes/test_visitor_placement.cjs prototypes/test_topic_selection.cjs prototypes/test_topic_relationships.cjs
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/docs/index.html`. The canonical builder writes byte-identical `docs/index.html`, `docs/stance.html`, `prototypes/app_hierarchy.html`, and the `sk13_app_asks.html` compatibility copy. The sk13 command delegates; its old template is retired. Legacy unified and POC builders cannot overwrite the release pages.

Browser smoke fixture: `prototypes/test_release.html` loads the actual docs release, scans all 633 profiles, exercises sections/rows/axis explanations and five quiz answers, then Relationships. Saved result: `docs/reviews/RELEASE_CHECK_2026_11_12.json`. Observed: no runtime errors, no profile issues, quiz reaches results. Existing four Node suites pass against their respective legacy modules/artifact. These checks verify integration, not stance truth or map validity.

The local build requires existing gitignored evidence, parquet files, excerpt cache and locally cached BGE model. It is not a clean-clone rebuild. Source backups are local and untracked; research work still needs a deliberate Git checkpoint. Do not stage browser-profile directories or credentials.

## Earlier handover (historical; app paths below are superseded)

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
