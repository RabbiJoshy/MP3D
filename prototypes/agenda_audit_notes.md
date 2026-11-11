# MP3D agenda audit — data and inference

Local exploratory audit, 7 November 2026. No commits, publication, credentials or paid model calls. App/lab UI files are owned by the other chat and were not edited here.

## Findings that change the design

1. **Candidate topic IDs do not describe the retrieved passages.** `contest_finder.py` discovers words inside clusters, then searches each word over the entire corpus. “Choices” from fiscal policy retrieves breastfeeding; “developments” from housing retrieves respiratory medicine. Opposite noun scores need not concern the same action. Preserve cluster membership or rerank for topic relevance before writing proposals. The original target screening is useful as a diagnostic, not as verified opposition.
2. **Vote-ranked discovery selects for the parliamentary process.** `contested_topics.py` gives 45/110 topics zero contest score because no division was linked. Only 268 of 617 matched chamber divisions had usable linked debate speech. A 25% debate-topic threshold does not establish that an amendment concerns that topic. Fiscal policy appears, but “fiscal, OBR, finances” also contains procedural oversight: it does not by itself recover tax/spending left–right.
3. **The map is very sparsely observed.** 2,634/15,741 MP–proposal cells are recorded: 83.3% missing among 583 MPs with at least one position. 58.6% of recorded positions come from one debate. The builder uses party-filled missing values to initialise iterative SVD, then party-specific MP priors. The voter has a different prior. Report a party-free sensitivity fit, posterior uncertainty and neighbour stability. The explained-variance share is computed on an iteratively completed table; it is not held-out evidence that three dimensions suffice.
4. **Current “adaptive” selection does not depend on yes versus no.** With fixed importance weights and the same answered/skipped items, Gaussian covariance depends only on which questions were answered. `pickNext()` therefore gives the same sequence regardless of answer direction. A binary ideal-point model has information proportional to p(1−p), which changes after an answer.
5. **“Defining” needs a target and a baseline.** The old mean absolute correlation rewards sibling questions and party sorting. The prototype tests whether one answer improves prediction of *other subjects* beyond party, using held-out MPs. Only 8/27 questions have sufficient co-observed coverage under the stated thresholds. Net zero has an exploratory Brier gain of about 0.010; the fixed-prediction bootstrap interval is roughly [0.003, 0.018]. Do not treat this as a discovery after multiplicity correction, causal evidence or voter validation. Missing coverage is not evidence of no structure.
6. **The annotation checks are helpful but incomplete.** Forty cached snippets have both support and oppose ≥0.5; the existing rule forces a side. Keep ambiguity/conditionality explicitly. The current held-out labels test the classifier, not retrieval recall or the geometry. Test flags do not exclude their predicted positions from the map. Iterated prompts/regexes need a fresh sealed test set. Vote/Wikipedia checks should record dates, matching policy scope, whipping and shared underlying sources.

## Recommended pipeline

**Freeze → discover → retrieve actions → write → retrieve counterevidence → score → validate → select a portfolio → fit.**

- Freeze an as-of date and split by debate families and future time before learning topics. Deduplicate text and repeated bill stages. Separate House, committee, Westminster Hall and speaker-role strata. Include ministerial speech for the institutional agenda, but distinguish government advocacy from personal preference.
- Reuse cheap TF–IDF/SVD clustering first. Compare topic resolutions and seeds; use soft membership or hierarchical clusters to recover economics scattered across tax, wages, welfare and services. Strip procedural language carefully: identity-word exclusion can also discard useful policy words. Retain unclustered/rare action candidates and an exploration reserve.
- Rank *attention* using word share, unique speaker-days, debate breadth and persistence; cap prolific speakers. Do not gate on division presence. Use a Pareto shortlist and sensitivity analysis rather than hiding normative weights in a single score.
- Generate single-action propositions from diverse source passages, with exact quotations, time/jurisdiction, status-quo comparator and an abstain option. **Do not require opposing MPs at this stage:** this encourages partisan pairing and makes sparse/new issues disappear. Find counterevidence after wording is fixed. A writer may return no proposal.
- Retrieve policy-specific passages across the whole corpus with lexical + latent similarity, diverse MPs/debates, and adjacent speech context. Topic relevance and policy relevance are separate. Check actor/negation/conditionality and whether opposition concerns implementation rather than the action. An initial lexical baseline is implemented here; semantic recall and neighbour context remain work.
- Jev screens a small independent sample. Support/oppose/relevance/conditionality are distinct, uncalibrated model outputs until validated. Both-high scores abstain. Aggregate by independent debate, not number of chunks; preserve mixed and changing positions. Require minimum effective support on both sides, rather than balancing the chamber artificially. Use shrinkage/Beta-binomial uncertainty for minority-side prevalence.
- Expand only survivors, with model/prompt/proposal/passage hashes, usage logging and a hard paid-call budget. Keep extraction, calibration, selection and final test partitions separate. Paraphrase/inversion checks and a sample of un-retrieved passages test failures accuracy alone misses.

### Selecting the questions

Keep four quantities separate: stance quality; institutional attention; public salience; incremental information. “Divisive” measures marginal balance (e.g. 4p(1−p)); “defining” is conditional predictive information; a rare stance can define a small group. A highly correlated question may add almost nothing once similar questions are selected.

For a validated candidate set, greedily maximise log det(I + Σ₀ Σⱼ wⱼ pⱼ(1−pⱼ)aⱼaⱼᵀ), with topic-coverage floors, near-duplicate penalties and reserved exploration slots. This is an approximate information objective, not an importance score. Compare the resulting portfolio under multiple salience weights and bootstrap samples. Use separate minimum coverage for neglected but important subjects. No fixed quota should pretend to be naturally discovered by the data.

### Maths worth implementing next

Start with a masked multidimensional logistic ideal-point model: P(yᵢⱼ=1)=sigmoid(aⱼᵀzᵢ−bⱼ). Use observed stances only, regularise item slopes, select dimensions 1–6 on held-out predictions and align rotations for presentation. Estimate uncertainty with a Laplace approximation or bootstrap before expensive Bayesian machinery. A three-dimensional display need not imply a three-dimensional latent world.

Compare global party-free positions with party-plus-individual-residual positions. Stable cross-party coalitions and *where party ceases to explain a position* are portfolio-worthy findings. Do not erase party signal and call the residual the whole political space. Separately model topic mentions using a hurdle or Poisson model with exposure/role controls. Silence is missing-not-at-random; the model cannot recover unspoken beliefs without additional assumptions. Avoid a joint selection model until the data can identify it.

Public salience should influence question coverage and matching preferences, not be interpreted as response precision. An MP attention index measures emphasis, not public perceptions of issue ownership. A survey “best party to handle” item would measure the latter. Voters need shared item wording and tests for differential item functioning; an MP-trained geometry need not transfer unchanged.

Text-based ideal points are a useful baseline, not an automatic upgrade. The repository already contains `tbip_test.py` and `data/quick/tbip_report.txt`; some displayed poles vary in subject vocabulary rather than explicit policy opposition. Compare it against proposal-based ideal points using sealed labels, without rerunning expensive models first.

## Public salience and external checks

[YouGov's tracker](https://yougov.com/en-gb/trackers/the-most-important-issues-facing-the-country) asks for up to three issues from a prompted list. Download a dated wave matching the corpus; do not normalise multi-response percentages as if they were exclusive. Map discovered topics to categories many-to-many, record mapping uncertainty and unmapped mass. The category scheme and portfolio weights are explicit external supervision; topic discovery can still be unsupervised.

[BES panel data](https://www.britishelectionstudy.com/data-objects/panel-study-data/) provide repeated respondent observations and separate open-ended response releases. Downloads require login. Use wave-specific weights and check questionnaire wording before claiming any particular scale is available. Public priorities, policy positions, party competence and personal importance are different variables. No polling values were imported for this audit, so public salience remains unmeasured.

Reserve aligned divisions for external checks if they were not used for selection; otherwise reserve separate divisions/time periods. Aye/no need a proposition-specific polarity, and procedural or bundled votes can be unsuitable. Wikipedia is corroboration only: freeze versions and inspect cited primary sources for dependence on Hansard or the same divisions.

Methods: [Clinton, Jackman & Rivers (2004)](https://www.cambridge.org/core/journals/american-political-science-review/article/abs/statistical-analysis-of-roll-call-data/75DBC6645F85A764AE9E5DBF468AB813); [Vafa, Naidu & Blei (2020)](https://aclanthology.org/2020.acl-main.475/).

## What runs now

```sh
.venv/bin/python prototypes/agenda_audit.py
.venv/bin/python prototypes/agenda_pipeline.py prepare
.venv/bin/python prototypes/agenda_pipeline.py retrieve data/quick/agenda_audit/example_proposals.json --budget-usd 0.01
```

- `agenda_audit.py`: reproducible cached-data statistics, cross-subject prediction audit and standalone `agenda_audit.html`. Historical contaminated target packs are labelled `legacy_target_diagnostic_packs.jsonl` and must not drive generation.
- `agenda_pipeline.py prepare`: 110 topic-specific writer packs, 1,320 passages, using discovered top-three terms, TF–IDF relevance, MMR redundancy reduction and distinct MPs/debates. It reserves passages on/after 1 July 2026. **Topics were learned on the full corpus**, so this is a prototype split, not a sealed evaluation. Refit topics on training data before reporting final performance.
- Three illustrative LLM extractions in `example_proposals.json` test the writer contract. They are selected demonstrations, not the result of a fully autonomous agenda run. The farm-clawback proposal still needs a specified duration. No candidate is yet certified contested.
- `retrieve` checks exact source spans and requires proposal-specific object/action concept groups and prepares up to 24 fresh passages per proposal, excluding the entire writer pack. It validates citations syntactically, not entailment. It produces unexecuted scoring jobs, with a conservative token estimate and planning budget. It cannot spend money. A production executor must use actual provider usage and account for retries/output.
- No full ideal-point fit, fresh human labels, autonomous writer calls, Jev rescoring or public-survey import was performed. These are the next experiment, not claimed results.

At the supplied $0.04/M rate: 100×24×200 tokens costs $0.0192; 20×300×200 costs $0.048. Total illustrative Jev cost $0.0672, excluding writer cost and additional prompt/output/retry tokens. The existing scoring code uses $0.042/M; verify billing rather than assuming either number is current. Writer selection and human checking likely dominate this small budget.

### UI handoff

Standalone local view: http://localhost:8776/agenda_audit.html (anchors `#agenda`, `#structure`, `#pipeline`). The UI chat can add iframe tabs to `lab_template.html` using those URLs; no app/lab/build files were edited by this audit. Preserve the distinction between measured cached results, illustrative candidates and unexecuted requests.
