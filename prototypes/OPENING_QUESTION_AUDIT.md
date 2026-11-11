# Opening questions and placement audit — 10 November 2026

## Delivered
Five presentation-reviewed questions: planning, assisted dying, Winter Fuel Payments, passenger rail ownership and water ownership. Ten exact excerpts have identifiable passage IDs, speaker IDs, dates and full cached context in `opening_question_review.json`. Regenerate and verify against local cleaned passages with `review_opening_questions.py`. No paid calls, original stance files, MP coordinates or model fitting changed.

This is a narrow wording/excerpt review, not validation of all underlying MP scores or verification against original Hansard. Cleaned corpus text sometimes omits names; the displayed extracts avoid those gaps where possible. Exact source text remains available on demand. Navigation retains all 72 topics; only the five reviewed questions currently accept visitor answers. Others remain available for evidence exploration.

## Decisions
- Assisted dying: question refers to the November 2024 proposal; remove the false choice between assisted dying and palliative care. Supporting excerpt explicitly supports both; opposition excerpt criticises the proposal's safeguards. These bill-specific arguments do not establish views on all possible assisted-dying legislation.
- Winter fuel: refer to the restriction debated in 2024, rather than claim opponents reject every means test or describe today's policy.
- Rail: specify passenger services, rather than the entire rail industry. Both excerpts explicitly address private provision / renationalisation.
- Water: both illustrative excerpts explicitly address ownership; replace a generic pollution/regulation passage with explicit opposition to nationalisation. Keep alternative models and mixed answers available.
- Planning: present a difference of emphasis; support for planning reform does not establish opposition to every green-belt protection.
- Gaza sanctions vs diplomacy and asylum safe routes vs controls: not admitted to opening pool because those alternatives can coexist and shown snippets do not establish opposition to the other option.
- Two-child cap: generic fiscal restraint is not proof of support for retaining the cap.
- North Sea: continuing existing extraction and refusing new expansion can coexist.
- Bus control: both original poles support forms of local control.
- Farm / inheritance tax: overlapping scales, bundled policy comparisons and weak illustrative extracts need a separate review.
- Remaining topics: not yet reviewed; exclusion is not a finding that they are wrong.

## Match safeguards
The UI and diagnostic import the same `visitor_placement.js`. Require three shared directional answers. Missing scores and mixed visitor answers do not count as agreement. Match ranking uses weighted signed agreement, shrunk by one unit of evidence; map distance only breaks ties. Cap score magnitude at one for matching. No party priors or imputations. Do not show party-distance rankings in the visitor results. Keep existing star geometry but label it provisional.

## Measured limitations
`evaluate_visitor_placement.cjs` uses available answers on these five topics; 83 MPs have at least three non-neutral scores, only four have all five. It excludes the target MP from neighbour candidates, but geometry was fitted upstream on all MPs, so this is reconstruction, not held-out validation.

Mean distance from reconstructed star to recorded MP position: 0.507 vs 0.558 for the origin. Only 48.2% improve over the origin. This does NOT validate accurate visitor placement. All 32 complete directional profiles produce finite bounded stars and match lists with at least three shared answers. That checks behaviour, not political truth.

Next priority is validating/upgrading the projection and auditing the matched MP scores, using independent source labels and held-out people/debates before claiming accuracy. Do not present the earlier selection-only Brier improvement as validation of this five-question projection. The old selector evaluation explicitly retains its original 20-topic candidate pool; its report is separate.
