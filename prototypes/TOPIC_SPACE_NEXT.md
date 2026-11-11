# Topic space direction — 9 November 2026

## Product direction
Explore Parliament’s conversations, then find yourself within its disagreements. Establish the topic landscape before changing the initial sphere, question pool or order. Do not mistake parliamentary attention for public importance. Missing topics may reflect retrieval/clustering failures; missing speech evidence is not a neutral position.

## Parked for later
Optional typed or dictated interests, interpreted by an LLM into candidate topics and explicit positions for the visitor to confirm. Never infer policy positions from concerns alone. Exact vs related matches must be distinguished; no inferred MP positions for uncovered issues. Not part of the current work.

## Review delivered
Run `.venv/bin/python prototypes/topic_review.py` to generate a separate `topic_review.html` and JSON from existing cached data. Production inputs and map untouched. No paid calls or scoring runs.

- Actual coverage comes from the live generated MP score arrays, not stale source metadata.
- The question diagnostic ranks coverage × side entropy, discounted by maximum squared correlation to previously selected items (minimum 30 co-observed MPs). This is exploratory and in-sample, not information gain or importance. It ignores neutral scores for entropy, retains them in coverage. Missing overlaps must not be interpreted as independence.
- The broader landscape groups the existing 72 non-chatter clusters using cosine average linkage into 12 candidate display groups. The 12 is a UI hypothesis, not a discovered natural number. Two representative excerpts per cluster come from highest existing similarity scores in different date/debate combinations, retaining original positional corpus alignment.
- This hierarchy FAILS a basic usability check: 33 clusters fall in the biggest group. Do not deploy these families or their raw labels.
- Current question wording also needs audit: some alternatives bundle distinct policies or imply false exclusivity (assisted dying vs palliative care; sanctions vs diplomacy). Good numerical coverage does not validate the question.

## Next experiment
Compare fresh local semantic topic discovery with this baseline in a separate experiment directory, never overwriting production data. Include ministers; remove procedural passages and duplicates; preserve passage IDs. Compare multiple resolutions and seeds, inspect coherence using diverse source excerpts, and assess coverage of small/scattered issues. Then screen genuine disagreements; only validated items should affect placement. Evaluate held-out prediction and stability against simple/party baselines before calling a question informative. Do not deploy the current diagnostic order as a new canonical quiz.
