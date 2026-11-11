# MP3D: instructions for every Claude session

Start with `HANDOVER.md`: goals, definition of done, current pipeline, next step.

## Working with the user
- Short, plain answers. Propose one small step, get a yes, then do it. Show results as charts, not prose.
- Tests cost **cents** and take **minutes**. Ask before anything over about 10¢ or longer than about 10 minutes.
- Never open or print `.env`; scripts read keys from it themselves. Commit or push only when asked.
- No hand-written proposition lists, no cherry-picked topics, no per-topic tuning. Prove methods on a **fresh random topic**.

## Delegate as much as possible (keep the main context small)
The main session plans, decides and talks to the user. Hand everything else to subagents with the Agent tool, and ask each for a **short summary plus file paths, never raw dumps**.
- **Default: Sonnet** (`model: "sonnet"`) for running scripts, data wrangling, writing or editing code, reading files, building charts, hand-checks against passages, and searches.
- **Opus** (`model: "opus"`) only when a step needs careful judgement: method design, tricky debugging, or reviewing a result's logic.
- **Haiku** (`model: "haiku"`) for trivial lookups.
- Give each subagent everything it needs in the prompt (paths, the exact task, output format, cost cap). Subagents start without this conversation.
- Run independent pieces in parallel.
- Long runs go in the background; don't poll them.

## Independent evidence audit (Codex)
Use `python3 scripts/codex_audit.py` (details in `docs/CODEX_AUDITOR.md`). It finds the Codex executable bundled with ChatGPT.app, runs read-only, and uses the user's ChatGPT quota, not Claude tokens.
- Call it **after a meaningful result or before scaling**, not after every edit. **At most two calls per experiment.**
- Send a short delta, the exact evidence-file paths (copy evidence into the repo first) and one question, via stdin. Ask for **≤180 words and at most 3 ranked findings**.
- Also ask it to end with a `DIGEST:` block of ≤100 words, and append that to `.codex-audit/digest.md`.
- Tell the user when an audit starts or fails. Show **Codex's exact reply**, then your decision in one sentence.
- Resolve criticism with evidence. Codex agreeing is not validation. Don't broaden scope or spend more just because it suggests it.
