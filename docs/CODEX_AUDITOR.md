# Claude Code → Codex auditor

From the MP3D root:

```sh
python3 scripts/codex_audit.py <<'AUDIT'
Audit this result against the agreed definition of done.
Changed: [short delta]. Evidence: [specific file paths and relevant sections].
Question: [one decision]. Do not rerun experiments.
AUDIT
```

The wrapper finds `codex` on PATH or falls back to the executable bundled inside
`/Applications/ChatGPT.app`. No separate CLI installation is needed on this machine.
Set `CODEX_AUDIT_BIN` to an absolute executable path to override discovery.

The wrapper uses the existing Codex ChatGPT login and configured model, with medium reasoning,
read-only shell sandbox and no approval prompts. It ignores user-level Codex config for the
run (including MCP integrations), retaining only the model preference. Repository instructions
may still load. No API key is required. Normal Codex usage limits still apply.

Each call resumes this project's auditor by explicit session ID. The initial prompt carries
MP3D requirements and prior audit findings; subsequent calls send only the new request. Prior
context still consumes context tokens (and may benefit from caching); persistence is not free.
No automatic loop, watcher, or per-edit calls are installed. Only ask for an audit when it can
change the next decision. One initial audit and at most one follow-up per experiment.

Output: final audit on stdout; status and token usage on stderr. Requests, raw events, responses
and usage live in `.codex-audit/runs/`. `.codex-audit/latest.md` is the latest response and
`.codex-audit/transcript.md` contains all requests and exact final replies. These files are
locally excluded from git. They can contain project evidence; do not put credentials in requests.

`python3 scripts/codex_audit.py --status` shows the session and paths without spending tokens.
`--new` starts a fresh session while retaining old logs. `--request PATH` reads a request file.
`--timeout 240` caps elapsed time, not tokens or spend. `--model NAME` explicitly changes model.
Concurrent calls are rejected. Failures return a nonzero exit code and preserve logs; there is
no silent retry or automatic permission escalation.

## Paste into the existing Claude Code chat

Use `python3 scripts/codex_audit.py` from the MP3D root as an independent read-only auditor.
You are authorized to send it project evidence and receive its feedback. Keep me in this chat.
After a meaningful experiment, or before scaling one, send a short delta, exact evidence-file
paths and one question via stdin. No entire chat transcripts or data dumps. Ask for <=180 words
and at most three ranked findings. Do not audit every edit. Maximum two calls per experiment;
stop sooner when the decision is clear. Show the exact Codex response, then your decision in
one sentence. Resolve criticism using evidence; do not treat Codex agreement as validation.
Do not run larger experiments or broaden scope just because Codex suggests it. Preserve my
existing cost limits and approval preferences. Tell me when an audit starts and fails. I can
inspect `.codex-audit/transcript.md` independently. Start by acknowledging this setup; do not
run another audit until there is a concrete result or decision to review.
