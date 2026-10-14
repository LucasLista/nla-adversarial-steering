# Unread experiment content

To keep the agent's context independent of this safety evaluation, agents must
not open, display, read, search, watch, summarize, interpret, or inspect:

- `prompts.json` and `editor_prompts.json` values (including `bio_question` and `poetic`).
- Anything under `runs/experiments/*/inputs/`.
- Model responses, NLA explanations, editor proposals, decoded generated tokens,
  activation vectors, and request/response payloads in `runs/`, `logs/`, or `examples/`.
- Copies of those inputs or outputs on HPC, in other checkouts, or in browser UI.

Do not include these files in recursive searches, diffs, terminal previews,
review tool inputs, or agent messages. Only the human user reviews their content.
Explicit subsequent human authorization is required for content access.

Application code may load inputs at the user's request. Agents may run scripts
that report only JSON structure, field types, exact requested ID matches,
scheduler/process state, token counts, ended flags, completion flags, or file
existence/hashes. Do not decode token IDs or print prompt/response values.

The user authorizes blind copying, staging, committing, and pushing model
response logs to `main`. Use `publish_run.py` to copy logs without inspecting
content. This exception permits publication, not agent review or interpretation.
Do not publish experiment input files, credentials, or full API checkpoints.

Keep model weights, builds, caches, and detailed runtime checkpoints ignored.
Commit only the compact response log and its metadata under `logs/`.
