# Unread experiment content

To keep the agent's context independent of this safety evaluation, agents must
not open, display, read, search, watch, summarize, interpret, or inspect:

- `prompts.json` and `editor_prompts.json` values (including `bio_question` and `poetic`).
- Anything under `runs/experiments/*/inputs/`.
- Model responses, NLA explanations, editor proposals, decoded generated tokens,
  activation vectors, and request/response payloads in `runs/`, `logs/`, or `examples/`.
- All log contents, including `logs/**`, runtime `.log` files, scheduler stdout/stderr,
  and controller output. Do not tail logs to monitor progress.
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

# Monitoring without reading logs

Use `run_status.py --run runs/RUN_NAME` and LSF scheduler metadata for status.
The status helper outputs only branch counts, ended flags, branch completion,
and overall completion. New runs maintain a separate `progress.json` with
these fields. For older running processes, the helper may programmatically
extract those fields from checkpoints without displaying any other content.
`complete.json` existence signals completion of all four branches.
Never read response logs, controller output, or scheduler log text to check
completion. Blind copy/hash/stage/push operations remain authorized.

Synthetic GPU validation may record exception class, backend phase, and source
frame filenames/functions/line numbers in dedicated `.runtime/hf-job-JOB.json`
metadata. These diagnostics may be inspected. Never expose exception messages,
frame locals, source payloads, or raw validation/model output.
