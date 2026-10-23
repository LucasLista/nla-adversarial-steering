# Notes for coding agents

This is a safety evaluation, so agents must stay blind to experiment content
unless the human explicitly authorizes a specific review.

- Do not open, print, search, diff or summarize prompts, model responses, NLA
  explanations, editor proposals, decoded tokens or activations. They live in
  `runs/`, `logs/`, `examples/` and copies of those (on the cluster or in other
  checkouts). This includes cases files, instruction files and all logs.
- Check progress only with `python -m nla.run --status OUTPUT`, `progress.json`,
  `complete.json` and scheduler state (`bjobs`). Copying, hashing and moving
  files without reading them is fine.
- Never commit results, response snapshots, scores or progress reports, or put
  them in the README, until the human approves them for publication. Never commit
  experiment inputs, credentials, model weights, builds or checkpoints.
- Changes to the intervention must keep existing checkpoints resumable: the
  request bodies in `nla/trajectories.py` and the config keys in `nla/run.py` are
  part of the checkpoint format.
