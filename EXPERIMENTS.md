# Experiments

`runs/experiments/<name>/inputs/config.json` contains human-owned input settings. The initial `first` experiment is blank. Edit it yourself using keys `prompt`, `instruction`, `editor` (`local` or `codex`), `mode` (`review` or `auto`), `token-mode` (`manual` or `auto`), and `steps` (1–20).

In the stepper, enter the experiment name and select **Load inputs**, then **Start**. **Save run** writes a complete snapshot under `results/`. **Results → Open** restores its token history, descriptions, edits, activation vectors, reconstruction errors and probabilities. Saves are explicit; browser autosave remains separate. A result may also contain prompts and settings, so treat it as private research data.

Copy the entire `runs/experiments/` folder separately when moving to HPC: runs are intentionally excluded from Git. Run the same launcher there and access its stepper through an SSH tunnel. This change does not submit HPC jobs. Each result JSON also works through the existing Import control.

Grey history tokens indicate all four distributions agree on the most likely token, regardless of which token you chose. Missing comparisons are never marked as agreement.
