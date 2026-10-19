# Experiment reference

Run commands from the repository root. For cluster jobs, see the [HPC guide](hpc.md).

## Experiment semantics

Qwen uses zero-based block 20 and 3,584-dimensional residuals.

For activation `h`, explanation `t`, and edited explanation `t'`, the stepper
compares `h`, `AR(t)`, `AR(t')`, and `AR(t') + h - AR(t)`. It saves the full
reconstruction error plus L2, relative L2, MSE, and cosine similarity. Unchanged
text should reproduce the baseline in the error-preserving branch, within
floating-point and quantization tolerance.

Mikupad's steering endpoint instead applies
`h + alpha * norm(h) * delta / norm(delta)`, where `delta = AR(t') - AR(t)`;
a zero direction is skipped. The stepper compensates for that normalization to
apply its target vector in native units.

Each step recomputes context from exact token IDs and intervenes at one position.
Earlier token choices persist; edited KV states do not. Distributions are raw
softmax probabilities after the remaining blocks, with only the top 30 exposed.
Keep local contexts below 1,900 tokens. Model/layer changes require their own
validation; explanations are hypotheses, not ground truth.

The editing loop separates edit approval from token selection. Both default to
manual. Review mode pauses before applying a proposal; automatic modes have a
token limit. Stop waits for the current request and prevents subsequent work.
History is read-only; return to Live to continue. Grey history tokens mean all
four distributions agree on the most likely token.

## Saved runs

Legacy human-owned settings live in `runs/experiments/NAME/inputs/config.json`, with
`prompt`, `instruction`, `editor`, `mode`, `token-mode`, and `steps` fields.
Load inputs in the stepper; Save run writes a snapshot under that experiment's
`results/`. Results → Open restores it. Browser autosave is separate from explicit
exports. Copy private runs separately when moving machines.

The batch runner uses four independent prefixes: `original`, `roundtrip`,
`edited_raw`, and `edited_corrected`. Successful stages are atomically saved and
reused on resume. Rerun with the same config and output directory; checkpoints
include inputs, vectors, proposals, probabilities, and timing. EOS, token limits,
and repeated-token termination remain recorded per branch.

```sh
uv run python -m tools.run_status --run runs/RUN_NAME
uv run python -m tools.batch_viewer --root runs/BATCH_NAME
uv run python -m tools.publish_run --run runs/RUN_NAME --name RUN_NAME
```

The viewer lists run directories beneath the supplied root and binds loopback
on port 18096. Publication copies compact response snapshots into `logs/`; it does
not commit or push. Use `--max-tokens` when reporting/publishing a nondefault
limit. Detailed checkpoints and input files stay private. Follow `AGENTS.md` for
content access and metadata-only monitoring.

## Editors

Local Qwen needs no external account. The subscription adapters use locally
signed-in Codex or Claude CLIs; the OpenAI adapter uses `OPENAI_API_KEY` from the
environment or root `.env` and incurs API charges. Provider choice is explicit;
subscription adapters do not fall back to API billing. Only the description and
editing instruction are sent to an external editor.

For a model host separate from a Codex-signed-in laptop, run the bridge and
reverse tunnel on the laptop:

```sh
uv run python -m nla.codex_editor --serve
ssh -N -R 127.0.0.1:18094:127.0.0.1:18094 YOUR_MODEL_HOST
```

Set `NLA_CODEX_EDITOR_URL=http://127.0.0.1:18094` on the model host. With a Windows
SSH host and WSL launcher, also set `NLA_CODEX_EDITOR_TRANSPORT=windows-curl`.
Keep both processes running. The bridge accepts server-to-server JSON on loopback
and rejects browser origins. Credentials stay on the editor host.

## Benchmark preparation

Prepare a ten-case benign pilot:

```sh
uv run python -m nla.benchmark --limit 10
```

The prompt catalog uses these cases. Pass `--instruction-file PATH` to enable
edited branches with a private editing instruction. Without it, batch runs use
only the original branch. The harmful split requires explicit `--category`
selection; physical harm is excluded. Keep the selection fixed across conditions.
These subsets are not official leaderboard submissions.

```sh
uv run python -m tools.benchmark_report --manifest PATH --results PATH
```

This reports completion. Optional `--grades` accepts external numeric scores;
it does not run a judge. Record the evaluator, version, and score definition.
The archived evaluation's hash-linked grade format is different from this
helper's `{case_id: {branch: score}}` input format.

`--fast` on `hpc.hpc_batch` computes only the selected branch distribution,
using the same intervention math. It records `comparison_mode` in the config;
use a fresh directory when changing this mode.

## Local interface options

Use `--threads` to limit CPU use, `--gpu-layers` to limit GPU offloading, and
`--cuda-arch` during setup to select an architecture. Setup's `--skip-download`
only builds the backend.

The upstream Mikupad interface is at `/`. Select its llama.cpp completion
endpoint, ChatML template, and layer 20. Enable token highlighting and probability
display for Explain NLA. Browser JavaScript dependencies require internet.

## Development checks

```sh
uv run python -m unittest discover -s tests
node tests/test_loop.mjs
```

These run without model weights or an editor account. Live checks need a running
launcher: `python -m tests.smoke_test` and `python -m tests.test_stepper_live`.
The smoke check prints model output and is for human review under AGENTS.md.
