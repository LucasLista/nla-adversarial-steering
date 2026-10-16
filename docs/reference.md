# Reference

Run commands from the repository root using modules, such as
`uv run python -m nla.launch`. Former root scripts now live in `nla/`;
model manifests are in `config/`. Weights, `.runtime/`, private catalogs, and
run directories retain their locations. The dated dashboard and Gemma campaign
wrapper were removed; use `tools.batch_viewer`, snapshot staging, HF job
submission, and `hpc.hpc_control` instead.

## Experiment semantics

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

The viewer expects `run-1`, `run-2`, and `run-3` subdirectories and binds loopback
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

## DTU HPC

Download Qwen weights on HPC with `uv run python -m nla.download`, then submit from the
repository root after `mkdir -p runs`:

| Job | Purpose |
| --- | --- |
| `bsub < hpc/server.lsf` | Authenticated Qwen server, one V100, four hours |
| `bsub < hpc/batch.lsf` | Qwen batch runner, one 32 GB V100, 24 hours |
| `bsub < hpc/batch-sol.lsf` | Same batch runner with the OpenAI API editor |
| `bsub < hpc/hf-server.lsf` | Validation-gated HF backend, two V100s, four hours |

Qwen jobs share compiler/build setup in `hpc/common.sh`; batch jobs share
`hpc/run_batch.sh`. They retain per-job ports and tokens, resident model processes,
and a 16,384-token base context. CUDA 12.4 targets architecture 70. All three
models plus contexts must fit in VRAM. Resource requests stay in each LSF file.

The laptop controller is `uv run python -m hpc.hpc_control`. Required flags are
`--job`, `--ssh-host`, `--ssh-socket`, `--remote-root`, and `--output`, plus either
`--configs`, preset IDs (`--probe-id`, `--editor-preset`), or inline inputs. Use
`--help` for editor, token-limit, and port options. It tunnels the authenticated
API, keeps editor credentials local, and copies checkpoints back to HPC. The
laptop must remain connected. `--keep-serving` retains the browser relay; jobs
otherwise remain allocated until walltime or `bkill JOB_ID` after successful runs.

`uv run python -m hpc.background_controller -- ...` detaches that controller and
prevents idle sleep on macOS. Use progress and scheduler metadata to monitor it;
controller and API logs may contain private experiment content.

For other released families, `hpc.model_preflight` checks pinned snapshot access.
`hpc.hf_stage` streams verified files to HPC through an SSH reverse tunnel while
keeping Hugging Face credentials local. Its required flags are `--model`,
`--ssh-host`, `--ssh-socket`, and `--remote-root`. Weights stay on HPC. Download
completion alone does not establish backend compatibility.

The HF job uses a separate Python 3.12/Torch 2.6 CUDA environment, NF4 weights,
and FP32 compute. Its initial resource plan is for Gemma-3-12B; larger models need
measured memory plans. The server requires synthetic finite-vector and
unchanged-text controls before private inference. Use `--hf-backend --model MODEL`
with the controller only after validation. Sanitized validation and failure
metadata live in `.runtime/`; never infer success from submission alone.
