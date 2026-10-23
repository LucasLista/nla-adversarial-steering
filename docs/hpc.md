# HPC guide

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

## Submitting with parameters

This cluster's LSF requires explicit `-env` parameters. For example, to set the
benchmark manifest and token limit for the self-contained batch runner:

```sh
bsub -env "all,NLA_BENCHMARK_MANIFEST=runs/benchmarks/jbb-benign/manifest.json,NLA_MAX_TOKENS=512" < hpc/batch.lsf
```

Plain `VAR=value bsub < script` does not propagate these settings to the job.
Use `hpc.hpc_batch --help` for optional sharding and steering multipliers; the
four-branch experiment uses the default correction multiplier of one.

## Subscription account

The `codex` editor runs Codex CLI on the controller machine. It uses that
machine's ChatGPT sign-in, inherited `CODEX_HOME`, and current credentials on
each invocation. It strips API-key environment variables and requires ChatGPT
authentication. `openai` is a separate, API-billed editor.

Before a campaign, verify `codex login status` in the controller's environment.
For multiple accounts, compare the CLI's `account/read` account ID with the
intended app account; a successful login alone does not identify the subscription.
Keep authentication on the controller host. The GPU job serves model inference
and does not need the ChatGPT credentials.

## Resume and inspect

Keep the same output directory and experiment configuration when resuming.
Completed stages are reused. Do not delete checkpoints to recover a failed job.
For laptop-controlled runs, the local checkpoint may be ahead of its HPC copy.
Use `tools.run_status` and scheduler metadata to check progress; do not infer
quality from completion or inspect controller logs as a status shortcut.

Connect to `login1.gbar.dtu.dk` with your own DTU account and set `--remote-root`
to your checkout. Shared-directory access is managed by the project owners.
