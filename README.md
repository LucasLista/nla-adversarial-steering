# NLA playground

Inspect and edit Qwen2.5-7B-Instruct's activations using Anthropic's released
Natural Language Autoencoder (NLA). Hover a token, read an explanation, edit
the text, and apply the reconstructed edit direction to generation.

This packages [Thomas Gauthier's nla.cpp](https://github.com/thomasgauthier/nla.cpp)
and its Mikupad interface with a laptop-oriented launcher. It loads **one model
at a time**, on CPU, so you do not need a GPU, API key, or another person's PC.
It does not train an NLA or claim a new steering method.

## Requirements

- uv, Python 3.15 (managed by uv), Git, curl, CMake 3.20+, and a C++17 compiler.
- **16 GB RAM recommended as a practical starting point**; close heavy apps.
  8 GB machines are not a supported target. Memory depends on the backend and OS.
- **30 GB free disk** for the first setup, including temporary downloads/builds.
  The three model files total 12.8 GB; download parts are removed after verification.
- macOS, Linux, or Ubuntu under WSL2 on Windows. Linux/WSL execution has been
  tested; macOS is a CPU build path, not yet hardware-verified.
- An internet connection for setup and the UI's JavaScript dependencies on esm.sh.

Expect pauses when switching models. On the tested Ryzen 5 3600 PC, the first
explanation including model switching took about 48 seconds; subsequent AR
reconstruction took about 5 seconds. These are individual measurements, not a
laptop performance guarantee. The interface binds only to localhost.

## Install and run

Clone this repository, then change into it. On Ubuntu/WSL install prerequisites:

```sh
sudo apt update
sudo apt install -y git curl cmake build-essential
```

On macOS, install Apple's command-line tools (`xcode-select --install`),
CMake and uv. If you already use Homebrew: `brew install uv cmake`.
On Windows, run all commands inside Ubuntu WSL2, and keep the project in the
Linux filesystem, such as `~/nla-playground`, rather than `/mnt/c`.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed,
then run:

```sh
uv python install 3.15
uv sync --locked
uv run --locked setup.py
uv run --locked launch.py
```

`.python-version` selects Python 3.15, and `uv.lock` records the project environment.
The Python scripts use only the standard library, so there are no Python package
dependencies. uv creates `.venv` automatically; no activation is required.
`setup.py` builds the pinned C++ backend and downloads models; uv manages the
Python environment. To build without downloading models, use
`uv run --locked setup.py --skip-download`.

Open **http://127.0.0.1:18090** in your browser. Leave the terminal open.
Ctrl+C stops the launcher and the model process it owns. Models remain on disk.
Setup can be rerun; downloads resume from completed 16 MB chunks and each final
file is checked against its pinned SHA-256 hash. No weights are committed here.

Optional: `uv run --locked launch.py --threads 4 --port 18090 --backend-port 18089`.
Use distinct unused ports; fewer threads may keep the laptop more responsive.
Run `uv run --locked smoke_test.py` while the launcher is open to exercise all model roles.
Its report is saved in `runs/smoke-results.json`.

## First experiment

1. The new-session prompt is already formatted for Qwen. Click **Predict**.
2. Hover or click a token in the editor, then click **Explain NLA**.
3. Keep **Layer 20**. Read the explanation and optionally **Score reconstruction**.
4. Make a minimal edit in **Edited explanation**. Start at **α = 0.1**.
5. Click **Save steering**, close the panel, and generate from the same prefix.
   Saving alone does not rewrite existing output. For a fair comparison, return
   to the same prefix and retain the same seed/sampling settings.
6. **Clear steering** restores ordinary generation.

Keep the endpoint API as **llama.cpp** and use completion mode, not Chat API.
For newly inserted turns keep the **ChatML** instruct template; the initial
prompt already contains the correct special tokens. Token highlighting and
probability display must be enabled in Editor Preferences for the Explain button.

If Mikupad warns that persistent storage was denied, click OK. Inference still
works. **Export** sessions you want to keep: browser storage can be cleared.
Existing browser sessions retain their settings despite new setup defaults.

`prompts.json` contains English/Chinese historical questions and a rhyme example.
Change only one thing at a time. Useful controls are no steering, identical text,
a meaning-preserving paraphrase, an unrelated edit, and opposite-sign strengths.
Measure output quality as well as the desired behavior; save all runs, not just
striking examples. Use held-out prompts for any claim in a write-up.

## What this actually changes

The target model has 28 blocks. **L20 is zero-based block 20**, i.e. the 21st
block; its output corresponds to Hugging Face `hidden_states[21]`. One selected
token supplies a 3,584-dimensional residual-stream vector. The AV maps that
vector to text; the AR maps explanation text back to an activation.

The backend applies this at the selected token and block:

```text
delta = AR(edited explanation) - AR(original explanation)
h_new = h + alpha * norm(h) * delta / norm(delta)
```

It skips a zero direction. It does not replace the entire activation with the
AR reconstruction. The released NLA is model- and layer-specific; changing the
layer is an out-of-distribution experiment, not a validated universal reader.

## Evidence and limitations

`examples/verified-smoke-results.json` records a real run of the same pinned
backend/checkpoints: extraction and reconstruction had 3,584 dimensions;
identical text produced zero direction; alpha zero preserved greedy output.
Alpha 0.1 left the poem output unchanged. Alpha 1 changed it but did not achieve
the intended mouse rhyme. This verifies the plumbing, not reliable semantic control.

These are **community Q4_K_M conversions** of the original model and released
NLA checkpoints. BF16 equivalence has not been established. Explanations can
hallucinate details even with apparently good reconstruction cosine. Treat them
as hypotheses, not literal thoughts or ground truth. Successful text editing
and steering already appear in Anthropic's paper; a research contribution needs
additional controlled evidence.

`examples/tiananmen-baselines.json` records two prompts with explicit military
and casualty questions in English and Chinese, using this local quantized model.
They are samples of this checkpoint's behavior, not claims about every Qwen
model, hosted Qwen products, or every wording. Model-generated historical claims
are not independently fact-checked by this repository.

## Provenance and licenses

- [Anthropic's NLA paper](https://transformer-circuits.pub/2026/nla/index.html)
- [Official training code](https://github.com/kitft/natural_language_autoencoders)
  and [inference recipe](https://github.com/kitft/nla-inference)
- [nla.cpp](https://github.com/thomasgauthier/nla.cpp), pinned to
  `7a3b9c72462d72414042e737dbb75353d75f0a27` (MIT backend; Mikupad frontend is AGPL-3.0).
  Setup preserves upstream source and license notices. Its generated frontend
  changes only the starting prompt, ChatML template choice, context/generation
  limits, seed, and alpha.
- `models.json` pins Hugging Face repository revisions, filenames, sizes and hashes.
  Model licenses remain those of their respective publishers.

The wrapper/setup code here is MIT-licensed. Development and packaging used AI
assistance. No upstream authorship is claimed. The build, model downloads, and
runtime logs are intentionally excluded from Git.

## Token-by-token activation laboratory

Open `http://127.0.0.1:18090/stepper` after starting the launcher.

1. Enter a prompt and press **Start / reset**, then **Inspect**. This extracts the
   last context token's residual after block 20 and shows the next-token top 30.
2. **Explain and reconstruct** generates the NLA text and saves the complete
   reconstruction error vector. Edit the text if desired.
3. **Compare** evaluates original activation, original-text reconstruction,
   edited-text reconstruction, and edited reconstruction plus the original error.
4. **Choose** a token from any distribution, then inspect again. Exact token IDs
   are appended, avoiding lossy decode/re-tokenize cycles. EOS ends generation.
5. **Export** saves a JSON copy in the model host’s `runs/` directory and requests a browser download containing tokens, choices, explanations, activation vectors, error vectors,
   metrics and distributions as JSON. Import restores that session.

For original activation `h`, original explanation `t`, and edited text `t'`:

```
r = AR(t)
e = h - r
raw round trip = r
raw edit = AR(t')
error-preserving edit = AR(t') + e = h + AR(t') - AR(t)
```

The original and reconstructed vectors use their native units; there is no norm
matching. The backend's normalized steering API is compensated with
`alpha = ||target-h|| / ||h||`, `direction = target-h`, to inject the actual delta.
Floating-point arithmetic and quantization can introduce small numerical differences.
With unchanged text the error-preserving branch should reproduce the baseline.
The full residual error is saved alongside L2, relative L2, MSE and cosine similarity.

Distributions are the model's raw softmax probabilities, before temperature or
sampling filters, obtained by continuing through the remaining blocks. The UI
shows top 30 and remaining probability mass; it does not claim to show the full
vocabulary. It is not a layer-20 logit lens. Generation is manual token selection.

**Intervention scope:** one position per step. Each new step recomputes the full
context from token IDs without replaying earlier activation edits. Earlier choices
persist as tokens, but edited KV states do not persist. This is a context-recompute
experiment, not continuous generation with a patched KV cache. CPU model swapping
can take minutes per explain/compare cycle. Stay below 1900 context tokens.

Run `uv run --locked test_stepper_live.py` against the running launcher to check the
unchanged-text control, an edited distribution, vector dimensions, error identity,
and the next generation step. Results go to `runs/stepper-validation.json`.

### Refusal probes

The dropdown includes political questions, candidate witness-intimidation and
illegal-drug-sale requests, and a benign drug-health question. They are original
probe wording, not extracted benchmark items; responses depend on the checkpoint,
quantization, context and system prompt. A drug-related topic alone is not a refusal
criterion. Record actual outputs and distinguish refusal from omission, factual
error, redirection and normal answers. For broader evaluation, see
[SORRY-Bench](https://sorry-bench.github.io/) and
[OR-Bench](https://raw.githubusercontent.com/mlresearch/v267/main/assets/cui25a/cui25a.pdf) for refusal and over-refusal methodology.

### Model-assisted editing loop

The stepper's **Model-assisted editing loop** uses the local base Qwen model as
an editor (no external service). Set an editing instruction, choose **Review every
edit** or **Auto-apply edits**, and choose a limit of 1–20 tokens.

In review mode the loop pauses after proposing an edit. You can change the proposed
text in **Edit NLA text**, approve it, or reject/stop. Apply edit runs the comparison and, by default, pauses at the token distributions.
Choose a token from any branch to continue. The separate Tokens selector can enable
automatic greedy selection from the edited-plus-error branch. Edit approval and
token selection are independent; both default to manual. The loop stops at its
limit, EOS, an error, or Stop. Stop waits for the current model request to return and
prevents subsequent application/generation. It does not kill a model subprocess.
Each completed step records the proposed text, editing instruction, actual applied
text, manual/automatic approval, distributions, activation vectors and selected ID.
The model may propose ineffective edits or decline an instruction; proposals are
experimental hypotheses, not guaranteed semantic changes.

Verified local probes are saved in `examples/refusal-probes.json` (100-token
responses, so some end mid-sentence). Both the witness-intimidation and illegal
sales prompts produced explicit refusals in this Q4 setup. These are examples,
not a refusal-rate estimate. The numerical round-trip control is recorded in
`examples/stepper-validation-summary.json`.

For the approval-loop regression checks (no models required), run
`node test_loop.mjs`. These check that review mode does not inject before approval,
that approval is recorded, that auto mode stops at its token limit, and that a stop
request before application prevents injection. Node is only needed for this test,
not to run the playground.

### GPT-6.1 Sol through your ChatGPT subscription

Select **GPT-6.1 Sol · ChatGPT subscription via Codex** in the editor dropdown.
It works with both review and auto modes. Only the NLA description and editing
instruction are sent to OpenAI; vectors and the underlying conversation are not
included. Proposals record the exact model `gpt-6.1-sol` and provider.

Install the official [Codex CLI](https://developers.openai.com/codex/cli), run
`codex login` using ChatGPT, and confirm `codex login status` reports ChatGPT.
If Codex is installed and signed in on the same machine as `launch.py`, no further
configuration is necessary. The adapter uses `codex exec`, ephemeral sessions,
a temporary working directory, a read-only sandbox, and structured output. It
requires ChatGPT authentication and never falls back to API-key billing or another
model. Normal subscription limits and model availability apply. Jonathan should
use his own account; do not share auth files.

For a model server on a separate PC, run on the signed-in laptop:

```
python3 codex_editor.py --serve
ssh -N -R 127.0.0.1:18094:127.0.0.1:18094 YOUR_MODEL_HOST
```

Then start the model host with:

```
NLA_CODEX_EDITOR_URL=http://127.0.0.1:18094 python3 launch.py
```

Both bridge and reverse tunnel bind loopback only. Keep them running while using
Sol. If the SSH server runs in Windows but the model launcher runs in WSL, use
`NLA_CODEX_EDITOR_TRANSPORT=windows-curl` as well; this uses Windows' loopback
through its bundled curl.exe. The helper runs Codex on the laptop, so ChatGPT
credentials stay there. All three processes must be restarted after a reboot.

The bridge rejects browser-origin requests and accepts server-to-server JSON only.
The CLI is instructed to perform a text rewrite without tools. No shell command
is constructed from model output or user text. Timeout, authentication, and access
errors stop the loop instead of applying a missing proposal.

Official references: [authentication](https://developers.openai.com/codex/auth),
[non-interactive mode](https://developers.openai.com/codex/noninteractive).

### NVIDIA GPU acceleration

CPU remains the default. Build CUDA into a separate directory to keep the CPU
fallback available:

```
uv run --locked setup.py --cuda --cuda-arch 61 --skip-download --jobs 2
uv run --locked launch.py --gpu-layers 99
```

`61` targets a GTX 1070 Ti (Pascal). This GPU needs CUDA 12.x, not CUDA 13.
Use an appropriate architecture for other cards (default `native`). Install a
compatible CUDA toolkit and host compiler first. With CUDA 12.0 and GCC 13 on
Ubuntu, set `CUDAHOSTCXX=/usr/bin/g++-12` for the setup command. Under WSL, keep the
Windows NVIDIA driver; the launcher prioritizes the WSL CUDA driver bridge.

`--gpu-layers 99` offloads all available model layers. Models still load one at a
time. Use fewer layers if VRAM is insufficient, or omit the flag for the CPU
backend. `--build-dir /path/to/build` selects an explicit backend build.
The GPU accelerates local inference; Sol editing still runs through Codex.

### Generation history

Click a numbered token under Generated text to inspect that step’s saved original
NLA, edit, error metrics, and distributions. The chosen token is highlighted.
History is read-only: click Live to resume choosing tokens. Viewing history does
not truncate or regenerate the sequence. Export/import includes the history.

### Four independent trajectories on DTU HPC

`hpc/server.lsf` requests one exclusive V100 for four hours, builds CUDA 12.4
for architecture 70, and launches the backend. Submit from the repository
on HPC after running `uv run --locked download.py`; all weights stay on HPC.
The server uses a per-job authentication token for cluster-network access.
Job output, CUDA device information and build/backend logs are saved in `runs/`.

Run `hpc_control.py` on your signed-in laptop with `--job`, `--ssh-host`,
`--ssh-socket`, `--remote-root`, `--output`, `--prompt`, and `--instruction`.
It waits for the allocation and verified models, opens an SSH tunnel and local
relay, then uses the existing `codex_editor.py` subscription editor locally.
Your laptop must remain awake and connected during the run. No Codex login
or credentials are needed on HPC.

The four branches are original, reconstructed (`roundtrip`), edited
(`edited_raw`), and edited plus reconstruction error (`edited_corrected`).
Each uses its own generated prefix and greedily chooses its own next token.
The reconstructed branch uses the original explanation without calling the
editor. Responses can coincide; no difference is forced. The intervention remains
one position per step, without replaying earlier activation edits in KV caches.

Every successful API stage is atomically saved and reused on resume, including
requests, complete returned vectors, explanations, errors, metrics, edits,
distributions and timing. Backend distributions contain top 30 tokens rather
than the whole vocabulary. Failed stages write separate error records. Each token
updates its branch state; `responses.json` collects the responses and stop reasons.
Rerun with the same arguments and output directory to resume. Completed stage
artifacts are saved before subsequent requests. Artifacts are copied back to HPC
when generation completes or fails. `python -m unittest test_trajectories` checks
independent prefixes, resume behavior, error records and EOS stopping.

Authenticated HPC launches also enable server-side API auditing: request and
response JSON files are written under `runs/api/` before replies are sent. This
preserves GPU results even when a client disconnects. `--audit-log` enables the
same logging for a loopback launcher. The laptop relay routes interactive Codex
editing requests locally too, so the subscription dropdown works without logging
Codex into HPC. An optional `--connect-script` reuses your existing local SSH
login helper after transient connection errors.

For a browser session, add `--keep-serving`; otherwise the controller closes
its relay and tunnel when the four trajectories finish. The cluster job remains
reserved until its walltime expires or you stop it with `bkill JOB_ID`.
The batch script pins the same GCC compiler for C++ and CUDA, reconfigures a
stale compiler cache, and requests 6 GB RAM per CPU slot with a 7 GB process limit
compatible with DTU's submission rules.

For multiple private experiment configs, the controller also accepts
`--configs runs/experiments/run-1/inputs/config.json ...` instead of inline
`--prompt` and `--instruction`. It loads prompts at runtime without printing
those inputs. `--max-tokens 10000` is a safety cap; EOS still stops each branch.
Branches and experiments run sequentially, with stage checkpoints on the laptop.
`--port 18095` (the default) keeps the PC interface on port 18090 available.
Set `NLA_CONTEXT_SIZE=16384` in the HPC server environment for long runs;
otherwise the original 2048-token context remains the default. The backend
reserves 148 context positions. Walltime can interrupt any stage; resume from
the same output directory. Model download/build time counts against walltime.

### Editor instruction presets

Add entries to `editor_prompts.json` as objects with `id` and `prompt` strings.
Refresh `/stepper`, select an Instruction preset, and choose Local Qwen or Sol.
The selected text fills the editable Instruction field; manual edits are retained
as a custom instruction. Presets change the editing instruction, while each
provider's system instructions remain in `stepper.py` or `codex_editor.py`.

For catalog-based HPC runs, pass `--probe-id ID --editor-preset ID` instead
of inline prompt and instruction text. Catalog values are loaded at runtime,
so command arguments contain only IDs. Checkpoints still contain experiment
text and should be treated as private run artifacts.

### Response logs and context isolation

`AGENTS.md` defines unread input and output artifacts for these evaluations.
`publish_run.py --run runs/RUN_NAME --name RUN_NAME` copies a compact response
snapshot into `logs/` without displaying model text. Commit that directory to
share authorized response logs; `status.json` distinguishes partial snapshots
from completed runs. Detailed checkpoints, inputs, and weights remain ignored.

### Released model evaluation matrix

`evaluation_models.json` records pinned upstream base, AV and AR revisions for
Qwen2.5-7B (L20, 3584 dimensions), Gemma-3-12B (L32, 3840), Gemma-3-27B
(L41, 5376), and Llama-3.3-70B (L53, 8192). These are the four model families
in the official NLA release; arbitrary Qwen/Gemma versions are not substitutes.

Run `uv run --locked model_preflight.py` for an access/readiness report that
contains no experiment inputs or model outputs. It uses a local Hugging Face
credential if available but never prints it or transfers it to HPC. Approve
the base-model licenses on Hugging Face and authenticate locally when needed.

Only Qwen currently has a validated backend here. The other families require
model-specific chat templates, EOS IDs, extraction layers, activation dimensions,
AV sidecar injection parameters, and AR conversion/loading support. A model is
not runnable just because its files are downloadable. `hpc_control.py --model`
rejects unvalidated families rather than silently applying Qwen settings.
Gemma-27B and Llama-70B also need a measured memory plan; fitting their weights
on one V100 must not be assumed. Existing Qwen experiments remain resumable.
