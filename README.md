# NLA playground

Inspect and edit Qwen2.5-7B-Instruct activations with Anthropic's released
[Natural Language Autoencoder](https://transformer-circuits.pub/2026/nla/index.html).
The local interface wraps [nla.cpp](https://github.com/thomasgauthier/nla.cpp):
explain a token's activation, edit the explanation, and compare continuations.

## Run locally

Requires uv, Git, curl, CMake 3.20+, and a C++17 compiler. Use macOS, Linux, or
Ubuntu under WSL2. Allow 30 GB free disk and at least 16 GB RAM; CPU inference
loads one model at a time and can pause for minutes when switching roles.

```sh
# Ubuntu/WSL prerequisites; on macOS use Xcode command-line tools and CMake.
sudo apt install git curl cmake build-essential
uv sync --locked
uv run setup.py
uv run launch.py
```

uv manages Python 3.15. Setup builds the pinned backend and downloads 12.8 GB of
SHA-256-verified checkpoints. Downloads resume; `setup.py --skip-download` only
builds. Keep WSL checkouts in the Linux filesystem. The macOS build path has not
been verified on hardware.

Open **http://127.0.0.1:18090/stepper**. Start a session, inspect a token, then
explain and reconstruct its activation. Compare the original, reconstructed,
edited, and error-preserving distributions; choose a token to continue. Export
sessions you want to keep. Ctrl+C stops the launcher and its model processes.

The upstream Mikupad interface is at `/`. Use its llama.cpp completion endpoint,
ChatML template, and layer 20. Token highlighting and probability display must
be enabled for **Explain NLA**. Browser JavaScript dependencies require internet.

For NVIDIA GPUs, install a compatible CUDA toolkit and build separately:

```sh
uv run setup.py --cuda --skip-download
uv run launch.py --gpu-layers 99
```

Use `--cuda-arch` for a specific architecture, `--gpu-layers` to limit offloading,
and `--threads` to limit CPU use. CPU remains the default.

## Repository

| Path | Purpose |
| --- | --- |
| `setup.py`, `download.py`, `launch.py` | Build, download, and serve the local playground |
| `trajectories.py` | Independent branches with resumable stage checkpoints |
| `nla/` | Backend adapters, editor providers, and session operations |
| `ui/` | Browser interfaces |
| `hpc/` | DTU LSF jobs, remote staging, and controllers |
| `tools/` | Progress reports, snapshot publication, and checkpoint viewer |
| `tests/` | Offline regression tests and opt-in live checks |
| `models.json`, `evaluation_models.json` | Pinned checkpoint manifests |
| `prompts.json`, `editor_prompts.json` | Human-owned experiment catalogs |

[The reference](docs/reference.md) covers experiment semantics, saved runs,
editors, and HPC operation. [AGENTS.md](AGENTS.md) defines unread research
artifacts. Weights, builds, and private checkpoints stay ignored; `logs/` holds
published snapshots and `examples/` holds historical validation artifacts.

## Checks

No weights, GPU, or editor account are needed:

```sh
uv run python -m unittest discover -s tests
node tests/test_loop.mjs
```

Live checks require a running local launcher: `uv run python -m tests.smoke_test`
and `uv run python -m tests.test_stepper_live`. They write private results under
`runs/`; the smoke check also prints model output and is for human review.

## Scope and provenance

The released NLA is specific to a model and layer. Qwen uses zero-based block 20
and 3,584-dimensional residuals. Quantized reconstructions and explanations are
experimental; passing numerical controls does not establish semantic control or
BF16 equivalence. This repo does not train an NLA or introduce a steering method.

The wrapper is MIT-licensed. The backend is pinned to
`7a3b9c72462d72414042e737dbb75353d75f0a27` (MIT); its Mikupad frontend is AGPL-3.0.
Setup preserves upstream notices. Checkpoint licenses belong to their publishers.
See the official [training code](https://github.com/kitft/natural_language_autoencoders)
and [inference recipe](https://github.com/kitft/nla-inference). Development used AI assistance.
