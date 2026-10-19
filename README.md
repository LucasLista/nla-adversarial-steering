# NLA adversarial steering

Can editing a natural-language description of an activation change what a model says next?
This repository runs that experiment on Qwen2.5-7B-Instruct, using Anthropic's
released natural language autoencoder (NLA) at layer 20.

An NLA turns an activation into text and reconstructs an activation from that
text. We edit the description, reconstruct it, and continue generation. Four
independent trajectories separate the effects of reconstruction and editing:

| Branch | Activation used for the next token |
| --- | --- |
| Original | `h` |
| Reconstructed | `AR(t)` |
| Edited | `AR(t')` |
| Edited with error correction | `AR(t') + h - AR(t)` |

Here `h` is the original activation, `t` its description, `t'` the edited
text, and `AR` the reconstructor. Each branch generates its own continuation.
The intervention is reapplied at each step; earlier edited KV states are not replayed.

## Results so far

The [October 19 results note](docs/results.md) reports a preliminary evaluation
of 42 matched benign cases. The refusal judge marked 1 original, 2 reconstructed,
0 edited, and 1 error-corrected response as refusals. Most responses reached the
128-token cap. These small counts do not establish a steering benefit.

The harmful-case campaign is still running. The saved evaluation contains no
graded harmful responses, so we do not yet have an attack-success estimate.

## Try it

You need [uv](https://docs.astral.sh/uv/), Git, curl, CMake 3.20+, and a C++17
compiler, with about 30 GB free disk and at least 16 GB RAM. Linux and WSL2 use
the CPU path by default. The macOS build path has not been verified on hardware.

```sh
# Ubuntu / WSL2; keep a WSL checkout in the Linux filesystem.
sudo apt install git curl cmake build-essential

git clone https://github.com/LucasLista/nla-adversarial-steering.git
cd nla-adversarial-steering
uv sync --locked
uv run python -m nla.setup
uv run python -m nla.launch
```

Setup builds the pinned [nla.cpp](https://github.com/thomasgauthier/nla.cpp)
backend and downloads 12.8 GB of hash-verified checkpoints. Interrupted downloads
resume. uv manages Python 3.15.

Open [localhost:18090/stepper](http://127.0.0.1:18090/stepper). Start a session,
inspect a token, and explain its activation. Edit the description to compare
the next-token distributions, then choose a token to continue. Export sessions
you want to keep. CPU inference can take minutes when switching models.

With an NVIDIA GPU and a compatible CUDA toolkit:

```sh
uv run python -m nla.setup --cuda --skip-download
uv run python -m nla.launch --gpu-layers 99
```

## Work with the code

The intervention math is in [nla/stepper.py](nla/stepper.py); the generation loop
and resumable checkpoints are in [nla/trajectories.py](nla/trajectories.py).
The browser interface is [ui/stepper.html](ui/stepper.html).

- [Experiment reference](docs/reference.md): intervention details, editors, saved runs.
- [HPC guide](docs/hpc.md): batch jobs, subscription-backed controllers, monitoring.
- [Results](docs/results.md): dated counts, grading provenance, and limits on interpretation.

`config/` pins the models, `hpc/` runs cluster jobs, and `tools/` reports progress
and exports snapshots. Private inputs and detailed checkpoints live in ignored
`runs/`; `logs/` contains published response snapshots. Agent access to experiment
content is governed by [AGENTS.md](AGENTS.md).

## Credits and scope

This project uses the released NLA; it does not train one or introduce a new
steering method. Explanations are hypotheses about activations. Quantized
inference and numerical controls alone do not establish semantic control or
agreement with BF16 inference.

See the NLA [training code](https://github.com/kitft/natural_language_autoencoders)
and [inference recipe](https://github.com/kitft/nla-inference). The wrapper is
[MIT-licensed](LICENSE). The backend is pinned to
`7a3b9c72462d72414042e737dbb75353d75f0a27` (MIT); its Mikupad frontend is AGPL-3.0.
Checkpoint licenses belong to their publishers. Development used AI assistance.
