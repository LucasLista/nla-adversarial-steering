# NLA adversarial steering

A small experimental harness for studying robustness to adversarial edits in
natural language autoencoders (NLAs). It runs Qwen2.5-7B-Instruct with Anthropic's
released layer-20 NLA: explain an activation, edit the text, reconstruct it,
and continue generation.

Research is ongoing. We will link the write-up here when it is published.

## Method

Four independent trajectories separate the effects of reconstruction and editing:

| Branch | Activation used for the next token |
| --- | --- |
| Original | `h` |
| NLA roundtrip | `AR(t)` |
| Edited | `AR(t')` |
| Edited + error | `AR(t') + h - AR(t)` |

Here `h` is the original activation, `t` its description, `t'` the edited
text, and `AR` the reconstructor. Each branch generates its own continuation.
At each generation step, we recompute the branch's activation from its own token
prefix and intervene at the last position. Earlier edited KV states are not
replayed. Generation is greedy. The corrected branch preserves the original
reconstruction error, equivalently adding `AR(t') - AR(t)` to `h`.

The editor receives the description and an explicit editing instruction.
Its proposal is saved separately, so changes to the text can be inspected
alongside changes to the model's continuation.

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

## Code

Start with three files:

- [nla/stepper.py](nla/stepper.py): activation interventions and next-token comparisons.
- [nla/trajectories.py](nla/trajectories.py): independent generation loops and resumable checkpoints.
- [ui/stepper.html](ui/stepper.html): interactive inspection and editing.

For usage details:

- [Experiment reference](docs/reference.md): intervention details, editors, saved runs.
- [HPC guide](docs/hpc.md): batch jobs, subscription-backed controllers, monitoring.

`config/` pins the models; `hpc/` contains optional DTU cluster adapters.
Inputs, checkpoints and local exports are ignored by Git. Agent access to
experiment content is governed by [AGENTS.md](AGENTS.md).

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
