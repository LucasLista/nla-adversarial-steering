# NLA adversarial steering

Can rewriting a model's *explanation of its own activations* talk it out of a refusal?

[Natural language autoencoders](https://www.anthropic.com/research/natural-language-autoencoders)
(NLAs) translate an activation into text and reconstruct an activation from that
text. Editing the text in between is a way to steer the model. This repository
uses that as an adversarial attack: at every generated token, an LLM editor
rewrites the NLA description of the model's state (say, from "about to refuse"
to "about to answer"), and the model continues from the reconstructed activation.
We use it to measure how robust refusals are to this kind of attack.

It runs on Qwen2.5-7B-Instruct with the released layer-20 NLA, on a laptop CPU
or a single GPU.

> **Status:** research in progress. We will link the write-up here when it is published.

## Method

At each step, take the residual activation `h` at the last position, explain it
as text `t = AV(h)`, edit `t` into `t'`, and predict the next token from one of
four independent branches:

| Branch | Activation at the last position |
| --- | --- |
| `original` | `h` |
| `roundtrip` | `AR(t)` |
| `edited_raw` | `AR(t')` |
| `edited_corrected` | `AR(t') + h - AR(t)` |

`AV` is the activation verbalizer and `AR` the reconstructor. `roundtrip`
isolates the reconstruction error. `edited_corrected` keeps that error and adds
only the change the edit causes, so an unchanged `t'` reproduces `original` up to numerical noise.
Each branch generates its own continuation greedily. The intervention is
recomputed at every step on that branch's own context, and edited activations
are never written back into earlier KV states. `--multipliers` scales the edit
direction: `h + m·(AR(t') - AR(t))`.

The editor receives the description and an editing instruction. Its proposal is
saved next to each step, so changes to the text can be read alongside changes to
the continuation. Running only the `original` branch is a cheap way to screen
prompts for refusals before running the full experiment on them.

## Quick start

You need [uv](https://docs.astral.sh/uv/), Git, curl, CMake 3.20+, a C++17
compiler, about 30 GB of disk and 16 GB of RAM. Linux and WSL2 are tested.

```sh
sudo apt install git curl cmake build-essential   # Ubuntu / WSL2
git clone https://github.com/LucasLista/nla-adversarial-steering.git
cd nla-adversarial-steering
uv run python -m nla.setup        # build pinned nla.cpp, download 12.8 GB of verified weights
uv run python -m nla.server       # http://127.0.0.1:18090
```

With an NVIDIA GPU, run `nla.setup --cuda` and `nla.server --gpu-layers 99`.

The browser stepper lets you go token by token. Inspect the activation, read
its explanation, edit it by hand or with the editor, and compare the four
next-token distributions before choosing a token.

## Running experiments

A cases file is a JSON list of `{"id", "prompt"}` objects, optionally with a
per-case `"instruction"`. We draw prompts from refusal datasets such as
[JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors)
and [DeCCP](https://huggingface.co/datasets/augmxnt/deccp). Inputs and results
stay in the ignored `runs/` directory.

```sh
# Screen: original branch only.
uv run python -m nla.run --cases runs/x/cases.json --variants original --output runs/x-screen

# Full four-branch run; {prompt} in the instruction is replaced by each case's prompt.
uv run python -m nla.run --cases runs/x/cases.json --instruction-file runs/x/instruction.txt \
    --output runs/x --max-tokens 512

uv run python -m nla.run --status runs/x
```

Every backend and editor call is checkpointed under the output directory, so
rerunning the same command resumes the run. The editor is `gpt-6.1-sol`,
either through the Codex CLI with a ChatGPT subscription (`--editor codex`, the
default) or the OpenAI API (`--editor openai`, using `OPENAI_API_KEY`).

### On a cluster

`hpc/` has LSF jobs for DTU's V100 queue. `serve.lsf` serves the API with all
three models resident. `batch.lsf` also runs `nla.run` on the node itself.

```sh
bsub < hpc/serve.lsf    # on the cluster; writes runs/{host,port,token}-JOB.txt
ssh -N -L 18090:NODE:PORT CLUSTER    # on your laptop, with NODE and PORT from those files
uv run python -m nla.run --url http://127.0.0.1:18090 --token-file token.txt ...
```

To use the stepper through the tunnel, open `http://127.0.0.1:18090/?token=TOKEN` once.

## Code

| File | |
| --- | --- |
| [nla/stepper.py](nla/stepper.py) | One step: extract, explain, reconstruct, and the four interventions |
| [nla/trajectories.py](nla/trajectories.py) | Four-branch generation with checkpoints and resume |
| [nla/run.py](nla/run.py) | Batch runner: cases, screening, multiplier sweeps, sharding |
| [nla/editor.py](nla/editor.py) | LLM editor (Codex subscription or OpenAI API) |
| [nla/server.py](nla/server.py) | Stepper API over the three nla.cpp model servers |
| [nla/setup.py](nla/setup.py), [nla/patches.py](nla/patches.py) | Pinned backend build, fixes and weight download |
| [ui/stepper.html](ui/stepper.html) | Browser stepper |

Tests: `uv run python -m unittest discover -s tests`.

## Credits and scope

This project uses the released NLA. It does not train one or introduce a new steering method. Explanations are
hypotheses about activations, and results with quantized (Q4_K_M) inference may
differ from BF16. See the NLA [training code](https://github.com/kitft/natural_language_autoencoders)
and [inference recipe](https://github.com/kitft/nla-inference). The backend is
[nla.cpp](https://github.com/thomasgauthier/nla.cpp) (MIT), pinned and patched to
capture and steer correctly across microbatches. This wrapper is
[MIT-licensed](LICENSE). Checkpoint licenses belong to their publishers.
Development used AI assistance.
