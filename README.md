# NLA playground

Inspect and edit Qwen2.5-7B-Instruct's activations using Anthropic's released
Natural Language Autoencoder (NLA). Hover a token, read an explanation, edit
the text, and apply the reconstructed edit direction to generation.

This packages [Thomas Gauthier's nla.cpp](https://github.com/thomasgauthier/nla.cpp)
and its Mikupad interface with a laptop-oriented launcher. It loads **one model
at a time**, on CPU, so you do not need a GPU, API key, or another person's PC.
It does not train an NLA or claim a new steering method.

## Requirements

- Python 3.10+, Git, curl, CMake 3.20+, and a C++17 compiler.
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
sudo apt install -y python3 git curl cmake build-essential
```

On macOS, install Apple's command-line tools (`xcode-select --install`),
Python 3, and CMake. If you already use Homebrew: `brew install python cmake`.
On Windows, run all commands inside Ubuntu WSL2, and keep the project in the
Linux filesystem, such as `~/nla-playground`, rather than `/mnt/c`.

```sh
python3 setup.py
python3 launch.py
```

Open **http://127.0.0.1:18090** in your browser. Leave the terminal open.
Ctrl+C stops the launcher and the model process it owns. Models remain on disk.
Setup can be rerun; downloads resume from completed 16 MB chunks and each final
file is checked against its pinned SHA-256 hash. No weights are committed here.

Optional: `python3 launch.py --threads 4 --port 18090 --backend-port 18089`.
Use distinct unused ports; fewer threads may keep the laptop more responsive.
Run `python3 smoke_test.py` while the launcher is open to exercise all model roles.
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
