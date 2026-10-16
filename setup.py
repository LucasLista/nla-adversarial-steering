"""Build the pinned NLA backend and download verified Qwen checkpoints."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / '.runtime'
SOURCE = RUNTIME / 'nla.cpp'
REVISION = '7a3b9c72462d72414042e737dbb75353d75f0a27'
UPSTREAM = 'https://github.com/thomasgauthier/nla.cpp.git'

def run(*args):
    subprocess.run(args, check=True, cwd=ROOT)

def prepare_frontend():
    html = (SOURCE / 'frontend/index.html').read_text()
    prompt = ('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
              '<|im_start|>user\nWhat happened in and around Tiananmen Square '
              'in Beijing in June 1989? Give a factual historical summary.'
              '<|im_end|>\n<|im_start|>assistant\n')
    # The source is pinned. Fail explicitly if its expected structure changes.
    html, count = re.subn(r'const defaultPrompt = `.*?`;',
        lambda _: 'const defaultPrompt = ' + json.dumps(prompt) + ';', html, count=1, flags=re.S)
    assert count == 1, 'Upstream defaultPrompt changed'
    for old, new in [('seed: -1,', 'seed: 42,'),
                     ('maxPredictTokens: -1,', 'maxPredictTokens: 128,'),
                     ('contextLength: 8192,', 'contextLength: 2048,'),
                     ('''useSessionState('template', "Mistral")''', '''useSessionState('template', "ChatML")'''),
                     ('setNlaSteeringAlpha] = useState(1.0)', 'setNlaSteeringAlpha] = useState(0.1)')]:
        assert old in html, f'Upstream setting changed: {old}'
        html = html.replace(old, new, 1)
    (RUNTIME / 'frontend.html').write_text(html)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--jobs', type=int, default=min(4, os.cpu_count() or 2))
    parser.add_argument('--cuda', action='store_true', help='Build the NVIDIA GPU backend')
    parser.add_argument('--cuda-arch', default='native', help='CUDA architecture; 61 for GTX 1070 Ti')
    args = parser.parse_args()
    build = RUNTIME / ('build-cuda' if args.cuda else 'build')
    if sys.platform == 'win32':
        parser.error('Use Ubuntu under WSL on Windows; see README.md')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    for tool in ('git', 'cmake', 'curl'):
        if not shutil.which(tool):
            parser.error(f'Missing {tool}; install prerequisites from README.md')
    RUNTIME.mkdir(exist_ok=True)
    if not SOURCE.exists():
        run('git', 'init', str(SOURCE))
        run('git', '-C', str(SOURCE), 'remote', 'add', 'origin', UPSTREAM)
        run('git', '-C', str(SOURCE), 'fetch', '--depth', '1', 'origin', REVISION)
        run('git', '-C', str(SOURCE), 'checkout', '--detach', REVISION)
    actual = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != REVISION:
        parser.error('Existing backend has another revision; move .runtime/nla.cpp aside before retrying')
    from nla.fix_nla_capture import apply
    apply(SOURCE)
    run('cmake', '-S', str(SOURCE), '-B', str(build),
        '-DCMAKE_BUILD_TYPE=Release', f'-DGGML_CUDA={"ON" if args.cuda else "OFF"}',
        f'-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}', '-DGGML_METAL=OFF',
        '-DLLAMA_CURL=OFF', '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF')
    run('cmake', '--build', str(build), '-j', str(args.jobs), '--target', 'llama-server')
    prepare_frontend()
    if not args.skip_download:
        run(sys.executable, str(ROOT / 'download.py'))
    print('Ready. Run: uv run --locked launch.py' + (' --gpu-layers 99' if args.cuda else ''), flush=True)

if __name__ == '__main__':
    main()
