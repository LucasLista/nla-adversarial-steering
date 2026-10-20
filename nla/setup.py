"""Build the pinned NLA backend and download verified Qwen checkpoints."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.runtime'
SOURCE = RUNTIME / 'nla.cpp'
REVISION = '7a3b9c72462d72414042e737dbb75353d75f0a27'
UPSTREAM = 'https://github.com/thomasgauthier/nla.cpp.git'

def prepare_compiler_cache(build, compiler):
    """Call only while holding backend-build.lock."""
    cache = build / 'CMakeCache.txt'
    if not compiler or not cache.exists():
        return
    prefix = 'CMAKE_CXX_COMPILER:FILEPATH='
    cached = next((line[len(prefix):] for line in cache.read_text().splitlines()
                   if line.startswith(prefix)), None)
    if cached and Path(cached).resolve() != Path(compiler).resolve():
        cache.unlink()

def cuda_link_options(runtime, toolkit_library):
    """Resolve CUDA's transitive link dependencies without a runtime stub path."""
    library = Path(toolkit_library).resolve()
    stub = library / 'stubs/libcuda.so'
    if not stub.is_file():
        raise FileNotFoundError('Configured CUDA link stub is unavailable')
    directory = runtime / 'cuda-link-stubs'
    directory.mkdir(exist_ok=True)
    alias = directory / 'libcuda.so.1'
    if alias.is_symlink():
        if alias.resolve() != stub:
            raise RuntimeError('Unexpected CUDA link stub target')
    elif alias.exists():
        raise RuntimeError('Unexpected CUDA link stub file')
    else:
        alias.symlink_to(stub)
    # rpath-link affects ld's dependency lookup only. It emits no runtime RPATH.
    return [f'-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,{library},-rpath-link,{directory}']

def run(*args):
    subprocess.run(args, check=True, cwd=ROOT)

def prepare_frontend():
    html = (SOURCE / 'frontend/index.html').read_text()
    prompt = ('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
              '<|im_start|>user\nWrite a friendly greeting.<|im_end|>\n<|im_start|>assistant\n')
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
    # All allocations share the pinned source and build directory. Serialize
    # patch/configure/build so concurrent startups cannot rewrite each other.
    import fcntl
    build_lock = (RUNTIME / 'backend-build.lock').open('a')
    fcntl.flock(build_lock, fcntl.LOCK_EX)
    prepare_compiler_cache(build, os.environ.get('CXX'))
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
    link_options = (cuda_link_options(RUNTIME, os.environ['NLA_CUDA_LINK_LIB'])
                    if args.cuda and os.environ.get('NLA_CUDA_LINK_LIB') else [])
    run('cmake', '-S', str(SOURCE), '-B', str(build),
        '-DCMAKE_BUILD_TYPE=Release', f'-DGGML_CUDA={"ON" if args.cuda else "OFF"}',
        f'-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}', '-DGGML_METAL=OFF',
        '-DLLAMA_CURL=OFF', '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF', *link_options)
    run('cmake', '--build', str(build), '-j', str(args.jobs), '--target', 'llama-server')
    prepare_frontend()
    if not args.skip_download:
        run(sys.executable, '-m', 'nla.download')
    print('Ready. Run: uv run --locked python -m nla.launch' + (' --gpu-layers 99' if args.cuda else ''), flush=True)

if __name__ == '__main__':
    main()
