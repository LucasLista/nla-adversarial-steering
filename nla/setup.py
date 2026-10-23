"""Build the pinned nla.cpp backend and download hash-verified Qwen NLA checkpoints."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from nla.patches import apply

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.runtime'
MODELS = ROOT / 'models'
SOURCE = RUNTIME / 'nla.cpp'
REVISION = '7a3b9c72462d72414042e737dbb75353d75f0a27'
UPSTREAM = 'https://github.com/thomasgauthier/nla.cpp.git'
CHUNK = 16 * 1024 * 1024


def run(*args):
    subprocess.run(args, check=True, cwd=ROOT)


def cuda_link_options(toolkit_library):
    """Link against CUDA's driver stub without baking a stub path into the binary."""
    library = Path(toolkit_library).resolve()
    stub = library / 'stubs/libcuda.so'
    alias = RUNTIME / 'cuda-link-stubs/libcuda.so.1'
    alias.parent.mkdir(exist_ok=True)
    if not alias.is_symlink():
        alias.symlink_to(stub)
    return [f'-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,{library},-rpath-link,{alias.parent}']


def build(cuda, arch, jobs):
    if not SOURCE.exists():
        run('git', 'init', str(SOURCE))
        run('git', '-C', str(SOURCE), 'remote', 'add', 'origin', UPSTREAM)
        run('git', '-C', str(SOURCE), 'fetch', '--depth', '1', 'origin', REVISION)
        run('git', '-C', str(SOURCE), 'checkout', '--detach', REVISION)
    head = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if head != REVISION:
        sys.exit(f'{SOURCE} is at another revision; move it aside and retry')
    apply(SOURCE)
    out = RUNTIME / ('build-cuda' if cuda else 'build')
    # A cached compiler path from another host or module breaks reconfiguration.
    cache = out / 'CMakeCache.txt'
    if cache.exists() and os.environ.get('CXX') and f"CMAKE_CXX_COMPILER:FILEPATH={os.environ['CXX']}" not in cache.read_text():
        cache.unlink()
    link = cuda_link_options(os.environ['NLA_CUDA_LINK_LIB']) if cuda and os.environ.get('NLA_CUDA_LINK_LIB') else []
    run('cmake', '-S', str(SOURCE), '-B', str(out), '-DCMAKE_BUILD_TYPE=Release',
        f'-DGGML_CUDA={"ON" if cuda else "OFF"}', f'-DCMAKE_CUDA_ARCHITECTURES={arch}', '-DGGML_METAL=OFF',
        '-DLLAMA_CURL=OFF', '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF', *link)
    run('cmake', '--build', str(out), '-j', str(jobs), '--target', 'llama-server')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while data := f.read(1 << 20):
            digest.update(data)
    return digest.hexdigest()


def download(row):
    """Resumable ranged download in 16 MB chunks, verified against the pinned SHA256."""
    path = MODELS / row['file']
    if path.exists() and path.stat().st_size == row['size'] and sha256(path) == row['sha256']:
        return
    parts = MODELS / (row['file'] + '.parts')
    parts.mkdir(parents=True, exist_ok=True)
    count = -(-row['size'] // CHUNK)

    def part(i):
        start, end = i * CHUNK, min((i+1) * CHUNK, row['size']) - 1
        target, tmp = parts / str(i), parts / f'{i}.tmp'
        if target.exists() and target.stat().st_size == end - start + 1:
            return
        url = f"https://huggingface.co/{row['repo']}/resolve/{row['revision']}/{row['file']}"
        for attempt in range(5):
            ok = subprocess.run(['curl', '-fLsS', '--connect-timeout', '20', '--max-time', '300',
                                 '-r', f'{start}-{end}', url, '-o', str(tmp)]).returncode == 0
            if ok and tmp.stat().st_size == end - start + 1:
                tmp.replace(target)
                return
            time.sleep(2 ** attempt)
        raise RuntimeError(f'Failed chunk {i} of {row["file"]}')

    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        list(pool.map(part, range(count)))
    tmp = path.with_suffix('.assembling')
    with tmp.open('wb') as out:
        for i in range(count):
            out.write((parts / str(i)).read_bytes())
    if sha256(tmp) != row['sha256']:
        raise RuntimeError('SHA256 mismatch: ' + row['file'])
    tmp.replace(path)
    shutil.rmtree(parts)
    print('verified', row['file'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cuda', action='store_true', help='build the NVIDIA GPU backend')
    p.add_argument('--cuda-arch', default='native', help='e.g. 70 for V100')
    p.add_argument('--jobs', type=int, default=min(4, os.cpu_count() or 2))
    p.add_argument('--skip-download', action='store_true')
    args = p.parse_args()
    for tool in ('git', 'cmake', 'curl'):
        if not shutil.which(tool):
            p.error(f'missing {tool}')
    RUNTIME.mkdir(exist_ok=True)
    # Concurrent cluster jobs share one checkout and build directory.
    with (RUNTIME / 'backend-build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        build(args.cuda, args.cuda_arch, args.jobs)
    if not args.skip_download:
        for row in json.loads((ROOT / 'config/models.json').read_text()).values():
            download(row)
    print('Ready: uv run python -m nla.server' + (' --gpu-layers 99' if args.cuda else ''))


if __name__ == '__main__':
    main()
