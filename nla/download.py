"""Resumable, range-checked model downloads with SHA256 verification."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'models'
CHUNK = 16 * 1024 * 1024

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while data := source.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()

def download_model(row):
    path = DEST / row['file']
    if path.exists() and path.stat().st_size == row['size']:
        if sha256(path) == row['sha256']:
            return
    parts = DEST / (row['file'] + '.parts')
    parts.mkdir(exist_ok=True)
    count = (row['size'] + CHUNK - 1) // CHUNK
    def part(i):
        start = i * CHUNK
        end = min(start + CHUNK, row['size']) - 1
        target = parts / str(i)
        if target.exists() and target.stat().st_size == end - start + 1:
            return
        url = f"https://huggingface.co/{row['repo']}/resolve/{row['revision']}/{row['file']}?part={i}"
        tmp = parts / f'{i}.tmp'
        headers = parts / f'{i}.headers'
        for attempt in range(5):
            result = subprocess.run(['curl', '-fLsS', '--connect-timeout', '20',
                '--max-time', '300', '-r', f'{start}-{end}', '-D', str(headers),
                url, '-o', str(tmp)])
            valid = f'content-range: bytes {start}-{end}/{row["size"]}'
            if result.returncode == 0 and tmp.stat().st_size == end-start+1 and valid in headers.read_text().lower():
                tmp.replace(target)
                headers.unlink()
                print(f'{row["file"]}: chunk {i+1}/{count}', flush=True)
                return
            time.sleep(2 ** attempt)
        raise RuntimeError(f'Failed chunk {i} of {row["file"]}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(part, range(count)))
    tmp = path.with_suffix('.assembling')
    digest = hashlib.sha256()
    with tmp.open('wb') as out:
        for i in range(count):
            with (parts / str(i)).open('rb') as source:
                while data := source.read(1024 * 1024):
                    digest.update(data)
                    out.write(data)
    if digest.hexdigest() != row['sha256']:
        raise RuntimeError('SHA256 mismatch: ' + row['file'])
    tmp.replace(path)
    for i in range(count):
        (parts / str(i)).unlink()
    parts.rmdir()
    print('VERIFIED ' + row['file'], flush=True)

if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    DEST.mkdir(exist_ok=True)
    rows = json.loads((ROOT / 'config/models.json').read_text())
    for row in rows:
        download_model(row)
