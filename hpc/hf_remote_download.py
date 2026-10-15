"""HPC-only snapshot downloader; receives no Hugging Face credential."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    plan = json.load(sys.stdin)
    destination = Path(plan['destination'])
    destination.mkdir(parents=True, exist_ok=True)
    completed = 0
    for index, item in enumerate(plan['files']):
        path = destination / item['role'] / item['file']
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = item.get('sha256')
        def verified(candidate):
            if not candidate.is_file() or candidate.stat().st_size != item['size']:
                return False
            if not digest:
                return True
            hasher = hashlib.sha256()
            with candidate.open('rb') as source:
                while chunk := source.read(8 * 1024 * 1024):
                    hasher.update(chunk)
            return hasher.hexdigest() == digest
        if not verified(path):
            temporary = path.with_suffix(path.suffix + '.partial')
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset > item['size']:
                temporary.unlink()
                offset = 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            request = urllib.request.Request(plan['proxy'] + '/' + str(index), headers=headers)
            with urllib.request.urlopen(request, timeout=600) as response:
                append = offset and response.status == 206
                if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Invalid snapshot download range')
                with temporary.open('ab' if append else 'wb') as target:
                    while chunk := response.read(1024 * 1024):
                        target.write(chunk)
            if not verified(temporary):
                raise ValueError('Snapshot file failed size/hash verification')
            temporary.replace(path)
        completed += 1
        atomic(destination / 'download-progress.json', {
            'model': plan['model'], 'files_complete': completed, 'files_total': len(plan['files']),
            'complete': completed == len(plan['files']),
        })
    atomic(destination / 'snapshot-manifest.json', {
        'model': plan['model'], 'roles': plan['roles'], 'files': plan['files'],
    })


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # URLs, HTTP bodies and credentials must never leak to stdout/stderr.
        print('Snapshot staging failed:', type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
