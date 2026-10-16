"""Stream approved HF snapshots to HPC through SSH without moving credentials."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import subprocess
import threading
import urllib.request
from hpc.model_preflight import credential

ROOT = Path(__file__).resolve().parents[1]


def manifest(profile):
    files = []
    token = credential()
    for role, row in profile['roles'].items():
        url = f"https://huggingface.co/api/models/{row['repo']}/revision/{row['revision']}?blobs=true"
        request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token} if token else {})
        with urllib.request.urlopen(request, timeout=30) as response:
            metadata = json.load(response)
        if metadata['sha'] != row['revision']:
            raise ValueError('Snapshot revision mismatch')
        for item in metadata['siblings']:
            filename = item['rfilename']
            if '/' in filename or not (filename.endswith(('.json', '.safetensors', '.model', '.yaml', '.jinja', '.txt'))):
                continue
            lfs = item.get('lfs', {})
            files.append({'role': role, 'file': filename, 'size': item['size'],
                          'sha256': lfs.get('sha256'),
                          'url': f"https://huggingface.co/{row['repo']}/resolve/{row['revision']}/{filename}"})
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--ssh-host', required=True)
    parser.add_argument('--ssh-socket', required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--port', type=int, default=18556)
    args = parser.parse_args()
    profiles = json.loads((ROOT / 'config/evaluation_models.json').read_text())
    profile = next((row for row in profiles if row['id'] == args.model), None)
    if profile is None:
        parser.error('Unknown model family')
    files = manifest(profile)
    token = credential()
    class Gateway(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                index = int(self.path.removeprefix('/'))
                if index < 0 or index >= len(files):
                    self.send_error(404)
                    return
                headers = {'Authorization': 'Bearer ' + token} if token else {}
                if self.headers.get('Range'):
                    headers['Range'] = self.headers['Range']
                request = urllib.request.Request(files[index]['url'], headers=headers)
                with urllib.request.urlopen(request, timeout=600) as response:
                    self.send_response(response.status)
                    for header in ('Content-Length', 'Content-Range', 'Content-Type'):
                        if response.headers.get(header):
                            self.send_header(header, response.headers[header])
                    self.end_headers()
                    while chunk := response.read(1024 * 1024):
                        self.wfile.write(chunk)
            except Exception:
                # Never relay upstream error bodies or credential-bearing redirects.
                self.close_connection = True
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Gateway)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ssh = ['ssh', '-S', args.ssh_socket, '-o', 'BatchMode=yes']
    forwarding = ['-R', f'127.0.0.1:{args.port}:127.0.0.1:{args.port}']
    subprocess.run(ssh + ['-O', 'forward', *forwarding, args.ssh_host], check=True, capture_output=True)
    destination = args.remote_root + '/.runtime/hf-models/' + args.model
    plan = {'model': args.model, 'destination': destination, 'proxy': f'http://127.0.0.1:{args.port}',
            'roles': profile['roles'], 'files': [{key: value for key, value in item.items() if key != 'url'} for item in files]}
    try:
        command = 'python3 ' + shlex.quote(args.remote_root + '/hpc/hf_remote_download.py')
        result = subprocess.run(ssh + [args.ssh_host, command], input=json.dumps(plan),
                                text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode:
            raise RuntimeError('HPC snapshot staging failed; check sanitized progress metadata')
        print('Snapshot staged on HPC:', args.model, 'files:', len(files), flush=True)
    finally:
        subprocess.run(ssh + ['-O', 'cancel', *forwarding, args.ssh_host], capture_output=True)
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
