"""Serve the stepper UI and API on top of three nla.cpp model servers.

By default one model (base, av or ar) is loaded at a time to fit a laptop;
--resident keeps all three loaded, which is what cluster jobs use.
"""
import argparse
import atexit
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from nla import stepper

ROOT = Path(__file__).resolve().parents[1]
RUNTIME, MODELS, RUNS = ROOT / '.runtime', ROOT / 'models', ROOT / 'runs'
FILES = json.loads((ROOT / 'config/models.json').read_text())
ROLES = ('base', 'av', 'ar')
# Native failures that a fresh model process recovers from.
TRANSIENT = ('Invalid input batch.', 'Compute error.')


class BackendError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f'Backend HTTP {status}: {message}')
        self.transient = status in (408, 502, 503, 504) or message in TRANSIENT


class Backend:
    def __init__(self, build, port, threads, gpu_layers, context, resident):
        self.build, self.port, self.threads = build, port, threads
        self.gpu_layers, self.context, self.resident = gpu_layers, context, resident
        self.processes = {}  # role -> (process, log)

    def start(self, role):
        if role in self.processes:
            process, log = self.processes[role]
            if process.poll() is None:
                return
            log.close()
            del self.processes[role]
        if not self.resident:
            self.stop()
        row = FILES[role]
        path = MODELS / row['file']
        if not path.exists() or path.stat().st_size != row['size']:
            raise RuntimeError(f"{row['file']} is missing; run python -m nla.setup")
        size = str(self.context if role == 'base' else 1024)
        env = os.environ.copy()
        # WSL must use the Windows driver's CUDA bridge, not a Linux libcuda stub.
        if self.gpu_layers and Path('/usr/lib/wsl/lib/libcuda.so.1').exists():
            env['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + env.get('LD_LIBRARY_PATH', '')
        RUNS.mkdir(exist_ok=True)
        log = open(RUNS / f'{role}.log', 'a')
        process = subprocess.Popen([
            str(self.build / 'bin/llama-server'), '-m', str(path), '--host', '127.0.0.1',
            '--port', str(self.port + ROLES.index(role)), '-ngl', str(self.gpu_layers), '-c', size,
            '-np', '1', '-t', str(self.threads), '--cache-ram', '0', '-fa', 'off',
            # Extraction decodes the whole context as one logical batch. NLA prompts
            # must fit one microbatch: split reconstruction produced NaNs.
            '-b', size, '-ub', '64' if role == 'base' else '1024',
        ], stdout=log, stderr=subprocess.STDOUT, env=env)
        self.processes[role] = (process, log)
        for _ in range(240):
            if process.poll() is not None:
                raise BackendError(503, f'{role} model process exited during startup')
            try:
                self.request(role, 'GET', '/health', None, timeout=2)
                return
            except (OSError, ValueError, BackendError):
                time.sleep(.5)
        raise BackendError(503, f'{role} model did not start')

    def stop(self):
        for process, log in self.processes.values():
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            log.close()
        self.processes.clear()

    def request(self, role, method, endpoint, body, timeout=1200):
        conn = http.client.HTTPConnection('127.0.0.1', self.port + ROLES.index(role), timeout=timeout)
        try:
            conn.request(method, endpoint, json.dumps(body) if body is not None else None,
                         {'Content-Type': 'application/json'})
            response = conn.getresponse()
            result = json.loads(response.read() or b'{}')
        finally:
            conn.close()
        if response.status != 200:
            error = result.get('error') if isinstance(result, dict) else None
            raise BackendError(response.status, error.get('message') if isinstance(error, dict) else error)
        return result

    def __call__(self, role, endpoint, body):
        """POST to a model; restart it on transient failures or invalid numbers, up to 3 tries."""
        for attempt in range(3):
            try:
                self.start(role)
                result = self.request(role, 'POST', endpoint, body)
                if endpoint == '/completion' and body.get('n_probs'):
                    stepper.prediction_token_id(result)
                if endpoint in ('/extract', '/reconstruct'):
                    stepper.validate_activation(result.get('activation'))
                return result
            except (OSError, BackendError, stepper.InvalidActivation, stepper.InvalidPrediction) as exc:
                if attempt == 2 or not getattr(exc, 'transient', True):
                    raise
                self.stop()
                time.sleep(attempt + 1)


def handler(backend, token):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def authenticated(self):
            """Accept a bearer token, or a cookie set by visiting /?token=..."""
            if token is None:
                return True
            cookie = re.search(r'nla_token=([0-9a-f]+)', self.headers.get('Cookie', ''))
            supplied = self.headers.get('Authorization', '').removeprefix('Bearer ') or (cookie and cookie[1]) or ''
            if hmac.compare_digest(supplied, token):
                return True
            query = re.search(r'[?&]token=([0-9a-f]+)', self.path)
            if self.command == 'GET' and query and hmac.compare_digest(query[1], token):
                self.send_response(303)
                self.send_header('Set-Cookie', f'nla_token={token}; HttpOnly; SameSite=Strict; Path=/')
                self.send_header('Location', '/')
                self.end_headers()
                return False
            self.send_response(401)
            self.end_headers()
            return False

        def respond(self, data, content_type='application/json', status=200):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self.authenticated():
                return
            if self.path in ('/', '/stepper'):
                self.respond((ROOT / 'ui/stepper.html').read_bytes(), 'text/html; charset=utf-8')
            elif self.path == '/health':
                self.respond(json.dumps({'ready': all(
                    (MODELS / row['file']).exists() and (MODELS / row['file']).stat().st_size == row['size']
                    for row in FILES.values())}).encode())
            else:
                self.respond(b'{}', status=404)

        def do_POST(self):
            if not self.authenticated():
                return
            if self.path != '/stepper-api':
                return self.respond(b'{}', status=404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
                with lock:
                    result, status = stepper.run(body, backend), 200
            except Exception as exc:
                result, status = {'error': str(exc)}, 400
            self.respond(json.dumps(result).encode(), status=status)

        def log_message(self, *args):
            pass

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=18090)
    p.add_argument('--backend-port', type=int, default=18100, help='first of three ports for the models')
    p.add_argument('--gpu-layers', type=int, default=0, help='99 offloads every layer')
    p.add_argument('--threads', type=int, default=min(8, os.cpu_count() or 4))
    p.add_argument('--context', type=int, default=2048, help='base-model context size')
    p.add_argument('--resident', action='store_true', help='keep all three models loaded')
    p.add_argument('--token-file', type=Path, help='require this bearer token (needed off loopback)')
    args = p.parse_args()
    token = args.token_file.read_text().strip() if args.token_file else None
    if token is not None and len(token) < 32:
        p.error('use a token of at least 32 hex characters')
    if args.host != '127.0.0.1' and token is None:
        p.error('listening off loopback requires --token-file')
    build = RUNTIME / ('build-cuda' if args.gpu_layers else 'build')
    frontend = RUNTIME / 'nla.cpp/frontend/index.html'
    if not (build / 'bin/llama-server').exists() or not frontend.exists():
        p.error('run python -m nla.setup first')
    # The pinned frontend carries the verbalizer's tokenized prompt.
    stepper.AV_INPUT_IDS = [int(x) for x in re.search(r'NLA_ACTOR_INPUT_IDS = `([^`]+)`',
                                                       frontend.read_text())[1].split()]
    stepper.CONTEXT = args.context
    backend = Backend(build, args.backend_port, args.threads, args.gpu_layers, args.context, args.resident)
    atexit.register(backend.stop)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print(f'NLA stepper: http://127.0.0.1:{args.port}', flush=True)
    try:
        ThreadingHTTPServer((args.host, args.port), handler(backend, token)).serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
