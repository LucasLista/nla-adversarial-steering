#!/usr/bin/env python3
"""NLA proxy with optional authenticated HPC access. Loads one model at a time."""
import atexit
import hmac
import argparse
import os
import socket
import http.client
import json
import pathlib
import subprocess
import signal
import sys
import threading
import time
import stepper
import experiments
import prompt_catalog
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parent
MODELS = ROOT / 'models'
RUNTIME = ROOT / '.runtime'
LOGS = ROOT / 'runs'
PORT = 18090
BACKEND_PORT = 18089
THREADS = min(8, os.cpu_count() or 4)
GPU_LAYERS = 0
BUILD_DIR = RUNTIME / 'build'
AUTH_TOKEN = None
AUDIT_LOG = False

def audit(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as out:
        json.dump(value, out, ensure_ascii=False, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)
FILES = {
    'base': ('Qwen2.5-7B-Instruct-Q4_K_M.gguf', 4683074240),
    'av': ('nla-qwen2.5-7b-L20-av-Q4_K_M.gguf', 4683073728),
    'ar': ('nla-qwen2.5-7b-L20-ar-Q4_K_M.gguf', 3412340480),
}
lock = threading.Lock()
process = None
active = None
logfile = None
RESIDENT = os.environ.get('NLA_RESIDENT_MODELS') == '1'
resident = {}

def stop():
    global process, active, logfile
    handles = list(resident.values()) + ([(process, logfile)] if process is not None else [])
    resident.clear()
    for child, log in handles:
        if child is not process:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
            if log: log.close()
    if process is not None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if logfile:
        logfile.close()
    process, active, logfile = None, None, None

atexit.register(stop)
signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

def ensure(role):
    global process, active, logfile, BACKEND_PORT
    if RESIDENT:
        if active and process:
            resident[active] = (process, logfile)
        BACKEND_PORT = int(os.environ.get('NLA_BACKEND_BASE','18089')) + {'base':0, 'av':1, 'ar':2}[role]
        if role in resident:
            process, logfile = resident[role]
            active = role
            if process.poll() is None: return
            if logfile: logfile.close()
            del resident[role]
        process, logfile, active = None, None, None
    if active == role and process and process.poll() is None:
        return
    name, size = FILES[role]
    path = MODELS / name
    if not path.exists() or path.stat().st_size != size:
        raise RuntimeError(f'{name} is still downloading. Try again after setup completes.')
    if not RESIDENT: stop()
    logfile = open(LOGS / f'{role}.log', 'a')
    child_env = os.environ.copy()
    # WSL must use the Windows driver's CUDA bridge, not a Linux libcuda stub.
    if GPU_LAYERS and pathlib.Path('/usr/lib/wsl/lib/libcuda.so.1').exists():
        child_env['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + child_env.get('LD_LIBRARY_PATH', '')
    process = subprocess.Popen([
        str(BUILD_DIR / 'bin/llama-server'), '-m', str(path),
        '--host', '127.0.0.1', '--port', str(BACKEND_PORT), '-ngl', str(GPU_LAYERS),
        '-c', os.environ.get('NLA_CONTEXT_SIZE','2048') if role == 'base' else '1024', '-np', '1',
        '-t', str(THREADS), '--cache-ram', '0',
        # NLA extraction decodes the full context in one logical batch.
        # Physical microbatches stay small to limit GPU working memory.
        '-b', os.environ.get('NLA_CONTEXT_SIZE','2048') if role == 'base' else '1024',
        '-ub', '64', '-fa', 'off',
    ], stdout=logfile, stderr=subprocess.STDOUT, env=child_env)
    for _ in range(240):
        if process.poll() is not None:
            raise RuntimeError(f'{role} failed to start; see {role}.log')
        try:
            conn = http.client.HTTPConnection('127.0.0.1', BACKEND_PORT, timeout=2)
            conn.request('GET', '/health')
            response = conn.getresponse()
            response.read()
            conn.close()
            if response.status == 200:
                active = role
                return
        except OSError:
            pass
        time.sleep(.5)
    stop()
    raise RuntimeError('Model loading timed out')

def backend_call(role, endpoint, body):
    ensure(role)
    conn = http.client.HTTPConnection('127.0.0.1', BACKEND_PORT, timeout=1200)
    try:
        conn.request('POST', endpoint, json.dumps(body), {'Content-Type':'application/json'})
        response = conn.getresponse()
        result = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(str(result))
        return result
    finally:
        conn.close()

class Handler(BaseHTTPRequestHandler):
    def authenticated(self):
        if AUTH_TOKEN is None or hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + AUTH_TOKEN):
            return True
        self.send_response(401)
        self.end_headers()
        return False

    def do_GET(self):
        if not self.authenticated():
            return
        if self.path == '/favicon.ico':
            self.send_response(204)
            self.end_headers()
        elif self.path in ('/prompt-catalog', '/editor-prompt-catalog'):
            try:
                filename = 'editor_prompts.json' if self.path == '/editor-prompt-catalog' else 'prompts.json'
                data = json.dumps(prompt_catalog.load(ROOT / filename)).encode()
            except ValueError:
                self.send_error(500, 'Invalid prompt catalog')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)
        elif self.path in ('/stepper', '/stepper.html'):
            data = (ROOT / 'stepper.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(data)
        elif self.path in ('/', '/index.html'):
            data = (RUNTIME / 'frontend.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(data)
        elif self.path == '/setup-status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({role: {
                'bytes': (MODELS / name).stat().st_size if (MODELS / name).exists() else 0,
                'expected': size,
            } for role, (name, size) in FILES.items()}).encode())
        else:
            self.proxy()

    def do_POST(self):
        if not self.authenticated():
            return
        if self.path != '/stepper-api':
            return self.proxy()
        started = time.time()
        record = LOGS / 'api' / str(time.time_ns())
        try:
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            if AUDIT_LOG:
                audit(record.with_suffix('.request.json'), {'request':body, 'started_at':started})
            with lock:
                if body.get('action') in ('experiments','experiment-input','results','save-result','open-result'):
                    result = experiments.run(LOGS, body)
                elif body.get('action') == 'save':
                    name = f'token-session-{time.time_ns()}.json'
                    (LOGS / name).write_text(json.dumps(body['session'], ensure_ascii=False, indent=2))
                    result = {'saved': str(LOGS / name)}
                else:
                    result = stepper.run(body, backend_call, (RUNTIME / 'frontend.html').read_text())
            status = 200
        except Exception as exc:
            result = {'error':str(exc)}
            status = 400
        if AUDIT_LOG:
            audit(record.with_suffix('.response.json'), {'response':result, 'status':status,
                                                       'finished_at':time.time()})
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def proxy(self):
        role = 'av' if self.path == '/explain' else 'ar' if self.path in (
            '/score', '/reconstruct', '/edit-direction') else 'base'
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        started = False
        with lock:
            conn = None
            try:
                ensure(role)
                conn = http.client.HTTPConnection('127.0.0.1', BACKEND_PORT, timeout=1200)
                conn.request(self.command, self.path, body=body,
                             headers={'Content-Type': 'application/json'})
                response = conn.getresponse()
                self.send_response(response.status)
                self.send_header('Content-Type', response.getheader('Content-Type', 'application/json'))
                self.end_headers()
                started = True
                while chunk := response.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                if not started:
                    self.send_response(503)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': {'message': str(exc)}}).encode())
            finally:
                if conn:
                    conn.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--listen-host', default='127.0.0.1')
    parser.add_argument('--auth-token-file', type=pathlib.Path)
    parser.add_argument('--audit-log', action='store_true', help='Save every stepper API request and response before sending it')
    parser.add_argument('--port', type=int, default=18090)
    parser.add_argument('--backend-port', type=int, default=18089)
    parser.add_argument('--threads', type=int, default=THREADS)
    parser.add_argument('--gpu-layers', type=int, default=0, help='GPU-offloaded layers; 99 offloads all layers')
    parser.add_argument('--build-dir', type=pathlib.Path, help='Backend build directory (defaults to .runtime/build)')
    args = parser.parse_args()
    AUDIT_LOG = args.audit_log or args.auth_token_file is not None
    if args.auth_token_file:
        AUTH_TOKEN = args.auth_token_file.read_text().strip()
        if len(AUTH_TOKEN) < 32:
            parser.error('Use an authentication token with at least 32 characters')
    if args.listen_host != '127.0.0.1' and AUTH_TOKEN is None:
        parser.error('Non-loopback listeners require --auth-token-file')
    GPU_LAYERS = args.gpu_layers
    BUILD_DIR = args.build_dir.resolve() if args.build_dir else RUNTIME / ('build-cuda' if GPU_LAYERS else 'build')
    if GPU_LAYERS < 0:
        parser.error('GPU layers must be nonnegative')
    PORT, BACKEND_PORT, THREADS = args.port, args.backend_port, args.threads
    os.environ["NLA_BACKEND_BASE"] = str(BACKEND_PORT)
    if THREADS < 1 or PORT == BACKEND_PORT:
        parser.error('Use positive threads and distinct ports')
    if not (BUILD_DIR / 'bin/llama-server').exists() or not (RUNTIME / 'frontend.html').exists():
        parser.error('Run uv run --locked setup.py first')
    # Refuse an occupied model-server port, rather than attaching to an unrelated server.
    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', BACKEND_PORT))
        except OSError:
            parser.error(f'Backend port {BACKEND_PORT} is in use; choose --backend-port')
    LOGS.mkdir(exist_ok=True)
    print(f'NLA playground: http://127.0.0.1:{PORT}', flush=True)
    try:
        ThreadingHTTPServer((args.listen_host, PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
