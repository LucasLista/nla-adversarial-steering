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
import re
import subprocess
import signal
import sys
import threading
import time
import traceback
from nla import stepper, experiments, prompt_catalog
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
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

# Keep operational failure metadata useful without exposing backend error bodies.
BackendHTTPError = type('BackendHTTPError', (RuntimeError,), {})
BackendStartupFailure = type('BackendStartupFailure', (RuntimeError,), {})
BackendInvalidInputBatch = type('BackendInvalidInputBatch', (BackendHTTPError,), {})
BackendComputeError = type('BackendComputeError', (BackendHTTPError,), {})
BackendContextExceeded = type('BackendContextExceeded', (BackendHTTPError,), {})
BACKEND_HTTP_ERRORS = {
    status: type(f'BackendHTTP{status}', (BackendHTTPError,), {})
    for status in (400, 401, 403, 404, 408, 409, 422, 429, 500, 502, 503, 504)
}

def backend_http_error(status, result):
    """Classify only fixed native error constants; never expose error bodies.

    These exact constants identify llama_decode return paths in the pinned
    server-context.cpp. Unrecognized messages stay an opaque HTTP failure.
    """
    error = result.get('error') if isinstance(result, dict) else None
    message = error.get('message') if isinstance(error, dict) else None
    native = {
        'Invalid input batch.': BackendInvalidInputBatch,
        'Compute error.': BackendComputeError,
        'Context size has been exceeded.': BackendContextExceeded,
    }
    classified = native.get(message) if status == 500 and isinstance(message, str) else None
    # Native nlohmann exceptions carry a stable class/id prefix. Preserve only
    # that identifier in health metadata, never its message or offending value.
    if classified is None and status == 500 and isinstance(message, str):
        matched = re.match(r'^\[json\.exception\.(parse_error|invalid_iterator|type_error|out_of_range|other_error)\.([1-5][0-9]{2})\]', message)
        if matched:
            category, identifier = matched.groups()
            name = 'BackendJSON' + ''.join(part.title() for part in category.split('_')) + identifier
            classified = type(name, (BackendHTTPError,), {})
        if classified is None:
            from nla.native_error_source import source_frame
            frame = source_frame(RUNTIME / 'nla.cpp', message)
            if frame is not None:
                classified = type('BackendNativeRuntimeError', (BackendHTTPError,),
                                  {'native_frames': [frame]})
    return classified or BACKEND_HTTP_ERRORS.get(status, BackendHTTPError)

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
    child_env['NLA_NATIVE_HEALTH_DIR'] = str(RUNTIME)
    child_env['NLA_NATIVE_ROLE'] = role
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
        # NLA prompts (<=1024 tokens) decode in one microbatch: multi-microbatch
        # reconstruction produced NaNs for some explanations.
        '-ub', '64' if role == 'base' else '1024', '-fa', 'off',
    ], stdout=logfile, stderr=subprocess.STDOUT, env=child_env)
    for _ in range(240):
        if process.poll() is not None:
            code = process.returncode
            suffix = f'Signal{-code}' if code < 0 else f'Exit{code}'
            kind = type('Backend' + role.title() + 'Process' + suffix, (BackendStartupFailure,), {})
            raise kind('Native backend process stopped during initialization')
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

def _backend_call_once(role, endpoint, body):
    ensure(role)
    conn = http.client.HTTPConnection('127.0.0.1', BACKEND_PORT, timeout=1200)
    try:
        conn.request('POST', endpoint, json.dumps(body), {'Content-Type':'application/json'})
        response = conn.getresponse()
        result = json.loads(response.read())
        if response.status != 200:
            raise backend_http_error(response.status, result)(
                'Backend HTTP request failed')
        return result
    finally:
        conn.close()


def backend_call(role, endpoint, body):
    # Reload transient invalid numerical state, retaining identical inputs and
    # model arguments. Never replace invalid values or accept a partial result.
    invalid = (stepper.InvalidActivationType, stepper.NonFiniteActivation,
               stepper.InvalidActivationShape, stepper.InvalidPredictionResponse)
    for attempt in range(3):
        try:
            result = _backend_call_once(role, endpoint, body)
        except (BackendStartupFailure, BackendInvalidInputBatch, BackendComputeError,
                *(BACKEND_HTTP_ERRORS[status] for status in (408, 502, 503, 504))):
            if attempt == 2:
                raise
            stop()
            time.sleep(attempt + 1)
            continue
        if endpoint == '/completion' and body.get('n_probs'):
            try:
                stepper.validate_prediction(result)
                return result
            except stepper.InvalidPredictionResponse:
                if attempt == 2:
                    raise
                stop()
            continue
        if endpoint not in ('/extract', '/reconstruct'):
            return result
        try:
            stepper.validate_activation(result.get('activation'))
            return result
        except invalid:
            if attempt == 2:
                raise
            stop()
    raise AssertionError('Unreachable numerical recovery state')

class Handler(BaseHTTPRequestHandler):
    def authenticated(self):
        if AUTH_TOKEN is None or hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + AUTH_TOKEN):
            return True
        self.send_response(401)
        self.end_headers()
        return False

    def respond(self, data, content_type='application/json', status=200, no_store=False):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        if no_store:
            self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.authenticated():
            return
        if self.path == '/favicon.ico':
            self.send_response(204)
            self.end_headers()
        elif self.path in ('/prompt-catalog', '/editor-prompt-catalog'):
            try:
                path = (ROOT / 'runs/experiments/editor_prompts.json' if self.path == '/editor-prompt-catalog'
                        else ROOT / os.environ.get('NLA_BENCHMARK_CATALOG', 'runs/benchmarks/jbb-benign/catalog.json'))
                data = json.dumps(prompt_catalog.load(path) if path.exists() else []).encode()
            except ValueError:
                self.send_error(500, 'Invalid prompt catalog')
                return
            self.respond(data, no_store=True)
        elif self.path in ('/stepper', '/stepper.html'):
            data = (ROOT / 'ui/stepper.html').read_bytes()
            self.respond(data, 'text/html; charset=utf-8')
        elif self.path in ('/', '/index.html'):
            data = (RUNTIME / 'frontend.html').read_bytes()
            self.respond(data, 'text/html; charset=utf-8')
        elif self.path == '/setup-status':
            self.respond(json.dumps({role: {
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
            # Diagnostic metadata excludes exception messages, locals and payloads.
            job_id = os.environ.get('LSB_JOBID', 'local')
            if not job_id.isdigit():
                job_id = 'local'
            health = {'error_type': type(exc).__name__, 'frames': [
                {'file': pathlib.Path(frame.filename).name,
                 'function': frame.name, 'line': frame.lineno}
                for frame in traceback.extract_tb(exc.__traceback__)]}
            health['frames'].extend(getattr(exc, 'native_frames', []))
            audit(RUNTIME / f'backend-health-{job_id}.json', health)
            result = {'error':str(exc)}
            status = 400
        if AUDIT_LOG:
            audit(record.with_suffix('.response.json'), {'response':result, 'status':status,
                                                       'finished_at':time.time()})
        self.respond(json.dumps(result).encode(), status=status)

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
                    self.respond(json.dumps({'error': {'message': str(exc)}}).encode(), status=503)
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
        parser.error('Run uv run --locked python -m nla.setup first')
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
