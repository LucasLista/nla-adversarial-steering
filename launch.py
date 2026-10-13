#!/usr/bin/env python3
"""Loopback-only NLA proxy. Loads one model at a time to limit laptop RAM use."""
import atexit
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parent
MODELS = ROOT / 'models'
RUNTIME = ROOT / '.runtime'
LOGS = ROOT / 'runs'
PORT = 18090
BACKEND_PORT = 18089
THREADS = min(8, os.cpu_count() or 4)
FILES = {
    'base': ('Qwen2.5-7B-Instruct-Q4_K_M.gguf', 4683074240),
    'av': ('nla-qwen2.5-7b-L20-av-Q4_K_M.gguf', 4683073728),
    'ar': ('nla-qwen2.5-7b-L20-ar-Q4_K_M.gguf', 3412340480),
}
lock = threading.Lock()
process = None
active = None
logfile = None

def stop():
    global process, active, logfile
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
    global process, active, logfile
    if active == role and process and process.poll() is None:
        return
    name, size = FILES[role]
    path = MODELS / name
    if not path.exists() or path.stat().st_size != size:
        raise RuntimeError(f'{name} is still downloading. Try again after setup completes.')
    stop()
    logfile = open(LOGS / f'{role}.log', 'a')
    process = subprocess.Popen([
        str(RUNTIME / 'build/bin/llama-server'), '-m', str(path),
        '--host', '127.0.0.1', '--port', str(BACKEND_PORT), '-ngl', '0',
        '-c', '2048' if role == 'base' else '1024', '-np', '1',
        '-t', str(THREADS), '--cache-ram', '0',
    ], stdout=logfile, stderr=subprocess.STDOUT)
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

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/favicon.ico':
            self.send_response(204)
            self.end_headers()
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
        self.proxy()

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
    parser.add_argument('--port', type=int, default=18090)
    parser.add_argument('--backend-port', type=int, default=18089)
    parser.add_argument('--threads', type=int, default=THREADS)
    args = parser.parse_args()
    PORT, BACKEND_PORT, THREADS = args.port, args.backend_port, args.threads
    if THREADS < 1 or PORT == BACKEND_PORT:
        parser.error('Use positive threads and distinct ports')
    if not (RUNTIME / 'build/bin/llama-server').exists() or not (RUNTIME / 'frontend.html').exists():
        parser.error('Run python3 setup.py first')
    # Refuse an occupied model-server port, rather than attaching to an unrelated server.
    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', BACKEND_PORT))
        except OSError:
            parser.error(f'Backend port {BACKEND_PORT} is in use; choose --backend-port')
    LOGS.mkdir(exist_ok=True)
    print(f'NLA playground: http://127.0.0.1:{PORT}', flush=True)
    try:
        ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
