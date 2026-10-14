"""Authenticated HPC NLA API; private inference is gated by synthetic validation."""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from hf_backend import Backend
from trajectories import write_json

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--models-root', type=Path, default=ROOT/'.runtime/hf-models')
    parser.add_argument('--listen-host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18097)
    parser.add_argument('--auth-token-file', type=Path)
    parser.add_argument('--job', default='manual')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    profile = next(row for row in json.loads((ROOT/'evaluation_models.json').read_text()) if row['id']==args.model)
    token = args.auth_token_file.read_text().strip() if args.auth_token_file else None
    if args.listen_host != '127.0.0.1' and (token is None or len(token)<32):
        parser.error('Remote binding requires a strong per-job authentication token')
    models = args.models_root/args.model
    progress = json.loads((models/'download-progress.json').read_text())
    if not progress['complete']:
        parser.error('Model snapshot staging is incomplete')
    status_path = ROOT/'.runtime'/f'hf-job-{args.job}.json'
    write_json(status_path, {'phase':'validating','model':args.model})
    backend = Backend(profile, models)
    try:
        validation = backend.validate()
    except Exception as error:
        write_json(status_path, {'phase':'validation_failed','model':args.model,'error_type':type(error).__name__})
        raise SystemExit(1)
    write_json(ROOT/'.runtime'/f'hf-validation-{args.model}-{args.job}.json', validation)
    write_json(status_path, {'phase':'validated','model':args.model,'passed':validation['passed']})
    if args.validate_only:
        return
    manifest = json.loads((models/'snapshot-manifest.json').read_text())
    sizes = {role: sum(item['size'] for item in manifest['files'] if item['role']==role) for role in ('base','av','ar')}
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def authenticated(self):
            if token is None or hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                return True
            self.send_response(401)
            self.end_headers()
            return False
        def send_json(self, value, code=200):
            encoded = json.dumps(value).encode()
            self.send_response(code)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        def do_GET(self):
            if not self.authenticated():return
            if self.path=='/setup-status':
                self.send_json({role:{'bytes':size,'expected':size} for role,size in sizes.items()})
            elif self.path=='/model-status':
                self.send_json(validation)
            else:
                self.send_response(404)
                self.end_headers()
        def do_POST(self):
            if not self.authenticated():return
            if self.path!='/stepper-api':
                self.send_response(404)
                self.end_headers()
                return
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<=4*1024*1024:
                self.send_json({'error':'Invalid request size'},400)
                return
            record=ROOT/'runs'/('hf-'+args.model)/'api'/str(time.time_ns())
            action='invalid'
            try:
                body=json.loads(self.rfile.read(size))
                action=body.pop('action')
                write_json(record.with_suffix('.request.json'),{'action':action,**body})
                with lock:
                    result=backend.run(action,**body)
                write_json(record.with_suffix('.response.json'),result)
                self.send_json(result)
            except Exception as error:
                # Exception messages can contain input/output fragments. Emit only type.
                write_json(status_path,{'phase':'request_failed','action':action,'model':args.model,'error_type':type(error).__name__})
                self.send_json({'error':'HF backend request failed: '+type(error).__name__},400)
        def log_message(self,*args):pass
    write_json(status_path,{'phase':'serving','model':args.model,'passed':True})
    ThreadingHTTPServer((args.listen_host,args.port),Handler).serve_forever()


if __name__=='__main__':main()
