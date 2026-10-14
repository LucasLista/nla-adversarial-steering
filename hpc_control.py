"""Wait for an LSF server, tunnel its authenticated API, then run four trajectories."""
import argparse
import atexit
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time
import urllib.request
import trajectories
import codex_editor
import prompt_catalog

def select_preset(path, identifier):
    matches = [entry['prompt'] for entry in prompt_catalog.load(path) if entry['id'] == identifier]
    if len(matches) != 1:
        raise ValueError('Preset ID must match exactly one catalog entry')
    return matches[0]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',required=True,type=int)
    p.add_argument('--model',default='qwen2.5-7b',help='Evaluated model family from evaluation_models.json')
    p.add_argument('--ssh-host',required=True)
    p.add_argument('--ssh-socket',required=True)
    p.add_argument('--connect-script',type=Path,help='Optional existing local SSH authentication helper')
    p.add_argument('--remote-root',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--prompt')
    p.add_argument('--probe-id',help='Select a probe by ID without putting its text in command arguments')
    p.add_argument('--editor-preset',help='Select an editor instruction by ID')
    p.add_argument('--configs',nargs='+',type=Path)
    p.add_argument('--port',type=int,default=18095)
    p.add_argument('--instruction')
    p.add_argument('--max-tokens',type=int,default=64)
    p.add_argument('--keep-serving',action='store_true',help='Keep the browser relay open while the HPC job is running')
    args=p.parse_args()
    models=json.loads((Path(__file__).parent/'evaluation_models.json').read_text())
    profile=next((item for item in models if item['id']==args.model),None)
    if profile is None: p.error('Unknown evaluated model family')
    if profile['runtime'] != 'gguf-qwen':
        p.error('This model requires a validated model-specific backend adapter; run model_preflight.py')
    if args.probe_id:
        if args.prompt or args.configs: p.error('--probe-id cannot be combined with --prompt or --configs')
        args.prompt = select_preset(Path(__file__).parent / 'prompts.json', args.probe_id)
    if args.editor_preset:
        if args.instruction or args.configs: p.error('--editor-preset cannot be combined with --instruction or --configs')
        args.instruction = select_preset(Path(__file__).parent / 'editor_prompts.json', args.editor_preset)
    if not args.configs and (not args.prompt or not args.instruction): p.error('Provide --configs, preset IDs, or --prompt and --instruction')
    if not 1<=args.max_tokens<=10000: p.error('Token limit must be 1–10000')
    ssh=['ssh','-S',args.ssh_socket,'-o','BatchMode=yes']
    def remote(cmd):
        for attempt in range(5):
            try:
                return subprocess.check_output(ssh+[args.ssh_host,'bash -lc '+shlex.quote(cmd)],text=True,timeout=30).strip()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if attempt == 4: raise
                print('SSH interrupted; reconnecting before retry',flush=True)
                if args.connect_script:
                    subprocess.run(['uv','run','--with','pexpect',str(args.connect_script)],check=True,timeout=60)
                time.sleep(5)
    root=shlex.quote(args.remote_root)
    args.output.mkdir(parents=True,exist_ok=True)
    while True:
        state=remote(f'bjobs -noheader -o stat {args.job}').strip()
        if state not in ('PEND','RUN','WAIT'):
            raise RuntimeError(f'HPC job {args.job} stopped: {state}')
        host=remote(f'test -f {root}/runs/host-{args.job}.txt && cat {root}/runs/host-{args.job}.txt || true')
        token=remote(f'test -f {root}/runs/token-{args.job}.txt && cat {root}/runs/token-{args.job}.txt || true')
        if host and token: break
        print(f'Job {args.job}: {state}; waiting for backend build',flush=True)
        time.sleep(30)
    subprocess.run(ssh+['-O','forward','-L',f'127.0.0.1:19090:{host}:18090',args.ssh_host],check=True,capture_output=True)
    class Relay(BaseHTTPRequestHandler):
        def relay(self):
            conn=http.client.HTTPConnection('127.0.0.1',19090,timeout=1200)
            try:
                body=self.rfile.read(int(self.headers.get('Content-Length',0)))
                if self.command == 'POST' and self.path == '/stepper-api':
                    payload=json.loads(body)
                    if payload.get('action') == 'propose' and payload.get('editor') == 'codex':
                        try:
                            result=codex_editor.propose(payload['instruction'],payload['text']);status=200
                        except Exception as exc:
                            result={'error':str(exc)};status=400
                        data=json.dumps(result).encode()
                        self.send_response(status);self.send_header('Content-Type','application/json')
                        self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
                        return
                headers={'Authorization':'Bearer '+token,'Content-Type':self.headers.get('Content-Type','application/json')}
                conn.request(self.command,self.path,body,headers)
                r=conn.getresponse();data=r.read()
                self.send_response(r.status)
                self.send_header('Content-Type',r.getheader('Content-Type','application/json'))
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            finally: conn.close()
        do_GET=relay
        do_POST=relay
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Relay)
    server.daemon_threads=True
    threading.Thread(target=server.serve_forever,daemon=True).start()
    def cleanup():
        server.shutdown()
        server.server_close()
        subprocess.run(ssh+['-O','cancel','-L',f'127.0.0.1:19090:{host}:18090',args.ssh_host],
                       capture_output=True,timeout=15)
    atexit.register(cleanup)
    print(f'HPC stepper ready at http://127.0.0.1:{args.port}/stepper',flush=True)
    while True:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/setup-status',timeout=10) as r: status=json.load(r)
            if all(v['bytes']==v['expected'] for v in status.values()): break
            print('Waiting for verified HPC model downloads',flush=True)
        except OSError: pass
        time.sleep(20)
    provenance={'resident_models':True,'job_id':args.job,'compute_host':host,'remote_root':args.remote_root,
                'models':json.loads((Path(__file__).parent/'models.json').read_text()),
                'started_at':time.time(),'local_weights':False,'evaluated_model':profile,
                'gpu_info':remote(f'cat {root}/runs/gpu-{args.job}.txt'),
                'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                 for name in ('trajectories.py','hpc_control.py','stepper.py','launch.py','codex_editor.py')}}
    trajectories.write_json(args.output/f'provenance-{args.job}.json',provenance)
    config={'prompt':args.prompt,'instruction':args.instruction,'max_tokens':args.max_tokens,
            'editor_model':codex_editor.MODEL,'variants':list(trajectories.VARIANTS),
            'probe_id':args.probe_id,'editor_preset':args.editor_preset,
            'policy':'independent prefixes; greedy; one-position intervention; no KV-edit replay'}
    try:
        inputs=args.configs or [None]
        for path in inputs:
            cfg={k:v for k,v in config.items() if k not in ('probe_id','editor_preset') or v is not None}
            destination=args.output
            if path:
                private=json.loads(path.read_text())
                if not private.get('prompt','').strip() or not private.get('instruction','').strip():
                    raise ValueError('Empty experiment prompt or editor instruction')
                cfg.update(prompt=private['prompt'],instruction=private['instruction'])
                destination=args.output/path.parent.parent.name
            api=trajectories.API(f'http://127.0.0.1:{args.port}')
            editor=codex_editor.propose
            if path and private.get('editor')=='local':
                editor=lambda instruction,text: api('propose',tokens=[1],editor='local',instruction=instruction,text=text)
                cfg['editor_model']='Qwen2.5-7B-Instruct-Q4_K_M'
            for attempt in range(20):
                try:
                    trajectories.generate(cfg,destination,api,editor)
                    break
                except (OSError, http.client.HTTPException):
                    if attempt == 19: raise
                    print('Connection lost; checkpoint preserved. Retrying in 30 seconds.',flush=True)
                    time.sleep(30)
                    state=remote(f'bjobs -noheader -o stat {args.job}').strip()
                    if state != 'RUN': raise RuntimeError('GPU allocation is no longer running')
                    subprocess.run(ssh+['-O','forward','-L',f'127.0.0.1:19090:{host}:18090',args.ssh_host],capture_output=True)
    finally:
        # Copy every completed stage to HPC as well, including after failures.
        subprocess.run(['rsync','-a','-e',shlex.join(ssh),str(args.output)+'/',
                        args.ssh_host+':'+args.remote_root+'/runs/'+args.output.name+'/'],check=True)
    print('Four trajectories complete; artifacts saved locally and on HPC',flush=True)
    if args.keep_serving:
        print('Keeping the stepper relay available while the HPC job runs',flush=True)
        while remote(f'bjobs -noheader -o stat {args.job}').strip() == 'RUN':
            time.sleep(30)

if __name__=='__main__': main()
