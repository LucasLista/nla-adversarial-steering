"""Read-only, loopback viewer for durable trajectory checkpoints."""
import argparse
import html
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
VARIANTS=('original','roundtrip','edited_raw','edited_corrected')
def read(p):
    return json.loads(p.read_text()) if p.exists() else {}
TEXT_CACHE={}
def token_text(step, token_id):
    """The chosen token's text, taken from the step's saved probability tables."""
    key=(str(step),token_id)
    if key not in TEXT_CACHE:
        text=None
        for name in ('compare','inspect'):
            r=read(step/(name+'.json')).get('response',{})
            for dist in [r.get(k) for k in VARIANTS]+[r.get('baseline')]:
                for row in (dist or {}).get('completion_probabilities',[{}])[0].get('top_logprobs',[]):
                    if row.get('id')==token_id: text=row.get('token')
                if text is not None: break
            if text is not None: break
        if text is None: return None
        TEXT_CACHE[key]=text
    return TEXT_CACHE[key]
def snapshot(root, run, branch, index=None):
    if run not in ('run-1','run-2','run-3') or branch not in VARIANTS: raise ValueError('Unknown run or branch')
    folder=root/run/branch
    state=read(folder/'state.json')
    if index is None:
        texts=[token_text(folder/f'step-{i:04d}',t) for i,t in enumerate(state.get('generated',[]))]
        return {'texts':texts,'tokens':state.get('generated',[]),'text':state.get('response',''),'ended':state.get('ended',False),'started':(folder/'state.json').exists(), 'complete':(root/run/'complete.json').exists()}
    steps=state.get('steps',[])
    if index<0 or index>=len(steps): raise ValueError('Token not saved yet')
    step=folder/f'step-{index:04d}'
    def response(name): return read(step/(name+'.json')).get('response',{})
    inspected=response('inspect');explained=response('explain');compared=response('compare');editor=response('editor')
    distributions={key:compared[key] for key in VARIANTS if key in compared} or {'original':inspected.get('baseline',{})}
    top=[distributions.get(k,{}).get('completion_probabilities',[{}])[0].get('top_logprobs',[{}])[0].get('id') for k in VARIANTS]
    return {'chosen':steps[index]['chosen_id'],'variant':branch,'original':explained.get('text',''),'edit':editor.get('proposal',explained.get('text','')),'metrics':compared.get('metrics',explained.get('metrics',{})), 'distributions':distributions,'agreement':None not in top and len(set(top))==1,'h':inspected.get('h'), 'reconstruction':explained.get('reconstruction'),'error':compared.get('error',explained.get('error'))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=18096);p.add_argument('--banner',default='',help='Notice shown above the results');args=p.parse_args()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                u=urlsplit(self.path)
                if u.path=='/':
                    data=Path(__file__).with_name('batch_viewer.html').read_text()
                    if args.banner:
                        data=data.replace('</h1>','</h1><p style="background:#3c3018;border:1px solid #d6a84b;color:#f6d788;padding:10px;border-radius:8px">'+html.escape(args.banner)+'</p>',1)
                    data=data.encode();kind='text/html'
                elif u.path=='/api':
                    q=parse_qs(u.query);index=int(q['index'][0]) if 'index' in q else None
                    data=json.dumps(snapshot(args.root,q.get('run',['run-1'])[0],q.get('branch',['original'])[0],index)).encode();kind='application/json'
                else: self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
            except (ValueError,KeyError,IndexError): self.send_error(400,'Invalid checkpoint request')
        def log_message(self,*args): pass
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
if __name__=='__main__': main()
