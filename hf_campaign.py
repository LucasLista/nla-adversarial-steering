"""Wait for verified snapshots, then submit a validation-gated Gemma HPC job."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from trajectories import write_json

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='gemma3-12b')
    parser.add_argument('--ssh-host',required=True)
    parser.add_argument('--ssh-socket',required=True)
    parser.add_argument('--remote-root',required=True)
    parser.add_argument('--connect-script',type=Path)
    parser.add_argument('--probe-id',required=True)
    parser.add_argument('--editor-preset',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--port',default=18096,type=int)
    parser.add_argument('--tunnel-port',default=19092,type=int)
    parser.add_argument('--max-tokens',default=256,type=int)
    args=parser.parse_args()
    if args.model!='gemma3-12b':
        parser.error('This initial two-V100 resource plan is for Gemma-3-12B only')
    state_path=ROOT.parent/('hf-campaign-'+args.model+'.json')
    if state_path.exists() and json.loads(state_path.read_text()).get('job_id'):
        parser.error('Campaign already submitted a job; inspect scheduler metadata before resuming')
    ssh=['ssh','-S',args.ssh_socket,'-o','BatchMode=yes',args.ssh_host]
    def remote(command):
        return subprocess.check_output(ssh+['bash -lc '+shlex.quote(command)],text=True,timeout=30).strip()
    progress=args.remote_root+'/.runtime/hf-models/'+args.model+'/download-progress.json'
    code='import json,pathlib; p=pathlib.Path('+repr(progress)+'); x=json.loads(p.read_text()) if p.exists() else {}; print(json.dumps({k:x.get(k) for k in ("complete","files_complete","files_total")}))'
    write_json(state_path,{'model':args.model,'phase':'waiting_for_snapshot'})
    while True:
        status=json.loads(remote('python3 -c '+shlex.quote(code)))
        if status.get('complete'):break
        time.sleep(30)
    command='cd '+shlex.quote(args.remote_root)+' && NLA_HF_MODEL='+shlex.quote(args.model)+' bsub < hpc/hf-server.lsf'
    submitted=remote(command)
    matched=re.search(r'Job <(\d+)>',submitted)
    if not matched:raise RuntimeError('HPC submission did not return a job ID')
    job=matched.group(1)
    write_json(state_path,{'model':args.model,'phase':'submitted_for_validation','job_id':int(job),
                          'port':args.port,'tunnel_port':args.tunnel_port,'output':str(args.output)})
    pid_file=ROOT/'runs'/('controller-'+args.model+'.pid')
    command=[sys.executable,str(ROOT/'background_controller.py'),'--pid-file',str(pid_file),
             '--log-file',str(ROOT/'runs'/('controller-'+args.model+'.log')),'--',
             '--hf-backend','--model',args.model,'--job',job,'--ssh-host',args.ssh_host,
             '--ssh-socket',args.ssh_socket,'--remote-root',args.remote_root,
             '--output',str(args.output),'--probe-id',args.probe_id,'--editor-preset',args.editor_preset,
             '--max-tokens',str(args.max_tokens),'--port',str(args.port),'--tunnel-port',str(args.tunnel_port)]
    if args.connect_script:command+=['--connect-script',str(args.connect_script)]
    subprocess.run(command,cwd=ROOT,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    pid=int(pid_file.read_text())
    write_json(state_path,{'model':args.model,'phase':'controller_started_pending_validation','job_id':int(job),
                          'pid':pid,'port':args.port,'tunnel_port':args.tunnel_port,'output':str(args.output)})


if __name__=='__main__':main()
