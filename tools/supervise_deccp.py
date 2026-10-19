"""Resume the existing five-case campaign; never create a second writer."""
import argparse,fcntl,json,os,re,shlex,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
from hpc.transport import SSHTransport,TransportUnavailable
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--campaign-dir',type=Path,required=True)
parser.add_argument('--ssh-host',required=True)
parser.add_argument('--ssh-socket',required=True)
parser.add_argument('--connect-script',type=Path)
parser.add_argument('--remote-root',required=True)
args=parser.parse_args()
B=args.campaign_dir.resolve()
remote_root=args.remote_root.rstrip('/')
transport=SSHTransport(args.ssh_host,args.ssh_socket,args.connect_script,B/'supervisor-transport.json')
lock=(B/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=json.loads((B/'status.json').read_text())
commands=json.loads((B/'controller-commands.json').read_text())
children={}
def save():
 state['supervisor_pid']=os.getpid();state['supervisor_updated']=time.time()
 p=B/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(B/'status.json')
def alive(pid,out):
 p=children.get(pid)
 if p is not None:return p.poll() is None
 result=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
 return result.returncode==0 and '-m hpc.hpc_control ' in result.stdout and str(out) in result.stdout

def launch(i,w,out):
 # An independent lifetime flock in hpc_control also prevents duplicate writers.
 cmd=commands[i].copy();cmd[cmd.index('--job')+1]=str(w['job'])
 (out/'controller-health.json').unlink(missing_ok=True)
 with (out/'controller.log').open('ab') as log:
  p=subprocess.Popen(cmd,cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 children[p.pid]=p
 w.update(controller_pid=p.pid,phase='running',updated=time.time(),error_type=None)
 save()

while True:
 pending=False
 for i,w in state['workers'].items():
  out=B/f'deccp-refusal-v2-case-{i}'
  if w['phase'] in ('complete','needs_attention','renewal_limit'):continue
  pending=True
  if alive(w.get('controller_pid',0),out):continue
  health_path=out/'controller-health.json'
  health=json.loads(health_path.read_text()) if health_path.exists() else {}
  if health.get('phase')=='complete':
   backup_path=out/'backup-health.json'
   backup=json.loads(backup_path.read_text()) if backup_path.exists() else {}
   if backup.get('phase')!='complete':
    try:
     transport.remote('true')
     result=subprocess.run(['rsync','-a','-e',shlex.join(transport.ssh),str(out)+'/',
          transport.host+':'+remote_root+'/runs/'+out.name+'/'],capture_output=True,timeout=180)
     if result.returncode: w.update(phase='backup_pending');save();continue
    except (subprocess.SubprocessError,OSError):
     w.update(phase='backup_pending');save();continue
    backup_path.write_text(json.dumps({'phase':'complete','updated':time.time()}))
   w.update(phase='complete');save();continue
  try:
   info=transport.remote(f'bjobs -a -noheader -o "stat exit_reason" {w["job"]} 2>/dev/null || true')
  except TransportUnavailable:
   w.update(phase='waiting_for_connection');save();continue
  if 'TERM_RUNLIMIT' in info:
   attempt=w.get('attempt',0)+1
   if attempt>8:w.update(phase='renewal_limit');save();continue
   # Submission is never blindly retried: its outcome could be unknown.
   name=f'nla-deccp-v2-{i}-renew-{attempt}'
   found=transport.remote(f'bjobs -noheader -J {name} -o "jobid stat" 2>/dev/null || true')
   jobs=[line.split()[0] for line in found.splitlines() if len(line.split())==2 and line.split()[1] in ('RUN','PEND','WAIT')]
   if len(jobs)>1:w.update(phase='needs_attention',error_type='MultipleAllocations');save();continue
   if jobs:job=int(jobs[0])
   else:
    cmd=f'cd {shlex.quote(remote_root)} && bsub -q {w["queue"]} -R '+shlex.quote('span[hosts=1] rusage[mem=6GB]')+f' -W 4:00 -J {name} < hpc/deccp-server.lsf'
    try:
     reply=subprocess.check_output(transport.ssh+[transport.host,'bash -lc '+shlex.quote(cmd)],text=True,stderr=subprocess.DEVNULL,timeout=45)
     job=int(re.search(r'Job <(\d+)>',reply)[1])
    except (subprocess.SubprocessError,TypeError):
     w.update(phase='needs_attention',error_type='SubmissionOutcomeUnknown');save();continue
   w.setdefault('previous_jobs',[]).append(w['job']);w.update(job=job,attempt=attempt)
   launch(i,w,out);continue
  if info.split()[:1] not in (['RUN'],['PEND'],['WAIT']):
   w.update(phase='needs_attention',error_type='AllocationStopped');save();continue
  error=health.get('error_type')
  if error and error not in ('TransportUnavailable','ConnectionError','ConnectionResetError','TimeoutError','URLError','RemoteDisconnected','BrokenPipeError'):
   w.update(phase='needs_attention',error_type=error);save();continue
  # A crash without failure metadata is recoverable, but bound crash loops.
  restarts=[t for t in w.get('restarts',[]) if time.time()-t<3600]
  if len(restarts)>=6:w.update(phase='needs_attention',error_type='RepeatedControllerFailure');save();continue
  restarts.append(time.time());w['restarts']=restarts
  launch(i,w,out)
 save()
 if not pending:break
 time.sleep(30)
