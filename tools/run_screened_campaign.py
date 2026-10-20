"""Run a priority-ordered private campaign across existing LSF allocations.

One coordinator owns the manifest. Each allocation processes one case at a time;
failed cases retain checkpoints and are retried only after allocation renewal.
Only status/exception metadata is emitted. Model output stays in private files.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nla.trajectories import write_json
from hpc.transport import SSHTransport


def next_case(cases):
    return next((c for c in cases if c['phase'] == 'pending'), None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True, type=Path)
    args = p.parse_args()
    campaign = args.campaign.resolve()
    lock = (campaign/'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((campaign/'status.json').read_text())
    ssh = ['ssh', '-S', state['ssh_socket'], '-o', 'BatchMode=yes', state['ssh_host']]
    transport = SSHTransport(state['ssh_host'], state['ssh_socket'],
                             helper=state.get('connect_script'),
                             health=campaign/'transport-health.json', retry_seconds=45)
    remote = transport.remote
    children = {}
    def submit_once(command):
        return subprocess.check_output(ssh + ['bash -lc '+shlex.quote(command)],
                                       stderr=subprocess.DEVNULL, text=True, timeout=45)
    def save():
        state['pid'] = os.getpid()
        state['updated_at'] = time.time()
        write_json(campaign/'status.json', state)
    while True:
        for worker in state['workers']:
            if worker['phase'] in ('stopped', 'needs_attention'):
                continue
            child = children.get(worker['slot'])
            if child and child.poll() is None:
                continue
            if child is None and worker.get('controller_pid'):
                try:
                    os.kill(worker['controller_pid'], 0)
                except ProcessLookupError:
                    pass
                else:
                    continue
            if worker.get('case'):
                case = next(c for c in state['cases'] if c['id'] == worker['case'])
                output = campaign/'results'/case['id']
                if (output/case['id']/'complete.json').exists():
                    case['phase'] = 'complete'
                    worker['case'] = None
                    worker['failures'] = 0
                elif child or case['phase'] == 'running':
                    health = output/'controller-health.json'
                    metadata = json.loads(health.read_text()) if health.exists() else {}
                    case['error_type'] = metadata.get('error_type', 'ControllerStopped')
                    # Resume against a still-running allocation only for transient
                    # transport failures, and bound retries to avoid crash loops.
                    worker['failures'] = worker.get('failures', 0) + (0 if case['error_type'] in (
                        'TransportUnavailable', 'ConnectionError', 'TimeoutError', 'URLError',
                        'RemoteDisconnected', 'ConnectionResetError', 'EditorTransportUnavailable') else 1)
                    case['phase'] = 'pending'
                    worker['case'] = None
                    if worker['failures'] >= 3:
                        case['phase'] = 'needs_attention'
                        worker['phase'] = 'needs_attention'
                        save()
                        continue
            worker['controller_pid'] = None
            try:
                status = remote(f'bjobs -a -noheader -o "stat exit_reason" {worker["job"]} 2>/dev/null || true').strip()
            except (OSError, subprocess.SubprocessError):
                save()
                continue
            if status.split()[:1] not in (['RUN'], ['PEND'], ['WAIT']):
                # Never repeat an ambiguous submission. Persist its outcome before
                # attempting recovery on a subsequent coordinator invocation.
                if worker.get('renewals', 0) >= 8 or 'TERM_RUNLIMIT' not in status:
                    worker['phase'] = 'needs_attention'
                    worker['scheduler_state'] = status.split()[:1]
                    save()
                    continue
                worker['phase'] = 'needs_attention'
                save()
                command = worker['submit_command']
                import re
                try:
                    reply = submit_once(command)
                    job = int(re.search(r'Job <(\d+)>', reply)[1])
                except (OSError, subprocess.SubprocessError, TypeError):
                    save()
                    continue
                worker.setdefault('previous_jobs', []).append(worker['job'])
                worker.update(job=job, phase='waiting', renewals=worker.get('renewals', 0)+1)
                save()
                continue
            worker['scheduler_state'] = status.split()[0]
            if not status.startswith('RUN'):
                continue
            case = next_case(state['cases'])
            if case is None:
                if all(c['phase'] in ('complete', 'needs_attention') for c in state['cases']):
                    remote(f'bkill {worker["job"]} >/dev/null 2>&1 || true')
                    worker['phase'] = 'stopped'
                continue
            slot = worker['slot']
            output = campaign/'results'/case['id']
            output.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, '-m', 'hpc.hpc_control', '--job', str(worker['job']),
                   '--ssh-host', state['ssh_host'], '--ssh-socket', state['ssh_socket'],
                   '--remote-root', state['remote_root'], '--configs', case['config'],
                   '--output', str(output), '--port', str(19200+slot*2),
                   '--tunnel-port', str(19201+slot*2), '--fast', '--cache-shared-prefixes',
                   '--max-tokens', str(state['max_tokens']), '--editor', 'codex']
            if state.get('connect_script'):
                cmd += ['--connect-script', state['connect_script']]
            (output/'controller-health.json').unlink(missing_ok=True)
            case.update(phase='running', job=worker['job'])
            worker.update(case=case['id'], phase='running')
            save()
            with (output/'controller.log').open('ab') as log:
                children[slot] = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=log)
            worker['controller_pid'] = children[slot].pid
            save()
        save()
        if all(w['phase'] in ('stopped', 'needs_attention') for w in state['workers']):
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
