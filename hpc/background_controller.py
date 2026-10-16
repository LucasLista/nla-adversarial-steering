"""Detach the laptop HPC controller; emit only process metadata, never logs."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid-file',type=Path,help='Separate process marker for another model controller')
    parser.add_argument('--log-file',type=Path,help='Separate unread controller log')
    parser.add_argument('controller_args', nargs=argparse.REMAINDER)
    options = parser.parse_args()
    args = options.controller_args
    if args and args[0] == '--':
        args = args[1:]
    if not args:
        parser.error('Pass hpc.hpc_control arguments after --')
    runs = ROOT / 'runs'
    runs.mkdir(exist_ok=True)
    pid_file = options.pid_file or runs / 'controller.pid'
    log_file = options.log_file or runs / 'controller.log'
    pid_file.parent.mkdir(parents=True,exist_ok=True)
    log_file.parent.mkdir(parents=True,exist_ok=True)
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text().strip()), 0)
        except (ProcessLookupError, ValueError):
            pass
        else:
            parser.error('Recorded controller process is still alive; check process metadata first')
    with log_file.open('ab') as log:
        process = subprocess.Popen([sys.executable, '-m', 'hpc.hpc_control', *args],
                                   cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, start_new_session=True)
    pid_file.write_text(str(process.pid) + '\n')
    if sys.platform == 'darwin' and shutil.which('caffeinate'):
        subprocess.Popen(['caffeinate', '-i', '-w', str(process.pid)],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    print('Detached HPC controller PID:', process.pid)


if __name__ == '__main__':
    main()
