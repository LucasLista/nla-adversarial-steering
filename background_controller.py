"""Detach the laptop HPC controller; emit only process metadata, never logs."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('controller_args', nargs=argparse.REMAINDER)
    args = parser.parse_args().controller_args
    if args and args[0] == '--':
        args = args[1:]
    if not args:
        parser.error('Pass hpc_control.py arguments after --')
    runs = ROOT / 'runs'
    runs.mkdir(exist_ok=True)
    pid_file = runs / 'controller.pid'
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text().strip()), 0)
        except (ProcessLookupError, ValueError):
            pass
        else:
            parser.error('Recorded controller process is still alive; check process metadata first')
    with (runs / 'controller.log').open('ab') as log:
        process = subprocess.Popen([sys.executable, str(ROOT / 'hpc_control.py'), *args],
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
