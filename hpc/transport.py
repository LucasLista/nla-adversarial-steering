"""Recover a shared SSH connection without racing authentication helpers."""
import fcntl
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time


class TransportUnavailable(ConnectionError):
    pass


class SSHTransport:
    def __init__(self, host, socket, helper=None, health=None, retry_seconds=900):
        self.host, self.socket, self.helper = host, socket, helper
        self.health, self.retry_seconds = health, retry_seconds
        self.ssh = ['ssh', '-S', socket, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']
        self.forwards = []

    def status(self, phase):
        if self.health:
            path = Path(self.health)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps({'phase': phase, 'updated': time.time()}))
            temporary.replace(path)

    def reconnect(self):
        # All controllers share the socket. Only one may repair it at a time.
        with open(self.socket + '.reconnect.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            check = subprocess.run(self.ssh + ['-O', 'check', self.host], capture_output=True, timeout=20)
            if check.returncode and self.helper:
                uv = shutil.which('uv') or '/opt/homebrew/bin/uv'
                subprocess.run([uv, 'run', '--with', 'pexpect', str(self.helper)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               check=True, timeout=90)
            subprocess.run(self.ssh + ['-O', 'check', self.host], capture_output=True, check=True, timeout=20)
            # OpenSSH treats re-registering an identical forwarding as idempotent.
            for specification in self.forwards:
                subprocess.run(self.ssh + ['-O', 'forward', '-L', specification, self.host],
                               capture_output=True, check=True, timeout=20)

    def retry(self, operation):
        deadline = time.monotonic() + self.retry_seconds
        while True:
            try:
                result = operation()
                self.status('connected')
                return result
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
                # A remote program's nonzero exit is not a broken SSH connection.
                if isinstance(exc, subprocess.CalledProcessError) and exc.returncode != 255:
                    raise
                self.status('reconnecting')
                if time.monotonic() >= deadline:
                    self.status('transport_unavailable')
                    raise TransportUnavailable('SSH recovery deadline exceeded') from exc
                time.sleep(5)
                try:
                    self.reconnect()
                except (subprocess.SubprocessError, OSError):
                    # Authentication helpers may themselves fail transiently.
                    pass

    def remote(self, command):
        return self.retry(lambda: subprocess.check_output(
            self.ssh + [self.host, 'bash -lc ' + shlex.quote(command)],
            text=True, stderr=subprocess.DEVNULL, timeout=45).strip())

    def forward(self, specification):
        if specification not in self.forwards:
            self.forwards.append(specification)
        return self.retry(lambda: subprocess.run(
            self.ssh + ['-O', 'forward', '-L', specification, self.host],
            capture_output=True, check=True, timeout=20))
