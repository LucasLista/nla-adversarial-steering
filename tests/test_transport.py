import subprocess
from unittest.mock import patch
import pytest
from hpc.transport import SSHTransport, TransportUnavailable


def test_helper_failure_does_not_abort_retries():
    t = SSHTransport('host', '/tmp/test-socket')
    operation = __import__('unittest.mock', fromlist=['Mock']).Mock(side_effect=[
        subprocess.CalledProcessError(255, 'ssh'),
        subprocess.CalledProcessError(255, 'ssh'), 'recovered'])
    with patch('hpc.transport.time.sleep'), patch.object(t, 'reconnect', side_effect=[
        subprocess.CalledProcessError(1, 'helper'), None]):
        assert t.retry(operation) == 'recovered'
    assert operation.call_count == 3


def test_remote_program_failure_is_not_retried():
    t = SSHTransport('host', '/tmp/test-socket')
    with patch.object(t, 'reconnect') as reconnect:
        with pytest.raises(subprocess.CalledProcessError):
            t.retry(lambda: subprocess.run(['false'], check=True))
        reconnect.assert_not_called()


def test_outage_has_distinct_error():
    t = SSHTransport('host', '/tmp/test-socket', retry_seconds=0)
    def fail():
        raise subprocess.CalledProcessError(255, 'ssh')
    with pytest.raises(TransportUnavailable):
        t.retry(fail)


def test_reconnect_restores_forwardings(tmp_path):
    t = SSHTransport('host', str(tmp_path/'socket'), helper='helper.py')
    t.forwards = ['127.0.0.1:1234:compute:5678']
    with patch('hpc.transport.subprocess.run') as run:
        run.side_effect = [subprocess.CompletedProcess([], 255),
                           subprocess.CompletedProcess([], 0),
                           subprocess.CompletedProcess([], 0),
                           subprocess.CompletedProcess([], 0)]
        t.reconnect()
    assert run.call_count == 4
    assert '-L' in run.call_args_list[-1].args[0]
