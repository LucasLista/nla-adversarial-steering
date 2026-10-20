import unittest
import json
import tempfile
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, Mock
from nla import stepper
from nla import codex_editor

class EditorTests(unittest.TestCase):
    def test_subscription_lock_excludes_an_independent_process(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'editor.lock'
            code = 'import fcntl,sys; f=open(sys.argv[1],"a");\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(17)'
            with codex_editor.subscription_lock(path):
                self.assertEqual(subprocess.run([sys.executable,'-c',code,str(path)]).returncode,17)
            self.assertEqual(subprocess.run([sys.executable,'-c',code,str(path)]).returncode,0)
    def test_only_error_envelopes_classify_subscription_failure(self):
        for event in ({'type':'error','message':'Usage limit reached'},
                      {'type':'turn.failed','error':{'message':'Rate limit reached'}}):
            self.assertIs(codex_editor.execution_failure_class(json.dumps(event)), codex_editor.EditorSubscriptionLimited)
        content = json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'usage limit unauthorized'}})
        self.assertIs(codex_editor.execution_failure_class(content), RuntimeError)
        self.assertIs(codex_editor.execution_failure_class('usage limit unauthorized'), RuntimeError)
    def test_subscription_failures_have_safe_distinct_classes(self):
        self.assertIs(codex_editor.failure_class('Usage limit reached'), codex_editor.EditorSubscriptionLimited)
        self.assertIs(codex_editor.failure_class('Unauthorized'), codex_editor.EditorAuthenticationFailed)
        self.assertIs(codex_editor.failure_class('Synthetic infrastructure failure'), RuntimeError)
    def test_subscription_routing_does_not_call_local_model(self):
        call=Mock(side_effect=AssertionError('Local model must not be used'))
        with patch('nla.codex_editor.propose',return_value={'proposal':'mouse','editor_model':'gpt-6.1-sol'}) as editor:
            result=stepper.run({'action':'propose','tokens':[1],'editor':'codex','instruction':'rewrite','text':'rabbit'},call,'')
        editor.assert_called_once_with('rewrite','rabbit')
        self.assertEqual(result['editor_model'],'gpt-6.1-sol')
    def test_unknown_provider_fails(self):
        with self.assertRaises(ValueError):
            stepper.run({'action':'propose','tokens':[1],'editor':'other','instruction':'rewrite','text':'rabbit'},Mock(),'')
    def test_api_key_auth_rejected(self):
        with patch.dict('os.environ',{},clear=True), patch('shutil.which',return_value='/bin/codex'), patch('subprocess.run',return_value=Mock(returncode=0,stdout='Logged in using an API key',stderr='')):
            with self.assertRaisesRegex(RuntimeError,'requires Codex signed in with ChatGPT'):
                codex_editor.propose('rewrite','rabbit')

if __name__=='__main__': unittest.main()

class EditorNetworkRecoveryTests(unittest.TestCase):
    def test_structured_network_failure_is_resumable(self):
        event = json.dumps({'type':'error','message':'error sending request: connection reset'})
        kind = codex_editor.execution_failure_class(event)
        self.assertIs(kind, codex_editor.EditorTransportUnavailable)
        self.assertTrue(issubclass(kind, ConnectionError))
        self.assertIs(codex_editor.execution_failure_class(json.dumps(
            {'type':'item.completed','text':'connection reset'})), RuntimeError)
