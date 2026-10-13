import unittest
from unittest.mock import patch, Mock
import stepper
import codex_editor

class EditorTests(unittest.TestCase):
    def test_subscription_routing_does_not_call_local_model(self):
        call=Mock(side_effect=AssertionError('Local model must not be used'))
        with patch('codex_editor.propose',return_value={'proposal':'mouse','editor_model':'gpt-6.1-sol'}) as editor:
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
