import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import hf_campaign


class CampaignTests(unittest.TestCase):
    def test_waits_before_submission_and_uses_separate_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'repo';root.mkdir()
            calls=[]
            replies=[json.dumps({'complete':False}),json.dumps({'complete':True}),'Job <123> is submitted']
            def remote(command,**kwargs):
                calls.append(command)
                return replies.pop(0)
            def launch(command,**kwargs):
                marker=Path(command[command.index('--pid-file')+1]);marker.parent.mkdir(exist_ok=True)
                marker.write_text('321\n')
            argv=['hf_campaign','--ssh-host','synthetic','--ssh-socket','synthetic',
                  '--remote-root','/synthetic','--probe-id','synthetic','--editor-preset','synthetic',
                  '--output',str(root/'run')]
            with patch.object(hf_campaign,'ROOT',root),patch('sys.argv',argv),patch('subprocess.check_output',side_effect=remote),patch('subprocess.run',side_effect=launch),patch('time.sleep') as sleep:
                hf_campaign.main()
            self.assertEqual(sleep.call_count,1)
            self.assertIn('bsub',calls[-1][-1])
            status=json.loads((root.parent/'hf-campaign-gemma3-12b.json').read_text())
            self.assertEqual(status['job_id'],123)
            self.assertEqual(status['pid'],321)
            self.assertEqual(status['phase'],'controller_started_pending_validation')


if __name__=='__main__':unittest.main()
