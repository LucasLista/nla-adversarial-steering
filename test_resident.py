import unittest
from unittest.mock import Mock, patch
import launch
class ResidentTests(unittest.TestCase):
    def test_models_reused_and_all_stopped(self):
        old=launch.RESIDENT
        launch.RESIDENT=True
        children=[]
        def spawn(*args,**kwargs):
            child=Mock();child.poll.return_value=None;children.append(child);return child
        connection=Mock();connection.getresponse.return_value.status=200
        try:
            with patch.object(launch.pathlib.Path,'exists',return_value=True), patch.object(launch.pathlib.Path,'stat') as stat, patch('builtins.open',return_value=Mock()), patch.object(launch.subprocess,'Popen',side_effect=spawn), patch.object(launch.http.client,'HTTPConnection',return_value=connection):
                for role in ['base','av','ar','base','ar','av']:
                    stat.return_value.st_size=launch.FILES[role][1]
                    launch.ensure(role)
                self.assertEqual(len(children),3)
                for child in children: child.terminate.assert_not_called()
                launch.stop()
                for child in children: child.terminate.assert_called_once()
        finally:
            launch.RESIDENT=old
if __name__=='__main__':unittest.main()
