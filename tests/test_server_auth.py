import http.client
import json
from pathlib import Path
import tempfile
from http.server import ThreadingHTTPServer
import threading
import unittest
from unittest.mock import patch
from nla import launch

class AuthTests(unittest.TestCase):
    def test_hpc_endpoint_requires_token(self):
        with patch.object(launch, 'AUTH_TOKEN', 'a'*64):
            server=ThreadingHTTPServer(('127.0.0.1',0),launch.Handler)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                for headers, expected in [({},401),({'Authorization':'Bearer wrong'},401),({'Authorization':'Bearer '+ 'a'*64},200)]:
                    conn=http.client.HTTPConnection('127.0.0.1',server.server_port)
                    conn.request('GET','/setup-status',headers=headers)
                    response=conn.getresponse();self.assertEqual(response.status,expected);response.read();conn.close()
            finally:
                server.shutdown();server.server_close();thread.join()

    def test_stepper_response_is_archived_before_return(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launch,'LOGS',Path(tmp)), patch.object(launch,'RUNTIME',Path(tmp)), patch.object(launch,'AUDIT_LOG',True), patch.object(launch,'AUTH_TOKEN',None), patch('nla.stepper.run',return_value={'h':[1.,2.]}):
            (Path(tmp)/'frontend.html').write_text('')
            server=ThreadingHTTPServer(('127.0.0.1',0),launch.Handler)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                conn=http.client.HTTPConnection('127.0.0.1',server.server_port)
                request={'action':'inspect','tokens':[1]}
                conn.request('POST','/stepper-api',json.dumps(request),{'Content-Type':'application/json'})
                response=conn.getresponse();self.assertEqual(response.status,200)
                self.assertEqual(json.loads(response.read()),{'h':[1.,2.]});conn.close()
                saved_request=json.loads(next(Path(tmp).rglob('*.request.json')).read_text())
                saved_response=json.loads(next(Path(tmp).rglob('*.response.json')).read_text())
                self.assertEqual(saved_request['request'],request)
                self.assertEqual(saved_response['response'],{'h':[1.,2.]})
            finally:
                server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
