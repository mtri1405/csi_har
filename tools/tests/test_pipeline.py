"""Regression tests use synthetic signals, never claim real-world accuracy."""
import sys
import csv
import json
from pathlib import Path
import tempfile
import unittest
import contextlib
import io
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from csi.protocol import HEADER, parse_packet, unwrap_timestamps
from csi.signal import (prepare_feature_dataset, session_features, FEATURE_NAMES, ALL_FEATURE_NAMES,
                       load_receiver_csv)
from csi.realtime import StreamPredictor
from csi.training import train, predict

class PipelineTests(unittest.TestCase):
    def test_cli_commands_after_reorganization(self):
        import subprocess
        from unittest.mock import patch
        from csi import webapp
        from csi.paths import TOOLS_ROOT
        manifest = TOOLS_ROOT / 'tests' / 'manifest_not_written.csv'
        # Validate the exact subprocess routes emitted by the web, outside project cwd.
        with tempfile.TemporaryDirectory() as tmp, patch.object(webapp, 'create_manifest', return_value=manifest):
            for action in ('record', 'dataset', 'train', 'predict', 'monitor', 'audit'):
                command = webapp.make_command(action, {'port': 'COM_TEST'})
                self.assertEqual(Path(command[1]), TOOLS_ROOT / 'run.py')
                result = subprocess.run([*command, '--help'], cwd=tmp,
                    capture_output=True, text=True, encoding='utf-8', timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_protocol_and_clock(self):
        self.assertIsNone(parse_packet('log CSI_DATA [1,2]'))
        np.testing.assert_array_equal(unwrap_timestamps([2**32-10,5,15]),
                                      [2**32-10,2**32+5,2**32+15])
        with self.assertRaises(ValueError):
            unwrap_timestamps([100,20,30])

    def test_csv_signal_preview(self):
        from csi.viewer import csv_preview
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'empty.csv'
            with path.open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(HEADER)
                for i in range(100):
                    writer.writerow(['CSI_DATA',i,1000+i*40000,-50,-96,6,0,1,3,0,0,256,1,0,str([20,25]*128)])
                f.write('bad row\n')
            data=csv_preview(path,limit=10)
            self.assertEqual(data['total'],100)
            self.assertEqual(data['count'],10)
            self.assertEqual(data['rejected'],1)
            self.assertAlmostEqual(data['rate_hz'],25)
            self.assertEqual(len(data['frames'][0]['amplitude']),107)
            self.assertTrue(all(not f['gap_before'] for f in data['frames']))
            self.assertTrue(np.isfinite([f['normalized'] for f in data['frames']]).all())

    def test_live_signal_buffer(self):
        import sys
        from csi import webapp as web_app
        identifier='test_monitor'
        with web_app.jobs_lock:
            web_app.jobs[identifier]={'status':'running','action':'monitor','log':''}
        try:
            event={'type':'csi_frame','timestamp':1000,'amplitude':[20]*107,'rssi':-50,'channel':6}
            code=f'import json; event={event!r}; [print(json.dumps(event)) for _ in range(210)]'
            web_app.run_command(identifier,[sys.executable,'-c',code])
            self.assertEqual(web_app.jobs[identifier]['status'],'done')
            self.assertEqual(len(web_app.jobs[identifier]['frames']),200)
            self.assertEqual(web_app.jobs[identifier]['log'],'')
        finally:
            with web_app.jobs_lock:
                del web_app.jobs[identifier]

    def test_training_and_stream_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[]
            for label_index,label in enumerate(('empty','standing','walking')):
                for split_index,split in enumerate(('train','val','test')):
                    path=root/f'{label}_{split}.csv'
                    with path.open('w',newline='') as f:
                        writer=csv.writer(f); writer.writerow(HEADER)
                        for i in range(180):
                            # Jittered hardware timestamps, distinct recording bytes.
                            t=100000+i*40000+(i%3)*150
                            v=(20+np.arange(256)%13+label_index*3+
                               (int(5*np.sin(i/3)) if label_index==2 else 0))
                            writer.writerow(['CSI_DATA',i,t,-50,-96,6,0,1,3,0,0,256,1,0,str(v.tolist())])
                    # Add unique timestamp offset to prevent duplicate-content leakage.
                    text=path.read_text().replace('100000,',f'{100000+split_index},',1)
                    path.write_text(text)
                    rows.append([f'{label}_{split}',label,split,str(path)])
            manifest=root/'manifest.csv'
            with manifest.open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['session_id','label','split','rx_csv']);writer.writerows(rows)
            features=prepare_feature_dataset(manifest,root/'features')
            with contextlib.redirect_stdout(io.StringIO()):
                bundle=train(root/'features',root/'model',trees=5)
            self.assertEqual(set(bundle['phase1'].classes_),{0,1})
            self.assertEqual(set(bundle['phase2'].classes_),{1,2})
            self.assertEqual(len(predict(bundle,features[ALL_FEATURE_NAMES].to_numpy())),len(features))
            self.assertEqual(bundle['phase1'].n_features_in_,13)
            self.assertEqual(bundle['phase2'].n_features_in_,18)
            self.assertEqual(bundle['training_data']['phase2']['empty'],0)
            path=root/'walking_train.csv'
            X,_=session_features(path,bundle['baseline'])
            # Capture the exact feature vector passed to phase 1.
            original=bundle['phase1'].predict_proba
            captured=[]
            def capture(x):
                captured.append(x.copy());return original(x)
            bundle['phase1'].predict_proba=capture
            stream=StreamPredictor(bundle,vote=1)
            with path.open() as f:
                next(f)
                for line in f:
                    packet=parse_packet(line)
                    self.assertIsNotNone(packet)
                    stream.feed(packet)
            live=np.vstack(captured)
            np.testing.assert_allclose(live[:len(X)],X[:,:13],rtol=2e-4,atol=2e-5)
            self.assertGreaterEqual(len(live),len(X))
            # Large gaps must discard old data instead of emitting stale inference.
            packet['timestamp']+=1000000
            self.assertEqual(stream.feed(packet)['status'],'waiting_window')

    def test_phone_access_addresses(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from csi import webapp
        webapp.lan_ipv4_addresses.cache_clear()
        try:
            addresses = ['127.0.0.1', '169.254.1.2', '192.168.1.9', '0.0.0.0']
            with patch.object(webapp.socket, 'getaddrinfo', return_value=[
                    (None, None, None, None, (ip, 0)) for ip in addresses]):
                access = webapp.server_access_info(SimpleNamespace(server_address=('0.0.0.0', 8765)))
                self.assertEqual(access['lan_urls'], ['http://192.168.1.9:8765'])
                self.assertTrue(access['lan_enabled'])
            local = webapp.server_access_info(SimpleNamespace(server_address=('127.0.0.1', 8765)))
            self.assertFalse(local['lan_enabled'])
            self.assertEqual(local['lan_urls'], [])
        finally:
            webapp.lan_ipv4_addresses.cache_clear()

    def test_web_http_and_missing_data(self):
        import threading
        import urllib.request
        import urllib.error
        from csi import webapp as web_app
        server=web_app.ThreadingHTTPServer(('127.0.0.1',0),web_app.Handler)
        worker=threading.Thread(target=server.serve_forever,daemon=True)
        worker.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(base) as response:
                self.assertIn(b'CSI Action Lab',response.read())
            with opener.open(base+'/api/status') as response:
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
                payload = json.load(response)
                self.assertIn('recordings',payload)
                self.assertFalse(payload['access']['lan_enabled'])
                self.assertIn('server_time', payload)
            with opener.open(base+'/api/server-info') as response:
                self.assertEqual(json.load(response)['local_url'], base)
            with opener.open(base+'/signal.js') as response:
                self.assertIn(b'function drawSignal',response.read())
            with opener.open(base+'/api/signal?job=missing') as response:
                self.assertEqual(json.load(response)['frames'],[])
            with self.assertRaises(urllib.error.HTTPError) as blocked:
                opener.open(base+'/api/signal?file=README.md')
            self.assertEqual(blocked.exception.code,400)
            request=urllib.request.Request(base+'/api/run',data=b'{"action":"record","label":"../bad"}',
                                          headers={'Content-Type':'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as error:
                opener.open(request)
            self.assertEqual(error.exception.code,400)
            with tempfile.TemporaryDirectory() as tmp:
                previous=web_app.DATA_ROOT
                web_app.DATA_ROOT=Path(tmp)
                try:
                    with self.assertRaises(ValueError):
                        web_app.create_manifest()
                finally:
                    web_app.DATA_ROOT=previous
        finally:
            server.shutdown();server.server_close();worker.join()

if __name__=='__main__':
    unittest.main()
