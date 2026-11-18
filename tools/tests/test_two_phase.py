"""Behavioral cascade tests: labels, gating, votes, calibration and resets."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from csi.signal import (extract_motion_features, ALL_FEATURE_NAMES, FEATURE_NAMES,
                        MOTION_FEATURE_NAMES, fieldwise_ltf_normalize)
from csi.training import predict
from csi.realtime import StreamPredictor
from csi import webapp


class Presence:
    classes_=np.array([0,1])
    probability=.9
    def predict_proba(self,X):
        assert X.shape[1]==13
        return np.tile([1-self.probability,self.probability],(len(X),1))


class Motion:
    classes_=np.array([1,2])
    calls=0
    def predict(self,X):
        assert X.shape[1]==18
        self.calls+=len(X)
        return np.full(len(X),2)
    def predict_proba(self,X):
        assert X.shape[1]==18
        return np.tile([.2,.8],(len(X),1))


def bundle():
    A=np.linspace(15,40,107,dtype=np.float32)
    return dict(version=3,features=ALL_FEATURE_NAMES,phase1=Presence(),phase2=Motion(),
        baseline=fieldwise_ltf_normalize(A[None,:])[0],presence_threshold=.5,vote_windows=5,
        metadata=dict(fs_hz=25,window_s=2,step_s=.5,layout='ht20_256'))


def packet(i):
    return dict(timestamp=100000+i*40000,channel=6,
                amplitude=np.linspace(15,40,107,dtype=np.float32))


class TwoPhaseTests(unittest.TestCase):
    def test_resample_roundoff_is_not_extrapolation(self):
        from csi.signal import resample
        t=np.array([5.8,5.8999999999999995]);A=np.ones((2,107))
        self.assertEqual(resample(t,A,np.array([5.8,5.9])).shape,(2,107))
        with self.assertRaises(ValueError):resample(t,A,np.array([5.8,5.91]))

    def test_motion_matches_reference(self):
        root=Path(__file__).resolve().parents[2]
        path=root/'ThamKhao/esp32-based-presence-and-motion-detection-using-wifi-csi-main/phase2/python/phase2_utils.py'
        spec=importlib.util.spec_from_file_location('reference',path)
        ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
        W=np.random.default_rng(42).normal(1,.1,(50,107)).astype(np.float32)
        np.testing.assert_allclose(extract_motion_features(W,25),ref._receiver_motion_features(W,25),atol=1e-7)

    def test_offline_phase2_never_sees_empty(self):
        b=bundle();X=np.ones((3,31))
        b['phase1'].probability=.1
        np.testing.assert_array_equal(predict(b,X),[0,0,0])
        self.assertEqual(b['phase2'].calls,0)
        b['phase1'].probability=.9
        np.testing.assert_array_equal(predict(b,X),[2,2,2])
        self.assertEqual(b['phase2'].calls,3)

    def test_voting_gate_and_transition_to_empty(self):
        b=bundle();stream=StreamPredictor(b);events=[]
        for i in range(100):
            result=stream.feed(packet(i))
            if result:events.append(result)
        self.assertEqual([e['status'] for e in events[:4]],['stabilizing']*4)
        self.assertTrue(all(e['phase2']['status']=='blocked' for e in events[:4]))
        self.assertEqual(events[4]['phase1']['decision'],'present')
        self.assertEqual(events[4]['phase2']['decision'],'walking')
        b['phase1'].probability=.1
        for i in range(100,170):
            result=stream.feed(packet(i))
            if result:events.append(result)
        self.assertEqual(events[-1]['label'],'empty')
        self.assertEqual(events[-1]['phase2']['status'],'blocked')
        self.assertIsNone(events[-1]['phase2']['probabilities'])
        calls=b['phase2'].calls
        for i in range(170,200):stream.feed(packet(i))
        self.assertEqual(b['phase2'].calls,calls)
        result=stream.feed(packet(230))
        self.assertEqual(result['status'],'waiting_window')
        self.assertEqual(len(stream.recent),0)

    def test_calibration_requires_fresh_window_and_never_modifies_model(self):
        b=bundle();original=b['baseline'].copy()
        stream=StreamPredictor(b,vote=1,calibration_s=1)
        completed=None
        for i in range(30):
            result=stream.feed(packet(i))
            if result and result['status']=='calibration_complete':completed=i
        self.assertIsNotNone(completed)
        self.assertEqual(b['phase2'].calls,0)
        np.testing.assert_array_equal(b['baseline'],original)
        self.assertTrue(stream.calibrated)
        self.assertEqual(len(stream.recent),0)
        for i in range(30,completed+50):stream.feed(packet(i))
        self.assertEqual(b['phase2'].calls,0)

    def test_old_model_rejected_and_predict_options(self):
        with self.assertRaises(ValueError):StreamPredictor(dict(version=2))
        for vote in [0,2,32]:
            with self.assertRaises(ValueError):StreamPredictor(bundle(),vote=vote)
        cmd=webapp.make_command('predict',dict(port='COM_TEST',calibration=12,vote=5,countdown=7))
        self.assertEqual(cmd[cmd.index('--calibration')+1],'12.0')
        self.assertEqual(cmd[cmd.index('--countdown')+1],'7')
        with self.assertRaises(ValueError):webapp.make_command('predict',dict(port='x',vote=2))


if __name__=='__main__':unittest.main()
