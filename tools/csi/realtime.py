"""Single-RX cascade with optional startup EMPTY calibration and presence voting."""
import argparse
import json
import math
import time
from collections import deque
import joblib
import numpy as np
import serial
from csi.protocol import parse_packet
from csi.signal import (resample, fieldwise_ltf_normalize, temporal_smooth,
                       extract_two_phase_features, ALL_FEATURE_NAMES, MAX_INTERP_GAP_S)
from csi.training import phase1_probability


class StreamPredictor:
    def __init__(self, bundle, vote=None, calibration_s=0):
        if bundle.get('version') != 3 or bundle.get('features') != ALL_FEATURE_NAMES:
            raise ValueError('Model version 3 required. Rebuild dataset and train both phases.')
        vote = bundle.get('vote_windows',5) if vote is None else vote
        if not isinstance(vote,int) or vote < 1 or vote > 31 or vote % 2 == 0:
            raise ValueError('vote must be an odd integer in 1..31')
        if not math.isfinite(calibration_s) or calibration_s < 0 or calibration_s > 120:
            raise ValueError('calibration must be 0..120 seconds')
        self.bundle = bundle
        self.meta = bundle['metadata']
        self.fs = self.meta['fs_hz']
        self.win = round(self.fs*self.meta['window_s'])
        self.step = round(self.fs*self.meta['step_s'])/self.fs
        self.baseline = np.array(bundle['baseline'],dtype=np.float32,copy=True)
        self.calibration_s = calibration_s
        self.calibrated = calibration_s == 0
        self.recent = deque(maxlen=vote)
        self.frames = deque()
        self.previous = None
        self.offset = 0
        self.anchor = self.end = self.channel = None

    def clear_window(self):
        self.frames.clear()
        self.recent.clear()
        self.anchor = self.end = None

    def reset_stream(self):
        self.clear_window()
        self.previous = None
        self.offset = 0
        self.channel = None

    def _resampled(self, grid):
        times = np.array([f[0] for f in self.frames])
        amplitudes = np.stack([f[1] for f in self.frames])
        rate = (len(times)-1)/(times[-1]-times[0])
        if rate > self.fs*1.3:
            raise ValueError(f'CSI rate {rate:.1f}Hz exceeds model fs={self.fs}; check AP/traffic')
        return fieldwise_ltf_normalize(resample(times,amplitudes,grid),self.meta['layout'])

    def feed(self, packet):
        raw = packet['timestamp']
        reset = (self.previous is not None and raw <= self.previous
                 and self.previous-raw < 2**31)
        if reset:
            self.offset = 0
        elif self.previous is not None and raw < self.previous-2**31:
            self.offset += 2**32
        t = (raw+self.offset)*1e-6
        gap = bool(self.frames and t-self.frames[-1][0] > MAX_INTERP_GAP_S)
        changed = self.channel is not None and self.channel != packet['channel']
        if reset or gap or changed:
            self.clear_window()
        self.previous = raw
        self.channel = packet['channel']
        if self.anchor is None:
            self.anchor = t
            self.end = t+(self.win-1)/self.fs
        self.frames.append((t,packet['amplitude']))
        if not self.calibrated:
            elapsed = t-self.anchor
            if elapsed < self.calibration_s:
                return dict(status='calibrating',elapsed_s=elapsed,target_s=self.calibration_s,
                            phase2=dict(status='blocked',reason='empty_calibration'))
            grid = np.arange(self.anchor,t,1/self.fs)
            normalized = self._resampled(grid)
            # Match fit_empty_baseline: median of resampled, field-normalized CSI.
            self.baseline = np.median(normalized,axis=0).astype(np.float32)
            drift = float(np.sqrt(np.mean((self.baseline-self.bundle['baseline'])**2)))
            self.calibrated = True
            self.clear_window()
            return dict(status='calibration_complete',baseline_drift_rmse=drift,
                        phase2=dict(status='blocked',reason='waiting_for_fresh_window'))
        if t+1e-9 < self.end:
            if reset or gap or changed:
                return dict(status='waiting_window',reason='stream_reset_or_gap',
                            phase2=dict(status='blocked',reason='invalid_window'))
            return None
        grid = self.end-(self.win-1)/self.fs+np.arange(self.win)/self.fs
        W = temporal_smooth(self._resampled(grid))
        X = extract_two_phase_features(W,self.baseline,self.fs)[None,:]
        probability = float(phase1_probability(self.bundle,X)[0])
        threshold = self.bundle['presence_threshold']
        raw_present = probability >= threshold
        self.recent.append(int(raw_present))
        enough_votes = len(self.recent) == self.recent.maxlen
        present = sum(self.recent)*2 > len(self.recent)
        phase1 = dict(probabilities=dict(empty=1-probability,present=probability),
                      threshold=threshold,raw_decision='present' if raw_present else 'empty',
                      decision=('present' if present else 'empty') if enough_votes else None,
                      present_votes=sum(self.recent),vote_count=len(self.recent),
                      vote_target=self.recent.maxlen)
        phase2 = dict(status='blocked',reason='phase1_empty' if enough_votes else 'phase1_stabilizing',
                      decision=None,probabilities=None)
        label = None
        if enough_votes:
            label = 'empty'
            if present:
                model = self.bundle['phase2']
                motion = X[:,13:]
                class_id = int(model.predict(motion)[0])
                label = {1:'standing',2:'walking'}[class_id]
                probabilities = { {1:'standing',2:'walking'}[int(c)]:float(p)
                                  for c,p in zip(model.classes_,model.predict_proba(motion)[0])}
                phase2 = dict(status='active',decision=label,probabilities=probabilities)
        result = dict(status='prediction' if enough_votes else 'stabilizing',
                      timestamp=time.time(),label=label,present=present if enough_votes else None,
                      phase1=phase1,phase2=phase2,
                      phase1_presence_probability=probability,
                      phase2_probabilities=phase2['probabilities'],
                      baseline_source='startup_empty_calibration' if self.calibration_s else 'train_empty')
        self.end += self.step
        start = self.end-(self.win-1)/self.fs
        while len(self.frames)>2 and self.frames[1][0] <= start:
            self.frames.popleft()
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',required=True)
    parser.add_argument('--baudrate',type=int,default=2000000)
    parser.add_argument('--model',required=True)
    parser.add_argument('--vote',type=int,default=5)
    parser.add_argument('--calibration',type=float,default=0,
                        help='Seconds of known EMPTY room before prediction; 0 uses training baseline')
    parser.add_argument('--countdown',type=int,default=5)
    args=parser.parse_args()
    if not 0 <= args.countdown <= 3600:
        parser.error('countdown must be 0..3600')
    predictor=StreamPredictor(joblib.load(args.model),vote=args.vote,calibration_s=args.calibration)
    def emit(value):
        print(json.dumps(value,ensure_ascii=False),flush=True)
    with serial.Serial(args.port,args.baudrate,timeout=.1) as connection:
        if args.calibration:
            for remaining in range(args.countdown,0,-1):
                emit(dict(status='countdown',remaining_s=remaining,message='Leave the sensing area empty for calibration'))
                time.sleep(1)
        connection.reset_input_buffer()
        emit(dict(status='calibrating' if args.calibration else 'waiting_window',
                  elapsed_s=0,target_s=args.calibration))
        last_valid=time.monotonic()
        last_emit=0
        pending=bytearray()
        while True:
            pending.extend(connection.read(max(1,min(connection.in_waiting,65536))))
            lines=pending.split(b'\n');pending=bytearray(lines.pop())
            for line in lines:
                packet=parse_packet(line.decode('utf-8',errors='replace'),predictor.meta['layout'])
                if packet is None:
                    continue
                last_valid=time.monotonic()
                result=predictor.feed(packet)
                if result and (result['status']!='calibrating' or time.monotonic()-last_emit>=.5):
                    emit(result);last_emit=time.monotonic()
            if len(pending)>65536:
                pending.clear()
            if time.monotonic()-last_valid>3:
                predictor.reset_stream()
                emit(dict(status='no_signal',phase2=dict(status='blocked',reason='no_signal')))
                last_valid=time.monotonic()


if __name__=='__main__':
    main()
