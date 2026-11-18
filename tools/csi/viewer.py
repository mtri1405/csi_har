"""Bounded CSI previews for the local web panel (no model required)."""
import argparse
import csv
import io
import json
import time
import numpy as np
from csi.protocol import HEADER, parse_packet, unwrap_timestamps
from csi.signal import fieldwise_ltf_normalize


def preview(packets, total=None):
    if not packets:
        return {'frames': [], 'count': 0, 'total': total or 0}
    timestamps = unwrap_timestamps([p['timestamp'] for p in packets]) * 1e-6
    times = timestamps - timestamps[0]
    amplitude = np.stack([p['amplitude'] for p in packets])
    normalized = fieldwise_ltf_normalize(amplitude)
    return {'frames': [{'t': float(t), 'amplitude': a.tolist(), 'normalized': n.tolist(),
                         'rssi': int(p['rssi'])}
                        for t, a, n, p in zip(times, amplitude, normalized, packets)],
            'count': len(packets), 'total': total or len(packets),
            'duration_s': float(times[-1]),
            'rate_hz': float((len(times)-1)/times[-1]) if times[-1] else 0,
            'max_gap_s': float(np.diff(times).max()) if len(times)>1 else 0}


def csv_preview(path, limit=1000):
    # Parse sequentially, then thin uniformly for display only, never for training.
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError('File vượt 50 MB; hãy chia thành các phiên nhỏ hơn để xem.')
    packets=[]
    rejected=0
    with path.open(encoding='utf-8-sig') as handle:
        for fields in csv.reader(handle):
            if not fields or fields[0] == 'type':
                continue
            # New recordings append host timing columns; serial parser stays strict.
            if len(fields) not in (len(HEADER), len(HEADER)+2):
                rejected += 1
                continue
            line = io.StringIO()
            csv.writer(line).writerow(fields[:len(HEADER)])
            packet=parse_packet(line.getvalue())
            if packet is None:
                rejected+=1
            else:
                packets.append(packet)
    if not packets:
        raise ValueError('Không có CSI HT20 256 byte hợp lệ trong file.')
    # Validate the complete clock before thinning; keep rate/gaps from original data.
    times=unwrap_timestamps([p['timestamp'] for p in packets])*1e-6
    total=len(packets)
    selected=np.unique(np.linspace(0,total-1,min(limit,total),dtype=int))
    result=preview([packets[i] for i in selected],total)
    for i, index in enumerate(selected):
        previous = selected[i-1] if i else index
        result['frames'][i]['gap_before'] = bool((np.diff(times[previous:index+1]) > .25).any())
    result.update(rejected=rejected,
                  rate_hz=float((total-1)/(times[-1]-times[0])) if total>1 else 0,
                  max_gap_s=float(np.diff(times).max()) if total>1 else 0)
    return result


def monitor(port, baudrate):
    import serial
    with serial.Serial(port,baudrate,timeout=1) as connection:
        connection.reset_input_buffer()
        last_emit=0
        while True:
            packet=parse_packet(connection.readline().decode('utf-8',errors='replace'))
            now=time.monotonic()
            if packet and now-last_emit >= .1:
                print(json.dumps({'type':'csi_frame', 'timestamp':packet['timestamp'],
                      'rssi':packet['rssi'], 'channel':packet['channel'],
                      'amplitude':packet['amplitude'].tolist()}),flush=True)
                last_emit=now


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',required=True)
    p.add_argument('--baudrate',type=int,default=2000000)
    args=p.parse_args()
    monitor(args.port,args.baudrate)
