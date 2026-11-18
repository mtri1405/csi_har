"""Shared strict serial protocol and timestamp handling."""
import ast
import csv
import numpy as np
from csi.signal import raw_iq_to_amplitude

HEADER = ['type', 'seq', 'timestamp', 'rssi', 'noise_floor', 'channel',
          'second_channel', 'sig', 'mcs', 'cwb', 'stbc', 'len',
          'first_word_invalid', 'dropped', 'data']

def parse_packet(line, layout='ht20_256'):
    try:
        fields = next(csv.reader([line.strip()]))
        if len(fields) != len(HEADER) or fields[0] != 'CSI_DATA':
            return None
        packet = dict(zip(HEADER, fields))
        for name in HEADER[1:-1]:
            packet[name] = int(packet[name])
        if (packet['sig'] != 1 or packet['cwb'] or packet['stbc']
                or packet['second_channel']):
            return None
        amplitude = raw_iq_to_amplitude(packet['data'], layout)
        if packet['len'] != len(ast.literal_eval(packet['data'])):
            return None
        if layout == 'htltf_128' and packet['first_word_invalid']:
            return None  # selected carrier 1 overlaps invalid first four bytes
        packet['amplitude'] = amplitude
        return packet
    except (ValueError, TypeError, SyntaxError, csv.Error, StopIteration):
        return None

def unwrap_timestamps(values):
    """Preserve acquisition order; unwrap uint32 rollover, reject resets/reordering."""
    t = np.asarray(values, dtype=float)
    if not np.isfinite(t).all() or len(t) == 0:
        raise ValueError('Invalid timestamps')
    out = t.copy()
    offset = 0
    for i in range(1, len(t)):
        delta = t[i] - t[i-1]
        if delta < -(2**31):
            offset += 2**32
        elif delta <= 0:
            raise ValueError('Timestamp reset, duplicate or out-of-order packet; split recording')
        out[i] += offset
    return out
