"""Capture clock diagnostics and serial timeout regression tests."""
import contextlib
import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csi import recording, webapp


class RecordingTests(unittest.TestCase):
    def test_countdown_finishes_before_capture(self):
        now = [0.0]
        def advance(seconds):
            now[0] += seconds
        log = io.StringIO()
        with patch.object(recording.time, 'monotonic', side_effect=lambda: now[0]), \
             patch.object(recording.time, 'sleep', side_effect=advance), \
             contextlib.redirect_stdout(log):
            recording.wait_for_countdown(3)
        import json
        self.assertEqual([json.loads(line)['remaining_s'] for line in log.getvalue().splitlines()], [3, 2, 1, 0])
        command = webapp.make_command('record', {'port': 'COM_TEST', 'countdown': 7})
        self.assertEqual(command[command.index('--countdown')+1], '7')
        self.assertEqual(command[command.index('--duration')+1], '30.0')

    def test_clock_mismatch_and_late_start_are_distinct(self):
        slow = recording.capture_timing_summary([(0, 100), (60, 20000100)], 60, 60)
        self.assertEqual(slow['esp_packet_span_s'], 20)
        self.assertEqual(len(slow['warnings']), 1)
        late = recording.capture_timing_summary([(40, 100), (60, 20000100)], 60, 60)
        self.assertEqual(late['pc_packet_span_s'], 20)
        self.assertEqual(len(late['warnings']), 1)
        normal = recording.capture_timing_summary([(0, 100), (60, 60000100)], 60, 60)
        self.assertEqual(normal['warnings'], [])

    def test_serial_partial_line_survives_timeout(self):
        row = ['CSI_DATA', 1, 100000, -50, -96, 6, 0, 1, 0, 0, 0, 256, 1, 0, str([20, 25]*128)]
        out = io.StringIO(); csv.writer(out).writerow(row)
        line = out.getvalue().encode()
        class Connection:
            now = 0
            in_waiting = 1
            chunks = iter([line[:50], b'', line[50:]])
            def reset_input_buffer(self): pass
            def close(self): pass
            def read(self, size):
                self.now += .1
                return next(self.chunks, b'')
        connection = Connection()
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(recording, 'DATA_ROOT', Path(tmp)), \
             patch.object(recording.serial, 'Serial', return_value=connection), \
             patch.object(recording.time, 'monotonic', side_effect=lambda: connection.now), \
             patch.object(sys, 'argv', ['record', '--duration', '.5']), \
             contextlib.redirect_stdout(io.StringIO()):
            recording.main()
            path = next(Path(tmp).rglob('*.csv'))
            with path.open(encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['timestamp'], '100000')
            self.assertAlmostEqual(float(rows[0]['pc_elapsed_s']), .3)
            from csi.viewer import csv_preview
            self.assertEqual(csv_preview(path)['total'], 1)

    def test_web_passes_sixty_seconds_and_handles_summary(self):
        command = webapp.make_command('record', {'port': 'COM_TEST', 'duration': 60})
        self.assertEqual(command[command.index('--duration')+1], '60.0')
        identifier = 'test_capture_summary'
        webapp.jobs[identifier] = dict(status='running', action='record', log='')
        try:
            webapp.run_command(identifier, [sys.executable, '-c',
                'print(\'{"type":"capture_summary","pc_elapsed_s":60,"esp_packet_span_s":20}\')'])
            job = webapp.jobs[identifier]
            self.assertEqual(job['capture_summary']['esp_packet_span_s'], 20)
            self.assertNotIn('prediction', job)
        finally:
            del webapp.jobs[identifier]


if __name__ == '__main__':
    unittest.main()
