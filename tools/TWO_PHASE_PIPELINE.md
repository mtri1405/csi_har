# Two-phase CSI detection (single receiver, model version 3)

## Decision flow

1. One CSI window is resampled using ESP timestamps, normalized separately by LTF,
   and smoothed with Savitzky–Golay 5 / polynomial order 2 **within that window**.
2. Phase 1 receives 13 presence features, including deviation from an EMPTY baseline.
   Training targets: `empty -> 0`, `standing -> 1`, `walking -> 1`.
3. Phase 1 uses `P(present) >= 0.5`. Live decisions optionally use majority voting
   over an odd number of windows (default 5). The UI shows both raw probability
   and the voted decision. Initial predictions wait until the vote buffer is full.
4. If the phase-1 decision is EMPTY, phase 2 is not called; its probabilities are
   cleared and the final result is EMPTY.
5. If the phase-1 decision is PRESENT, phase 2 receives 18 motion features from the
   same CSI window and predicts STANDING or WALKING. It is trained only on these
   two labels; no EMPTY rows are used by the phase-2 model.

## What was learned from the reference implementation

- `ThamKhao/.../phase1/presence/csi_utils.py`: fixed PHY layout, 107 selected
  LLTF/HT-LTF amplitudes, fieldwise median normalization, Savitzky–Golay smoothing,
  TRAIN-EMPTY reference, and 13 presence descriptors per receiver.
- `ThamKhao/.../phase1/python/05_train_presence.py`: a shallow, balanced presence
  Random Forest and an explicit probability threshold. This implementation retains
  Random Forest rather than introducing the reference's optional logistic model selection.
- `ThamKhao/.../phase1/python/07_live_presence.py`: optional startup EMPTY calibration,
  fresh post-calibration windows, stale-data rejection, probability display and voting.
- `ThamKhao/.../phase2/python/phase2_utils.py`: the 18 per-receiver motion descriptors.
  No RX2 values or cross-receiver features are synthesized.

Unlike whole-session reference exports, training and inference here both smooth
within each window so feature extraction is reproducible online.

## Data separation

All selected **training** EMPTY recordings contribute EMPTY examples. All selected
training STANDING and WALKING recordings contribute PRESENT examples. Validation
and test recordings remain held out. The default web manifest assigns whole
recordings per label in a train/val/test cycle; use a custom manifest to group by
day, subject or collection block when assessing generalization.

The baseline is fitted only on training EMPTY recordings. With the current room,
the automatic manifest produces 177 EMPTY and 236 PRESENT training windows;
PRESENT consists of 118 STANDING and 118 WALKING windows. Phase 2 uses the 236
person windows only. These counts will change when new data are added.

## Start the UI

```powershell
tools/venv/Scripts/python.exe -X utf8 tools/run.py web
```

After source updates, restart the web process and refresh the page.

The realtime panel offers:

- **Training EMPTY baseline**: starts without assuming that the room is empty.
- **Startup EMPTY calibration**: leave the sensing area during the countdown and
  keep it empty for the full calibration period (default 12 seconds). A gap or
  reset restarts the contiguous calibration segment. The reference is held fixed
  after calibration, is not fitted to predicted EMPTY windows, and does not alter
  the saved model. New geometry or a new room can still require new training data.
- **Vote count**: 1 disables voting, 3/5/7 smooth the phase-1 decision. Five windows
  add warm-up/transition latency; at 25 Hz the rounded 0.5-second hop is 0.48 seconds.

Missing CSI, timestamp reset, channel change, or >250ms gaps discard the current
window and vote history. The UI clears old phase-2 results on waiting/stopped/error
and no-signal states. Serial input keeps partial lines across timeouts.

## Outputs

- `tools/reports/features`: version-3 dataset with 31 ordered columns. `phase1_y`
  and `phase1_label` make the collapsed EMPTY/PRESENT targets explicit.
- `tools/reports/two_phase_rf/model.joblib`: complete cascade used by realtime.
- `phase1_presence.joblib`: separate presence model, threshold and training baseline.
- `phase2_motion.joblib`: separate STANDING/WALKING motion model.
- `metrics.json`: training counts, separate phase reports and end-to-end reports.

Version-2 models/datasets are rejected with instructions to rebuild. Exploratory
exports from previous reviews are not used for training.

Offline validation/test metrics describe individual windows with the training
baseline. They do not measure startup calibration or majority voting. In the
current split, presence validation includes substantial EMPTY false positives;
the high test score does not establish reliable behavior across days or rooms.

## Checks

```powershell
tools/venv/Scripts/python.exe -m unittest discover -s tools/tests -p test_*.py
```

Tests cover phase gating, no EMPTY in phase-2 training, feature agreement with
the reference, offline/streaming feature parity, voting transitions, calibration,
stale/reset handling, CLI routes, and the HTTP interface. UI view-model checks
also cover stale/EMPTY states clearing STANDING/WALKING probabilities.
