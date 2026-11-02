# CSI subcarrier selection and preprocessing

## Input layout

Each accepted CSI frame contains 128 interleaved I/Q integers:

```text
imaginary_0, real_0, imaginary_1, real_1, ..., imaginary_63, real_63
```

The parser converts them to 64 amplitude channels:

```text
amplitude_i = sqrt(imaginary_i^2 + real_i^2)
```

## Removed subcarriers

The following raw indices are removed for every frame:

```text
0, 1, 2, 3, 4, 5, 11, 25, 32, 39, 53, 58, 59, 60, 61, 62, 63
```

The remaining 47 indices are:

```text
6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
23, 24, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 40, 41,
42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57
```

The mask is defined by `REMOVED_SUBCARRIER_INDICES` and
`SUBCARRIER_INDICES` in [`csi_preprocessing.py`](./csi_preprocessing.py).
Keeping a fixed mask is important: every recording must use the same
subcarrier coordinate system.

## Preprocessing pipeline

For both dataset creation and realtime inference:

1. Parse 128 I/Q values and calculate 64 amplitudes.
2. Reject incomplete or non-finite frames.
3. Remove the 17 listed subcarriers, leaving 47 channels.
4. Apply a Hampel filter per channel to replace temporal spikes.
5. Apply a fourth-order zero-phase Butterworth low-pass filter.
   The current configuration uses a 100 Hz sample rate and 10 Hz cutoff.
6. Take a 60-frame window.
7. Apply PCA with 10 components.
8. Apply min-max normalization to `[0, 1]`.

The CNN therefore receives a tensor with shape:

```text
(batch, 1, 60 frames, 10 PCA components)
```

The realtime and dataset paths call the same `preprocess_window` function so
that training and inference use identical transformations. A new dataset and
checkpoint must be generated after changing this mask or the PCA setting.

## Why use 10 PCA components?

PCA is applied after subcarrier selection and temporal filtering. It finds
directions that explain the largest variance across the 47 retained
subcarriers and projects each 60-frame window onto the first 10 directions.
This reduces the input width from 47 to 10 while preserving the dominant
motion-related variation.

Ten is a project configuration, not a universal optimum. It is a practical
trade-off between information retention, noise reduction, model size and
inference speed. The number should be validated by comparing explained
variance and held-out validation/test F1; if too much variance or accuracy is
lost, increase it and retrain.

## Visualization

To select one CSV file explicitly:

```powershell
python visualize_csi.py --csv Data\walking\sample.csv
```

The CSV viewer provides independent `1s`, `5s` and `15s` controls for:

- raw CSI amplitude (64 channels);
- processed CSI after selection, filtering, PCA and normalization (10 components).

The serial viewer remains available with:

```powershell
python visualize_csi.py --port COM8 --baudrate 2000000 --model csi_cnn_model.pth
```

To inspect the tensor written by `dataset_pipeline.py`, run:

```powershell
python visualize_csi.py --dataset csi_dataset.npz --sample 0
```

The viewer prints the dataset shape and value range, then displays one sample
as a `(60, 10)` heatmap and ten component timelines. Use `Sample trước` and
`Sample sau` to inspect other windows and their labels. This is the data passed
to the CNN, with the extra channel dimension added by `train.py`.
