# CSI subcarrier selection

## Purpose

The recorder stores 128 integer values per CSI frame in HT20 mode. They are
interpreted as 64 interleaved complex samples:

```text
imaginary_0, real_0, imaginary_1, real_1, ..., imaginary_63, real_63
```

The dataset pipeline converts each complex sample to amplitude and then removes
subcarriers that cannot provide useful, stable information for the CNN.

## Selection policy

The policy is implemented in
[`csi_preprocessing.py`](./csi_preprocessing.py):

| Group | Raw indices | Decision | Reason |
| --- | ---: | --- | --- |
| Low-index noise band | `0..5` | Remove | Edge/guard samples are outside the useful occupied band and are commonly zero or very weak. |
| Noise carriers | `11`, `25`, `32`, `39`, `53` | Remove | Project-level noisy carrier mask. |
| High-index noise band | `58..63` | Remove | Edge/guard samples are outside the useful occupied band and are commonly zero or very weak. |
| All-zero columns | Any remaining index | Remove | If the complex amplitude is zero for every valid frame in a CSV, that column is not carrying usable CSI in that recording. |
| Remaining columns | All other indices | Keep | These are retained as the model input. |

With no additional all-zero columns, this leaves **47 subcarriers**
(`64 - 6 - 5 - 6`). If a recording contains additional columns that are zero
for its entire duration, the pipeline removes them too.

## Why remove columns, not individual values?

A zero amplitude in one frame is not automatically a bad subcarrier. It can be
caused by a temporary deep fade, packet corruption, or a valid instantaneous
channel response. Removing individual values would shift the time/subcarrier
matrix and make frames impossible to align.

Therefore, the pipeline removes a subcarrier only when the whole column is zero
across the valid frames of a CSV. The resulting width must be identical for all
CSV files. If files produce different widths, dataset creation stops with an
explicit error rather than silently creating incompatible training data.

## Important mapping assumption

The current layout is the direct 64-value HT20 sequence produced by the
receiver. The exact order can vary with ESP32 chip, driver, and firmware, so
the project-level mask should be revalidated if the hardware or CSI settings
change.

Verification steps:

1. Confirm that every accepted frame has 128 CSI integers (64 complex samples).
2. Inspect the amplitude plot before and after filtering.
3. Check that edge columns are consistently empty/weak and that the center
   column is the DC/null carrier.
4. Compare the model accuracy with and without the mask on a held-out test set.

The realtime predictor removes the fixed 64-channel mask as well, so its input
width matches the dataset for the normal case. It does not dynamically remove
all-zero columns from one frame because a single frame is not enough
evidence that a subcarrier is permanently unusable.

## References

- [Espressif ESP-CSI repository](https://github.com/espressif/esp-csi)
- [Espressif CSI data parsing example](https://github.com/espressif/esp-csi/blob/master/examples/get-started/tools/csi_data_read_parse.py)
- [IEEE 802.11 OFDM overview](https://standards.ieee.org/ieee/802.11/10561/)

These references describe the CSI tool and the wireless/OFDM context. The exact
index mask remains a project-level choice and must be validated against the
actual ESP32 firmware output.
