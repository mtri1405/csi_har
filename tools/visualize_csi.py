"""Visualize the first CSI recording of every label before and after processing.

Run from the tools directory:
    python visualize_csi.py

The window shows one line per PCA component and has buttons for 1, 15 and 30
seconds. Raw amplitudes are collected directly as 64 HT20 channels.
"""

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from dataset_pipeline import ACTIONS, LABEL_NAMES, parse_csi_file
from csi_preprocessing import (
    SAMPLE_RATE_HZ,
    butterworth_filter,
    min_max_scale,
    pca_reduce,
    preprocess_csi,
    remove_unusable_subcarriers,
)


MIN_PROCESSING_FRAMES = 32
DISPLAY_DURATIONS = (1, 15, 30)


@dataclass
class LabelRecording:
    label: int
    name: str
    path: str | None
    raw: np.ndarray | None
    processed: np.ndarray | None
    keep_indices: list[int] | None


def _first_csv(data_dir: str, action_name: str) -> str | None:
    folder = os.path.join(data_dir, action_name)
    if not os.path.isdir(folder):
        return None
    files = sorted(
        os.path.join(folder, filename)
        for filename in os.listdir(folder)
        if filename.lower().endswith(".csv")
    )
    return files[0] if files else None


def load_recordings(data_dir: str) -> list[LabelRecording]:
    recordings = []
    for action_name, label in ACTIONS.items():
        path = _first_csv(data_dir, action_name)
        if path is None:
            recordings.append(
                LabelRecording(label, LABEL_NAMES[label], None, None, None, None)
            )
            continue

        raw = parse_csi_file(path)
        if raw.size == 0:
            recordings.append(
                LabelRecording(label, LABEL_NAMES[label], path, raw, None, None)
            )
            continue

        processing_input = raw
        if len(processing_input) < MIN_PROCESSING_FRAMES:
            padding = MIN_PROCESSING_FRAMES - len(processing_input)
            processing_input = np.pad(processing_input, ((0, padding), (0, 0)), mode="edge")

        filtered = preprocess_csi(processing_input)
        cleaned, _, keep_indices = remove_unusable_subcarriers(filtered)
        processed = min_max_scale(pca_reduce(butterworth_filter(cleaned)))[: len(raw)]
        recordings.append(
            LabelRecording(
                label,
                LABEL_NAMES[label],
                path,
                raw,
                processed,
                [int(index) for index in keep_indices],
            )
        )
    return recordings


def _slice_for_duration(data: np.ndarray, duration: int) -> np.ndarray:
    frame_count = max(1, int(round(duration * SAMPLE_RATE_HZ)))
    return data[:frame_count]


def _draw_line_graph(axis, data: np.ndarray, title: str, duration: int):
    if data is None or data.size == 0:
        axis.text(0.5, 0.5, "Không có dữ liệu", ha="center", va="center")
        axis.set_title(title)
        axis.set_axis_off()
        return None

    shown = _slice_for_duration(data, duration)
    time_axis = np.arange(len(shown), dtype=np.float64) / SAMPLE_RATE_HZ
    axis.plot(
        time_axis,
        shown,
        linewidth=0.7,
        alpha=0.35,
    )
    axis.set_title(title)
    axis.set_xlabel("Thời gian (s)")
    axis.set_ylabel("Biên độ CSI")
    axis.set_xlim(0, max(duration, 1))
    axis.grid(True, alpha=0.25)
    return None


def create_visualization(recordings: list[LabelRecording]) -> None:
    figure, axes = plt.subplots(
        len(recordings), 2, figsize=(15, max(7, 3.5 * len(recordings))), squeeze=False
    )
    figure.subplots_adjust(bottom=0.14, hspace=0.55, wspace=0.25)
    current_duration = DISPLAY_DURATIONS[0]

    def redraw(duration: int) -> None:
        nonlocal current_duration
        current_duration = duration
        for row, recording in enumerate(recordings):
            for axis in axes[row]:
                axis.clear()
            if recording.path is None:
                message = f"{recording.name}\nChưa tìm thấy file CSV đầu tiên"
                axes[row, 0].text(0.5, 0.5, message, ha="center", va="center")
                axes[row, 0].set_axis_off()
                axes[row, 1].set_axis_off()
                continue
            filename = os.path.basename(recording.path)
            _draw_line_graph(
                axes[row, 0],
                recording.raw,
                f"{recording.name} - chưa xử lý\n{filename}",
                current_duration,
            )
            _draw_line_graph(
                axes[row, 1],
                recording.processed,
                f"{recording.name} - đã xử lý\n10 PCA components",
                current_duration,
            )
        figure.suptitle(
            f"So sánh đồ thị CSI theo nhãn - cửa sổ {current_duration}s "
            f"(tần số mẫu {SAMPLE_RATE_HZ:g} Hz)",
            fontsize=14,
        )
        figure.canvas.draw_idle()

    button_axes = []
    for index, duration in enumerate(DISPLAY_DURATIONS):
        button_axis = figure.add_axes([0.34 + index * 0.12, 0.035, 0.1, 0.05])
        button = Button(button_axis, f"{duration}s")
        button.on_clicked(lambda _event, value=duration: redraw(value))
        button_axes.append(button)

    redraw(current_duration)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Xem file CSV đầu tiên của mỗi nhãn trước và sau xử lý CSI."
    )
    parser.add_argument(
        "--data-dir",
        default="Data",
        help="Thư mục chứa các thư mục nhãn (mặc định: Data).",
    )
    args = parser.parse_args()

    recordings = load_recordings(args.data_dir)
    for recording in recordings:
        if recording.path is None:
            print(f"[!] {recording.name}: không tìm thấy file CSV.")
        elif recording.raw is None or recording.raw.size == 0:
            print(f"[!] {recording.name}: file không có frame hợp lệ: {recording.path}")
        else:
            print(
                f"[OK] {recording.name}: {recording.path} "
                f"({len(recording.raw)} frame)"
            )
    create_visualization(recordings)


if __name__ == "__main__":
    main()
