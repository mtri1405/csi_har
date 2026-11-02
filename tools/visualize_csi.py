"""Visualize CSI from a serial port in real time.

Run from the tools directory:
    python visualize_csi.py --port COM8 --baudrate 2000000

The dashboard shows raw CSI, a raw-amplitude heatmap, filtered CSI and an
optional prediction timeline. ``--data-dir`` keeps the old file-based preview
available when no serial port is specified.
"""

import argparse
import os
import threading
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import serial
import torch
from matplotlib.widgets import Button, RadioButtons, TextBox

from csi_model import CSI_CNN
from csi_preprocessing import (
    RAW_SUBCARRIERS,
    PCA_COMPONENTS,
    SAMPLE_RATE_HZ,
    TARGET_SUBCARRIERS,
    WINDOW_SIZE,
    preprocess_series,
    preprocess_window,
)
from csi_visualization import apply_filter, parse_csi_line
from dataset_pipeline import parse_csi_file


DISPLAY_DURATIONS = (1, 5, 15)
BUFFER_SECONDS = max(DISPLAY_DURATIONS)
MAX_CSV_PREVIEW_FILES = 3
FILTER_MODES = {
    "Không lọc": "none",
    "Hampel": "hampel",
    "Butterworth": "butterworth",
    "Hampel + Butterworth": "hampel_butterworth",
    "Full pipeline": "pipeline",
}


@dataclass
class SerialState:
    port: str
    baudrate: int
    stop_event: threading.Event
    thread: threading.Thread | None = None
    serial_error: str | None = None


def _draw_chart(axis, data: np.ndarray, title: str, duration: int) -> None:
    axis.clear()
    if data.size == 0:
        axis.text(0.5, 0.5, "Đang chờ dữ liệu CSI...", ha="center", va="center")
        axis.set_axis_off()
        return

    shown = data[-max(1, int(duration * SAMPLE_RATE_HZ)) :]
    time_axis = np.arange(-len(shown) + 1, 1) / SAMPLE_RATE_HZ
    axis.plot(time_axis, shown, linewidth=0.7, alpha=0.55)
    axis.set_title(title)
    axis.set_xlabel("Thời gian tương đối (s)")
    axis.set_ylabel("Biên độ / thành phần")
    axis.set_xlim(-max(duration, 1), 0)
    axis.grid(True, alpha=0.25)


def _draw_grouped_chart(
    axis,
    data: np.ndarray,
    title: str,
    duration: int,
    selected_group: int | None = None,
    max_groups: int = 8,
) -> None:
    """Draw all channels or one equal-sized subcarrier group."""
    axis.clear()
    if data.size == 0:
        axis.text(0.5, 0.5, "Không có dữ liệu", ha="center", va="center")
        axis.set_axis_off()
        return

    channel_count = data.shape[1]
    group_count = max(
        (candidate for candidate in range(min(max_groups, channel_count), 0, -1)
         if channel_count % candidate == 0),
        default=1,
    )
    group_size = channel_count // group_count
    shown = data[-max(1, int(duration * SAMPLE_RATE_HZ)) :]
    time_axis = np.arange(-len(shown) + 1, 1) / SAMPLE_RATE_HZ
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, group_count))
    group_indices = (
        range(group_count)
        if selected_group is None
        else [selected_group]
    )
    for group_index in group_indices:
        start = group_index * group_size
        end = start + group_size
        axis.plot(
            time_axis,
            shown[:, start:end],
            color=colors[group_index],
            linewidth=0.7,
            alpha=0.65,
            label=f"Nhóm {group_index + 1}: SC {start}-{end - 1}",
        )
        for line in axis.lines[-group_size:]:
            line.set_label(
                f"Nhóm {group_index + 1}: SC {start}-{end - 1}"
                if line is axis.lines[-group_size] else "_nolegend_"
            )
    axis.set_title(title)
    axis.set_xlabel("Thời gian tương đối (s)")
    axis.set_ylabel("Biên độ / thành phần")
    axis.set_xlim(-max(duration, 1), 0)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper left", fontsize="x-small", ncol=2)


class RealtimeVisualizer:
    def __init__(self, port: str, baudrate: int, model_path: str | None = None):
        self.port = port
        self.baudrate = baudrate
        self.model = None
        self.labels = {}
        self.pca_mean = None
        self.pca_components = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prediction_history: deque[tuple[float, int, float]] = deque(maxlen=300)
        self.total_frames = 0
        self.last_prediction_frame = 0
        if model_path:
            self.model, self.labels, self.pca_mean, self.pca_components = (
                self._load_model(model_path)
            )
        self.duration = DISPLAY_DURATIONS[0]
        self.filter_mode = "none"
        self.running = False
        self.lock = threading.Lock()
        self.samples: deque[np.ndarray] = deque(
            maxlen=int(BUFFER_SECONDS * SAMPLE_RATE_HZ)
        )
        self.serial_state: SerialState | None = None
        self.status_message = "Chưa kết nối serial"
        self.status_color = "tab:blue"

        self.figure, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.figure.subplots_adjust(
            left=0.06, right=0.98, bottom=0.23, top=0.9, wspace=0.2, hspace=0.35
        )
        self.status = self.figure.text(
            0.06, 0.92, self.status_message, color=self.status_color
        )

        port_axis = self.figure.add_axes([0.06, 0.08, 0.12, 0.055])
        baud_axis = self.figure.add_axes([0.20, 0.08, 0.12, 0.055])
        self.port_box = TextBox(port_axis, "Cổng: ", initial=port)
        self.baud_box = TextBox(baud_axis, "Baud: ", initial=str(baudrate))

        connect_axis = self.figure.add_axes([0.35, 0.08, 0.12, 0.055])
        self.connect_button = Button(connect_axis, "Kết nối")
        self.connect_button.on_clicked(self.toggle_connection)

        duration_axes = [
            self.figure.add_axes([0.51 + index * 0.06, 0.08, 0.05, 0.055])
            for index in range(len(DISPLAY_DURATIONS))
        ]
        self.duration_buttons = []
        for axis, duration in zip(duration_axes, DISPLAY_DURATIONS):
            button = Button(axis, f"{duration}s")
            button.on_clicked(lambda _event, value=duration: self.set_duration(value))
            self.duration_buttons.append(button)

        filter_axis = self.figure.add_axes([0.76, 0.025, 0.21, 0.17])
        self.filter_buttons = RadioButtons(filter_axis, list(FILTER_MODES), active=0)
        self.filter_buttons.on_clicked(self.set_filter)
        filter_axis.set_title("Bộ lọc")

        self.timer = self.figure.canvas.new_timer(interval=10)
        self.timer.add_callback(self.redraw)
        self.timer.start()
        self.redraw()

    def set_duration(self, duration: int) -> None:
        self.duration = duration

    def set_filter(self, label: str) -> None:
        self.filter_mode = FILTER_MODES[label]

    def toggle_connection(self, _event) -> None:
        if self.running:
            self.stop_serial()
            return
        try:
            baudrate = int(self.baud_box.text.strip())
            if baudrate <= 0:
                raise ValueError("baudrate phải lớn hơn 0")
        except ValueError as exc:
            self.set_status(f"Lỗi cấu hình: {exc}", "tab:red")
            return
        self.port = self.port_box.text.strip()
        if not self.port:
            self.set_status("Lỗi cấu hình: chưa nhập cổng serial", "tab:red")
            return

        state = SerialState(self.port, baudrate, threading.Event())
        state.thread = threading.Thread(
            target=self.read_serial, args=(state,), daemon=True
        )
        self.serial_state = state
        self.running = True
        self.connect_button.label.set_text("Ngắt kết nối")
        state.thread.start()

    def stop_serial(self) -> None:
        if self.serial_state is not None:
            self.serial_state.stop_event.set()
            if self.serial_state.thread is not None:
                self.serial_state.thread.join(timeout=2)
        self.running = False
        self.connect_button.label.set_text("Kết nối")
        self.set_status("Đã ngắt kết nối", "tab:blue")

    def set_status(self, message: str, color: str = "tab:blue") -> None:
        self.status_message = message
        self.status_color = color
        self.status.set_text(message)
        self.status.set_color(color)

    def _load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint không có metadata preprocessing.")
        metadata = checkpoint.get("metadata", {})
        if (
            metadata.get("window_size") != WINDOW_SIZE
            or metadata.get("raw_subcarriers") != RAW_SUBCARRIERS
            or metadata.get("target_subcarriers") != TARGET_SUBCARRIERS
            or metadata.get("pca_components") != PCA_COMPONENTS
            or "pca_mean" not in metadata
            or "pca_components_matrix" not in metadata
        ):
            raise ValueError("Checkpoint không khớp cấu hình cửa sổ/subcarrier.")
        labels = {int(key): value for key, value in metadata["labels"].items()}
        model = CSI_CNN(num_classes=int(metadata["num_classes"])).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return (
            model,
            labels,
            np.asarray(metadata["pca_mean"], dtype=np.float64),
            np.asarray(metadata["pca_components_matrix"], dtype=np.float64),
        )

    def _predict_latest(self, raw: np.ndarray, frame_count: int) -> None:
        if (
            self.model is None
            or len(raw) < WINDOW_SIZE
            or frame_count == self.last_prediction_frame
        ):
            return
        self.last_prediction_frame = frame_count
        if frame_count % 2:
            return
        window = preprocess_window(
            raw[-WINDOW_SIZE:], self.pca_mean, self.pca_components
        )
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            probabilities = torch.softmax(
                self.model(tensor.to(self.device)), dim=1
            )[0].cpu().numpy()
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])
        timestamp = frame_count / SAMPLE_RATE_HZ
        self.prediction_history.append((timestamp, class_id, confidence))

    def _draw_heatmap(self, axis, data: np.ndarray) -> None:
        axis.clear()
        shown = data[-max(1, int(self.duration * SAMPLE_RATE_HZ)) :]
        image = axis.imshow(
            shown.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )
        axis.set_title("CSI raw heatmap")
        axis.set_xlabel("Frame")
        axis.set_ylabel("Subcarrier")

    def _draw_prediction_timeline(self, axis) -> None:
        axis.clear()
        if self.model is None:
            axis.text(
                0.5,
                0.5,
                "Dùng --model csi_cnn_model.pth để bật prediction timeline",
                ha="center",
                va="center",
                wrap=True,
            )
            axis.set_axis_off()
            return
        if not self.prediction_history:
            axis.text(0.5, 0.5, "Đang chờ đủ 60 frame để dự đoán...", ha="center")
            axis.set_axis_off()
            return
        timestamps, classes, confidences = zip(*self.prediction_history)
        axis.step(timestamps, classes, where="post")
        axis.scatter(timestamps, classes, c=confidences, cmap="viridis", vmin=0, vmax=1)
        unique_classes = sorted(set(classes))
        axis.set_yticks(unique_classes)
        axis.set_yticklabels([self.labels.get(value, str(value)) for value in unique_classes])
        axis.set_xlabel("Thời gian (s)")
        axis.set_ylabel("Nhãn")
        axis.set_title(f"Prediction timeline | confidence: {confidences[-1]:.1%}")
        axis.grid(True, alpha=0.25)

    def read_serial(self, state: SerialState) -> None:
        try:
            with serial.Serial(
                state.port,
                state.baudrate,
                timeout=0.2,
                dsrdtr=False,
                rtscts=False,
            ) as connection:
                try:
                    connection.setDTR(False)
                    connection.setRTS(False)
                except (AttributeError, serial.SerialException):
                    pass
                connection.reset_input_buffer()
                self.status_message = f"Đã kết nối {state.port} @ {state.baudrate}"
                self.status_color = "tab:green"
                while not state.stop_event.is_set():
                    raw_line = connection.readline()
                    if not raw_line:
                        continue
                    sample = parse_csi_line(
                        raw_line.decode("utf-8", errors="ignore").strip()
                    )
                    if sample is not None:
                        with self.lock:
                            self.samples.append(sample)
                            self.total_frames += 1
        except (serial.SerialException, OSError) as exc:
            state.serial_error = str(exc)
            self.running = False
            self.status_message = f"Lỗi serial: {exc}"
            self.status_color = "tab:red"

    def redraw(self) -> None:
        self.status.set_text(self.status_message)
        self.status.set_color(self.status_color)
        if not self.running and self.serial_state is not None:
            self.connect_button.label.set_text("Kết nối")
        with self.lock:
            raw = np.asarray(self.samples, dtype=np.float64)
            frame_count = self.total_frames
        filtered = apply_filter(raw, self.filter_mode) if raw.size else raw
        self._predict_latest(raw, frame_count)

        if raw.size == 0:
            for ax in self.axes.flat:
                ax.clear()
                ax.text(0.5, 0.5, "Đang chờ dữ liệu CSI...", ha="center", va="center")
                ax.set_axis_off()
        elif raw.ndim == 2 and raw.shape[1] == RAW_SUBCARRIERS:
            filter_label = next(label for label, mode in FILTER_MODES.items() if mode == self.filter_mode)
            _draw_chart(self.axes[0, 0], raw, "CSI chưa lọc - 64 subcarrier", self.duration)
            self._draw_heatmap(self.axes[0, 1], raw)
            _draw_chart(self.axes[1, 0], filtered, f"CSI đã lọc ({filter_label}) - 64 subcarrier", self.duration)
            self._draw_prediction_timeline(self.axes[1, 1])
        else:
            # Fallback for unexpected shape
            _draw_chart(
                self.axes[0, 0],
                raw,
                f"CSI chưa lọc ({raw.shape[1] if raw.ndim == 2 else 0} subcarrier)",
                self.duration,
            )
            _draw_chart(
                self.axes[1, 0],
                filtered,
                f"CSI đã lọc: {next(label for label, mode in FILTER_MODES.items() if mode == self.filter_mode)}",
                self.duration,
            )
            for ax in [self.axes[0, 1], self.axes[1, 1]]:
                ax.clear()
                ax.set_axis_off()

        self.figure.suptitle(
            f"CSI realtime | {len(raw)} frame | tốc độ mẫu {SAMPLE_RATE_HZ:g} Hz"
        )
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        try:
            plt.show()
        finally:
            self.stop_serial()


def show_csv_preview(
    data_dir: str | None = None,
    csv_files: list[str] | None = None,
    pca_source: str | None = None,
) -> None:
    from dataset_pipeline import ACTIONS, LABEL_NAMES

    initial_files = list(csv_files or [])
    if len(initial_files) > MAX_CSV_PREVIEW_FILES:
        raise ValueError(f"Chỉ hỗ trợ xem tối đa {MAX_CSV_PREVIEW_FILES} file cùng lúc.")
    if not initial_files and data_dir:
        for action_name, label in ACTIONS.items():
            folder = os.path.join(data_dir or "", action_name)
            try:
                filename = sorted(
                    name for name in os.listdir(folder) if name.lower().endswith(".csv")
                )[0]
            except (FileNotFoundError, IndexError):
                continue
            initial_files.append(os.path.join(folder, filename))
            if len(initial_files) == MAX_CSV_PREVIEW_FILES:
                break

    file_paths = (initial_files + [None] * MAX_CSV_PREVIEW_FILES)[:MAX_CSV_PREVIEW_FILES]
    pca_state = None
    if pca_source:
        checkpoint = torch.load(pca_source, map_location="cpu")
        metadata = checkpoint.get("metadata", {})
        pca_state = (
            np.asarray(metadata["pca_mean"], dtype=np.float64),
            np.asarray(metadata["pca_components_matrix"], dtype=np.float64),
        )
    figure, axes = plt.subplots(
        MAX_CSV_PREVIEW_FILES, 2, figsize=(15, 4 * MAX_CSV_PREVIEW_FILES), squeeze=False
    )
    figure.subplots_adjust(bottom=0.30, hspace=0.55, wspace=0.25)
    raw_duration = DISPLAY_DURATIONS[0]
    processed_duration = DISPLAY_DURATIONS[0]
    preview_filter = "none"
    selected_group = None
    scroll_offset = 0

    def redraw() -> None:
        for row, file_path in enumerate(file_paths):
            for col in range(2):
                axes[row, col].clear()
                axes[row, col].set_axis_on()
            if not file_path:
                axes[row, 0].text(0.5, 0.5, f"Plot {row + 1}: chưa chọn file", ha="center", va="center")
                axes[row, 1].text(0.5, 0.5, "Chưa có dữ liệu", ha="center", va="center")
                axes[row, 0].set_axis_off()
                axes[row, 1].set_axis_off()
                continue
            label = os.path.basename(file_path)
            raw = parse_csi_file(file_path)
            if raw.size == 0 or raw.ndim != 2 or raw.shape[1] != RAW_SUBCARRIERS:
                for col in range(2):
                    ax = axes[row, col]
                    ax.text(0.5, 0.5, "Không hỗ trợ định dạng này", ha="center", va="center")
                    ax.set_axis_off()
                continue
            filtered = apply_filter(raw, preview_filter)
            processed = preprocess_series(raw, *(pca_state or (None, None)))
            _draw_grouped_chart(
                axes[row, 0],
                filtered,
                f"{label} - CSI ({preview_filter}, {group_title()}, nhóm màu)",
                raw_duration,
                selected_group=selected_group,
            )
            _draw_chart(
                axes[row, 1], processed,
                f"{label} - tiền xử lý + PCA ({processed.shape[1]} component)",
                processed_duration,
            )
        figure.suptitle(
            f"So sánh CSI từ CSV | raw: {raw_duration}s | processed: {processed_duration}s "
            f"(tần số mẫu {SAMPLE_RATE_HZ:g} Hz)"
        )
        figure.canvas.draw_idle()

    def group_title() -> str:
        if selected_group is None:
            return "64 subcarrier"
        start = selected_group * 8
        return f"subcarrier {start}-{start + 7}"

    def choose_file(row: int) -> None:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title=f"Chọn CSV cho plot {row + 1}",
            initialdir=data_dir or os.getcwd(),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
        if selected:
            file_paths[row] = selected
            redraw()

    def set_filter(label: str) -> None:
        nonlocal preview_filter
        preview_filter = FILTER_MODES[label]
        redraw()

    def set_group(label: str) -> None:
        nonlocal selected_group
        selected_group = None if label == "Tất cả" else int(label.split()[-1]) - 1
        redraw()

    def scroll_controls(step: int) -> None:
        nonlocal scroll_offset
        scroll_offset = max(0, min(2, scroll_offset + step))
        for control in scrollable_controls:
            control.ax.set_visible(control.row == scroll_offset)
        figure.canvas.draw_idle()

    def on_scroll(event) -> None:
        if event.inaxes in axes.flat:
            scroll_controls(1 if event.button == "down" else -1)

    controls = []
    scrollable_controls = []
    for row in range(MAX_CSV_PREVIEW_FILES):
        button_axis = figure.add_axes([0.02, 0.18 - row * 0.05, 0.18, 0.04])
        button = Button(button_axis, f"Chọn file cho plot {row + 1}")
        button.on_clicked(lambda _event, index=row: choose_file(index))
        button.row = row
        controls.append(button)
        scrollable_controls.append(button)

    scroll_up_axis = figure.add_axes([0.02, 0.235, 0.085, 0.035])
    scroll_down_axis = figure.add_axes([0.115, 0.235, 0.085, 0.035])
    scroll_up = Button(scroll_up_axis, "Lên")
    scroll_down = Button(scroll_down_axis, "Xuống")
    scroll_up.on_clicked(lambda _event: scroll_controls(-1))
    scroll_down.on_clicked(lambda _event: scroll_controls(1))
    controls.extend([scroll_up, scroll_down])

    for index, selected_duration in enumerate(DISPLAY_DURATIONS):
        button_axis = figure.add_axes([0.28 + index * 0.09, 0.08, 0.07, 0.05])
        button = Button(button_axis, f"{selected_duration}s")
        button.on_clicked(
            lambda _event, value=selected_duration: set_raw_duration(value)
        )
        controls.append(button)
    for index, selected_duration in enumerate(DISPLAY_DURATIONS):
        button_axis = figure.add_axes([0.64 + index * 0.09, 0.08, 0.07, 0.05])
        button = Button(button_axis, f"{selected_duration}s")
        button.on_clicked(
            lambda _event, value=selected_duration: set_processed_duration(value)
        )
        controls.append(button)

    filter_axis = figure.add_axes([0.78, 0.02, 0.20, 0.22])
    filter_buttons = RadioButtons(filter_axis, list(FILTER_MODES), active=0)
    filter_buttons.on_clicked(set_filter)
    controls.append(filter_buttons)

    group_axis = figure.add_axes([0.50, 0.02, 0.25, 0.22])
    group_labels = ["Tất cả"] + [f"Nhóm {index}" for index in range(1, 9)]
    group_buttons = RadioButtons(group_axis, group_labels, active=0)
    group_buttons.on_clicked(set_group)
    controls.append(group_buttons)

    def set_raw_duration(value: int) -> None:
        nonlocal raw_duration
        raw_duration = value
        redraw()

    def set_processed_duration(value: int) -> None:
        nonlocal processed_duration
        processed_duration = value
        redraw()

    figure.text(0.28, 0.15, "Raw / filter", ha="left")
    figure.text(0.64, 0.15, "Đã tiền xử lý", ha="left")
    figure._csv_preview_controls = controls
    figure.canvas.mpl_connect("scroll_event", on_scroll)
    redraw()
    plt.show()


def show_dataset_preview(dataset_file: str, sample_index: int = 0) -> None:
    with np.load(dataset_file, allow_pickle=False) as data:
        required = {"X", "y"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Dataset thiếu trường bắt buộc: {sorted(missing)}")
        raw_shape = data["X"].shape
        samples = np.asarray(data["X"]).copy()
        labels = np.asarray(data["y"]).copy()
        label_names_json = (
            str(data["label_names_json"]) if "label_names_json" in data else None
        )
    if samples.ndim == 4 and samples.shape[-1] == 1:
        samples = samples[..., 0]
    if samples.ndim != 3:
        raise ValueError(
            f"X phải có dạng (N, frames, components, 1), nhận được {raw_shape}."
        )
    if len(samples) == 0 or len(samples) != len(labels):
        raise ValueError("Dataset không có sample hoặc X/y không cùng số lượng.")
    if not 0 <= sample_index < len(samples):
        raise ValueError(
            f"sample phải trong khoảng 0..{len(samples) - 1}, nhận {sample_index}."
        )

    label_names = {}
    if label_names_json is not None:
        import json
        label_names = {
            int(key): value
            for key, value in json.loads(label_names_json).items()
        }
    print(f"Dataset: {dataset_file}")
    print(f"X shape: {raw_shape} -> sample sau preprocessing: {samples.shape[1:]}")
    print(f"y shape: {labels.shape} | samples: {len(samples)}")
    print(f"X range: [{samples.min():.4f}, {samples.max():.4f}]")

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    figure.subplots_adjust(bottom=0.18, wspace=0.3)

    def redraw(index: int) -> None:
        sample = samples[index]
        label = int(labels[index])
        axes[0].clear()
        image = axes[0].imshow(
            sample.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        axes[0].set_title(f"Sample {index}: heatmap sau preprocessing")
        axes[0].set_xlabel("Frame trong cửa sổ (60)")
        axes[0].set_ylabel("PCA component (10)")

        axes[1].clear()
        axes[1].plot(sample, linewidth=0.8)
        axes[1].set_title(
            f"Label {label}: {label_names.get(label, 'Unknown')} | "
            f"min={sample.min():.3f}, max={sample.max():.3f}"
        )
        axes[1].set_xlabel("Frame trong cửa sổ")
        axes[1].set_ylabel("Giá trị chuẩn hóa")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.25)
        figure.suptitle(
            f"Dataset sample {index + 1}/{len(samples)} | "
            f"shape {sample.shape} | PCA components={sample.shape[1]}"
        )
        figure.canvas.draw_idle()

    previous_axis = figure.add_axes([0.38, 0.04, 0.1, 0.06])
    next_axis = figure.add_axes([0.52, 0.04, 0.1, 0.06])
    previous_button = Button(previous_axis, "Sample trước")
    next_button = Button(next_axis, "Sample sau")
    current_index = [sample_index]

    def change_sample(step: int) -> None:
        current_index[0] = (current_index[0] + step) % len(samples)
        redraw(current_index[0])

    previous_button.on_clicked(lambda _event: change_sample(-1))
    next_button.on_clicked(lambda _event: change_sample(1))
    redraw(sample_index)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize CSI realtime từ serial.")
    parser.add_argument("--port", default="COM8", help="Cổng serial, ví dụ COM8.")
    parser.add_argument("--baudrate", "--baud", type=int, default=2000000)
    parser.add_argument(
        "--data-dir",
        help="Tùy chọn: giữ chế độ xem CSV cũ từ thư mục Data/<label>.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_file",
        nargs="+",
        help="Hiển thị tối đa 3 file CSV cùng lúc.",
    )
    parser.add_argument(
        "--dataset",
        help="Hiển thị dataset .npz sau khi chạy dataset_pipeline.py.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Chỉ số sample bắt đầu khi xem --dataset (mặc định: 0).",
    )
    parser.add_argument(
        "--model",
        help="Checkpoint .pth để bật prediction timeline.",
    )
    parser.add_argument(
        "--pca-source",
        help="Checkpoint .pth chứa PCA fit trên training set cho CSV preview.",
    )
    args = parser.parse_args()
    if args.dataset:
        show_dataset_preview(args.dataset, args.sample)
    elif args.csv_file:
        show_csv_preview(csv_files=args.csv_file, pca_source=args.pca_source)
    elif args.data_dir:
        show_csv_preview(data_dir=args.data_dir, pca_source=args.pca_source)
    else:
        RealtimeVisualizer(args.port, args.baudrate, args.model).show()


if __name__ == "__main__":
    main()
