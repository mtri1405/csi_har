# train.py
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from csi_model import CSI_CNN
from csi_preprocessing import (
    FILTER_CUTOFF_HZ,
    FILTER_ORDER,
    PCA_COMPONENTS,
    RAW_SUBCARRIERS,
    REMOVED_SUBCARRIER_INDICES,
    SAMPLE_RATE_HZ,
    TARGET_SUBCARRIERS,
    WINDOW_SIZE,
)

SEED = 42
EPOCHS = 40
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 10
DATASET_FILE = "csi_dataset.npz"
MODEL_FILE = "csi_cnn_model.pth"

DEFAULT_LABEL_NAMES = {
    0: "Phòng trống (Empty)",
    1: "Đang đi lại (Walking)",
    2: "Đang đứng yên (Standing)",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(X, y, shuffle, pin_memory):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, pin_memory=pin_memory)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n_samples = 0.0, 0
    preds, labels_all = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
            labels_all.extend(labels.cpu().numpy().tolist())
    return total_loss / max(n_samples, 1), np.array(preds), np.array(labels_all)


def print_distribution(name, labels, class_ids):
    counts = {int(c): int((labels == c).sum()) for c in class_ids}
    print(f"    {name:<5}: {len(labels):>6} mẫu | theo nhãn: {counts}")


def main():
    set_seed(SEED)

    # --- THIẾT BỊ ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[*] Thiết bị huấn luyện: GPU ({torch.cuda.get_device_name(0)})")
        use_pin_memory = True
    else:
        device = torch.device("cpu")
        print("[*] Không có GPU CUDA, huấn luyện bằng CPU")
        use_pin_memory = False

    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Không tìm thấy '{DATASET_FILE}'! Hãy chạy dataset_pipeline.py trước.")

    # --- NẠP DỮ LIỆU ---
    print(f"[*] Đang tải dữ liệu từ {DATASET_FILE}...")
    data = np.load(DATASET_FILE)
    if (
        "groups" not in data
        or "subcarrier_indices" not in data
        or "pca_mean" not in data
        or "pca_components" not in data
        or "train_indices" not in data
        or "val_indices" not in data
        or "test_indices" not in data
    ):
        raise ValueError("Dataset thiếu metadata. Hãy chạy lại dataset_pipeline.py.")

    X = data["X"]  # (N, WINDOW_SIZE, W, 1)
    y = data["y"]
    groups = data["groups"]
    subcarrier_indices = data["subcarrier_indices"].tolist()
    pca_mean = data["pca_mean"]
    pca_components = data["pca_components"]
    if pca_mean.shape != (TARGET_SUBCARRIERS,) or pca_components.shape != (
        PCA_COMPONENTS,
        TARGET_SUBCARRIERS,
    ):
        raise ValueError("PCA state trong dataset không khớp cấu hình hiện tại.")
    print(f"    Tổng số mẫu: {X.shape[0]} | Kích thước mẫu: {X.shape[1:]}")

    X = np.transpose(X, (0, 3, 1, 2))  # -> (N, 1, WINDOW_SIZE, W)

    # Nhãn phải liên tục 0..K-1, nếu không CrossEntropyLoss sẽ lỗi hoặc gán sai lớp
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    if unique_classes.tolist() != list(range(num_classes)):
        raise ValueError(
            f"Nhãn phải là 0..{num_classes - 1} liên tục, hiện có {unique_classes.tolist()}. "
            "Có thể thiếu hẳn một thư mục hành động khi tạo dataset."
        )
    print(f"[*] Phát hiện {num_classes} nhãn: {unique_classes.tolist()}")

    # Tên nhãn lấy từ dataset, không hard-code lại ở đây
    if "label_names_json" in data:
        raw_names = json.loads(str(data["label_names_json"]))
        label_names = {int(k): v for k, v in raw_names.items()}
    else:
        label_names = DEFAULT_LABEL_NAMES
    label_names = {c: label_names.get(c, f"Class {c}") for c in range(num_classes)}

    # Dataset builder already created one deterministic, stratified window split.
    train_idx = data["train_indices"]
    val_idx = data["val_indices"]
    test_idx = data["test_indices"]
    all_indices = np.concatenate((train_idx, val_idx, test_idx))
    if not np.array_equal(np.sort(all_indices), np.arange(len(y))):
        raise ValueError("Các split window trong dataset không bao phủ đúng toàn bộ dữ liệu.")
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("[*] Phân bố mẫu sau khi chia:")
    print_distribution("train", y_train, unique_classes)
    print_distribution("val", y_val, unique_classes)
    print_distribution("test", y_test, unique_classes)
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        assert len(np.unique(y[idx])) == num_classes, f"Tập {name} thiếu nhãn!"
    train_loader = make_loader(X_train, y_train, True, use_pin_memory)
    val_loader = make_loader(X_val, y_val, False, use_pin_memory)
    test_loader = make_loader(X_test, y_test, False, use_pin_memory)

    # --- MÔ HÌNH, LOSS (CÓ TRỌNG SỐ LỚP), OPTIMIZER ---
    model = CSI_CNN(num_classes=num_classes).to(device)

    counts = np.bincount(y_train, minlength=num_classes)
    class_weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    print(f"[*] Trọng số lớp: {np.round(class_weights, 3).tolist()}")
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )
    # Loss để đánh giá val/test không dùng trọng số, cho dễ so sánh
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )

    # --- VÒNG LẶP HUẤN LUYỆN ---
    best_f1 = -1.0
    epochs_no_improve = 0
    print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        val_loss, val_pred, val_true = evaluate(model, val_loader, eval_criterion, device)
        val_acc = 100.0 * (val_pred == val_true).mean()
        val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        val_recalls = recall_score(
            val_true, val_pred, labels=list(range(num_classes)), average=None, zero_division=0
        )

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] | LR: {old_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f} | "
            f"Recall/lớp: {np.round(val_recalls, 2).tolist()}"
        )
        if new_lr < old_lr:
            print(f"    --> [Scheduler] LR giảm: {old_lr:.6f} -> {new_lr:.6f}")

        # Chọn checkpoint theo macro-F1: một lớp bị bỏ rơi sẽ kéo F1 xuống rõ rệt,
        # trong khi accuracy tổng có thể vẫn cao.
        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": {
                        "window_size": WINDOW_SIZE,
                        "sample_rate_hz": SAMPLE_RATE_HZ,
                        "filter_cutoff_hz": FILTER_CUTOFF_HZ,
                        "filter_order": FILTER_ORDER,
                        "filter_type": "butterworth_lowpass",
                        "raw_subcarriers": RAW_SUBCARRIERS,
                        "target_subcarriers": TARGET_SUBCARRIERS,
                        "removed_subcarrier_indices": list(REMOVED_SUBCARRIER_INDICES),
                        "pca_components": PCA_COMPONENTS,
                        "normalization": "min_max_0_1",
                        "subcarrier_indices": subcarrier_indices,
                        "pca_mean": pca_mean.tolist(),
                        "pca_components_matrix": pca_components.tolist(),
                        "labels": {str(k): v for k, v in label_names.items()},
                        "num_classes": num_classes,
                    },
                },
                MODEL_FILE,
            )
            print(f"    --> [Checkpoint] Lưu mô hình tốt nhất (Val F1 = {best_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"    --> [Early stop] Không cải thiện sau {EARLY_STOP_PATIENCE} epoch.")
                break

    print(f"\n[HOÀN TẤT] Val macro-F1 tốt nhất: {best_f1:.4f}")
    print(f"Trọng số tốt nhất đã lưu vào: '{MODEL_FILE}'")

    # --- ĐÁNH GIÁ CUỐI TRÊN TẬP TEST ĐỘC LẬP ---
    best_checkpoint = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    _, test_pred, test_true = evaluate(model, test_loader, eval_criterion, device)

    report_labels = list(range(num_classes))
    report_names = [label_names[c] for c in report_labels]
    cm = confusion_matrix(test_true, test_pred, labels=report_labels)

    print("\n--- CONFUSION MATRIX (hàng = nhãn thật, cột = dự đoán) ---")
    print("Thứ tự:", report_names)
    print(cm)
    print("\n--- CLASSIFICATION REPORT (TEST) ---")
    print(classification_report(
        test_true, test_pred,
        labels=report_labels, target_names=report_names,
        digits=4, zero_division=0,
    ))

    evaluation = {
        "labels": report_names,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            test_true, test_pred,
            labels=report_labels, target_names=report_names,
            output_dict=True, zero_division=0,
        ),
    }
    print(f"EVALUATION_JSON: {json.dumps(evaluation, ensure_ascii=False)}")


if __name__ == "__main__":
    main()