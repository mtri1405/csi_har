# train.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# --- 1. ĐỊNH NGHĨA KIẾN TRÚC MẠNG CNN ---
class CSI_CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CSI_CNN, self).__init__()
        self.features = nn.Sequential(
            # Conv block 1: Input (Batch, 1, 30, 192) -> Output (Batch, 16, 15, 96)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Conv block 2: Output (Batch, 32, 7, 48)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Conv block 3: Feature map cố định (Batch, 64, 2, 2)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def main():
    # --- THIẾT LẬP THIẾT BỊ: ƯU TIÊN GPU, KHÔNG CÓ THÌ DÙNG CPU ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[*] Thiết bị huấn luyện: GPU ({torch.cuda.get_device_name(0)})")
        use_pin_memory = True
    else:
        device = torch.device("cpu")
        print("[*] Không tìm thấy GPU CUDA, chuyển sang huấn luyện bằng: CPU")
        use_pin_memory = False

    dataset_file = "csi_dataset.npz"
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Không tìm thấy '{dataset_file}'! Hãy chạy dataset_pipeline.py trước.")

    # --- 2. NẠP VÀ TIỀN XỬ LÝ SHAPE TENSOR ---
    print(f"[*] Đang tải dữ liệu từ {dataset_file}...")
    data = np.load(dataset_file)
    X = data["X"]  # Kích thước: (N, 30, số_subcarrier_hợp_lệ, 1)
    y = data["y"]  # Kích thước: (N,)

    print(f"    Tổng số mẫu: {X.shape[0]} | Kích thước mẫu: {X.shape[1:]}")

    # Đổi chiều từ (N, H, W, C) sang chuẩn PyTorch (N, C, H, W)
    X = np.transpose(X, (0, 3, 1, 2))

    # Tự động trích xuất số lớp từ nhãn thực tế có trong dữ liệu
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    print(f"[*] Phát hiện {num_classes} nhãn: {unique_classes}")

    # Chia tập Train (80%) và Test/Validation (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Chuyển đổi sang PyTorch Tensor
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=use_pin_memory)

    # --- 3. KHỞI TẠO MÔ HÌNH, HÀM MẤT MÁT VÀ BỘ TỐI ƯU ---
    model = CSI_CNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --- BỘ ĐIỀU CHỈNH TỐC ĐỘ HỌC (LEARNING RATE SCHEDULER) ---
    # mode='min': theo dõi Loss giảm
    # factor=0.5: giảm 50% LR khi bị chững
    # patience=2: đợi 2 epoch liên tiếp không cải thiện thì mới hạ LR
    # min_lr=1e-5: giới hạn LR nhỏ nhất
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-5
    )

    # --- 4. VÒNG LẶP HUẤN LUYỆN ---
    EPOCHS = 25
    best_acc = -1.0
    print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Đánh giá trên tập Test
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100.0 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        # Lấy tốc độ học hiện tại
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}%")

        # Cập nhật Scheduler theo Loss
        old_lr = current_lr
        scheduler.step(avg_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"    --> [Scheduler] Tốc độ học giảm: {old_lr:.6f} -> {new_lr:.6f}")

        # Lưu checkpoint có độ chính xác cao nhất
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "csi_cnn_model.pth")
            print(f"    --> [Checkpoint] Đã lưu mô hình mới tốt nhất ({best_acc:.2f}%)")

    print(f"\n[HOÀN TẤT] Độ chính xác cao nhất đạt được: {best_acc:.2f}%")
    print("Trọng số mô hình tối ưu đã được lưu vào file: 'csi_cnn_model.pth'")

if __name__ == "__main__":
    main()