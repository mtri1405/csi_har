import numpy as np
import torch
from train import CSI_CNN
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. CHECK DATASET ---
data = np.load("csi_dataset.npz")
X, y = data["X"], data["y"]
test_idx = data["test_indices"]
print("=== DATASET ===")
print("X.shape =", X.shape, "(N,", X.shape[1], "frames,", X.shape[2], "subcarriers, 1 channel)")
print("y.shape =", y.shape)
uniq, cnt = np.unique(y, return_counts=True)
for u, c in zip(uniq, cnt):
    print("  label", u, ":", c, "samples")

# --- 2. CHECK MODEL ---
ckpt = torch.load("csi_cnn_model.pth", map_location=device)
print("\n=== MODEL ===")
model = CSI_CNN(num_classes=len(uniq)).to(device)
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
model.eval()
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total params:", f"{total_params:,}")
print("Trainable params:", f"{trainable_params:,}")
print("Layers:", [n for n, _ in model.named_modules()])

# --- 3. EVALUATE RESULTS (saved stratified window test split) ---
Xt = np.transpose(X, (0, 3, 1, 2))
X_test, y_test = Xt[test_idx], y[test_idx]
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

with torch.no_grad():
    logits = model(X_test_t)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)

acc = accuracy_score(y_test, preds)
print("\n=== TEST RESULTS (INDEPENDENT RECORDING SESSIONS) ===")
print("Test Accuracy:", f"{acc*100:.2f}%")
print("Confusion matrix:\n", confusion_matrix(y_test, preds))
print("Classification report:\n", classification_report(y_test, preds, digits=4))
