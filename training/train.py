import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.config import (
    BATCH_SIZE,
    DATA_ROOT,
    EPOCHS,
    LEARNING_RATE,
    OUTPUTS_DIR,
    SEED,
)
from training.dataset import CanteenAudioDataset
from training.models.ds_cnn import DSCNN
from training.utils import export_onnx, set_seed, split_audio_dataset


def train():
    set_seed(SEED)
    train_dir = DATA_ROOT / "train"
    val_dir = DATA_ROOT / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Chưa có dữ liệu tại '{train_dir}'. Hãy chạy split_audio_dataset trước.")

    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print(f"Phát hiện {len(class_names)} lớp: {class_names}")

    train_ds = CanteenAudioDataset(train_dir, class_names, is_train=True)
    val_ds = CanteenAudioDataset(val_dir, class_names, is_train=False)
    print(f"Số mẫu hợp lệ -> Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị huấn luyện: {device}")

    model = DSCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_model_path = OUTPUTS_DIR / "best_canteen.pth"
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_correct, train_total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += x.size(0)

        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total += x.size(0)

        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)
        history.append({"epoch": epoch, "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4)})

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Acc: {train_acc*100:5.1f}% | Val Acc: {val_acc*100:5.1f}%")

    with open(OUTPUTS_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nHoàn tất! Best Val Acc: {best_val_acc*100:.1f}%")
    model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
    export_onnx(model, class_names)


if __name__ == "__main__":
    split_audio_dataset()
    train()