import random
import shutil
from pathlib import Path
import numpy as np
import librosa
import math
import torch
import torch.nn as nn
import json

from config import DATA_ROOT, NUM_SAMPLES, RAW_COLLECTED_DIR, SAMPLE_RATE, SEED, VALID_EXTS, EXPORT_DIR


def split_audio_dataset(
    source_dir: Path = RAW_COLLECTED_DIR,
    output_dir: Path = DATA_ROOT,
    train_ratio: float = 0.8,
    seed: int = SEED,
):
    if not source_dir.exists():
        print(f"Không tìm thấy '{source_dir}'. Hãy đặt các thư mục lớp vào đây trước.")
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)

    random.seed(seed)
    class_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])

    if not class_dirs:
        print(f"Không tìm thấy thư mục lớp trong '{source_dir}'.")
        return

    total_train, total_val = 0, 0
    print(f"{'Lớp (Class)':<18} | {'Tổng':>5} | {'Train (80%)':>11} | {'Val (20%)':>9}")
    print("-" * 54)

    for class_dir in class_dirs:
        class_name = class_dir.name
        files = sorted(
            [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTS]
        )

        if not files:
            print(f"{class_name:<18} | {'0':>5} | {'0':>11} | {'0':>9}  (Trống)")
            continue

        random.shuffle(files)
        split_index = int(len(files) * train_ratio)
        if split_index == len(files) and len(files) > 1:
            split_index -= 1

        train_files = files[:split_index]
        val_files = files[split_index:]

        for subset_name, subset_files in [("train", train_files), ("val", val_files)]:
            subset_dir = output_dir / subset_name / class_name
            subset_dir.mkdir(parents=True, exist_ok=True)
            for i, src_file in enumerate(subset_files, start=1):
                safe_name = f"{class_name}_{i:03d}{src_file.suffix.lower()}"
                shutil.copy2(src_file, subset_dir / safe_name)

        total_train += len(train_files)
        total_val += len(val_files)
        print(f"{class_name:<18} | {len(files):>5} | {len(train_files):>11} | {len(val_files):>9}")

    print("-" * 54)
    print(f"{'Tổng cộng (Total)':<18} | {total_train + total_val:>5} | {total_train:>11} | {total_val:>9}")

def load_and_trim_audio(file_path: Path, is_unknown: bool = False) -> np.ndarray | None:
    wav, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    if len(wav) < 1600:
        return None

    if not is_unknown:
        trimmed, _ = librosa.effects.trim(wav, top_db=25)
        if len(trimmed) >= 1600:
            wav = trimmed

    return wav.astype(np.float32)

def fit_to_window(wav: np.ndarray, is_unknown: bool = False, is_train: bool = False) -> np.ndarray:
    if is_unknown:
        if len(wav) >= NUM_SAMPLES:
            start = random.randint(0, len(wav) - NUM_SAMPLES) if is_train else 0
            return wav[start : start + NUM_SAMPLES].copy()
        tiled = np.tile(wav, math.ceil(NUM_SAMPLES / len(wav)))
        return tiled[:NUM_SAMPLES].copy()

    if len(wav) >= NUM_SAMPLES:
        start = random.randint(0, len(wav) - NUM_SAMPLES) if is_train else (len(wav) - NUM_SAMPLES) // 2
        return wav[start : start + NUM_SAMPLES].copy()

    out = np.zeros(NUM_SAMPLES, dtype=np.float32)
    max_offset = NUM_SAMPLES - len(wav)
    offset = random.randint(0, max_offset) if is_train else (max_offset // 2)
    out[offset : offset + len(wav)] = wav
    return out

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def export_onnx(model: nn.Module, class_names: list[str]):
    model.to("cpu").eval()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = EXPORT_DIR / "canteen_rush.onnx"
    labels_path = EXPORT_DIR / "labels.json"

    dummy_input = torch.randn(1, NUM_SAMPLES, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["waveform"],
        output_names=["logits"],
        dynamic_axes={"waveform": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Đã xuất ONNX ({size_mb:.2f} MB) -> {onnx_path}")
    print(f"Đã xuất nhãn -> {labels_path}")