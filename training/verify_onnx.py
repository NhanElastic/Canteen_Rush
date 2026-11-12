import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from training.config import DATA_ROOT, EXPORT_DIR
from training.utils import fit_to_window, load_and_trim_audio

MODEL_PATH = EXPORT_DIR / "canteen_rush.onnx"
LABELS_PATH = EXPORT_DIR / "labels.json"
VAL_DIR = DATA_ROOT / "val"



def predict_file(session: ort.InferenceSession, audio_path: Path, is_unknown: bool = False) -> np.ndarray:
    wav = load_and_trim_audio(audio_path, is_unknown=is_unknown)
    chunk = fit_to_window(wav, is_unknown=is_unknown, is_train=False)
    logits = session.run(["logits"], {"waveform": chunk[np.newaxis, :]})[0][0]
    exp_x = np.exp(logits - np.max(logits))
    return exp_x / np.sum(exp_x)


def eval_validation_set():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    total, correct = 0, 0

    print(f"{'Lớp thực tế':<18} | {'Đúng / Tổng':>11} | {'Độ chính xác':>12}")
    print("-" * 48)

    for cls_idx, cls_name in enumerate(labels):
        cls_dir = VAL_DIR / cls_name
        if not cls_dir.exists():
            continue
        files = sorted([f for f in cls_dir.iterdir() if f.is_file()])
        cls_correct = 0

        for f_path in files:
            probs = predict_file(session, f_path, is_unknown=(cls_name == "nen_tieng_on"))
            if int(np.argmax(probs)) == cls_idx:
                cls_correct += 1

        total += len(files)
        correct += cls_correct
        acc = (cls_correct / max(1, len(files))) * 100
        print(f"{cls_name:<18} | {cls_correct:>4} / {len(files):<4} | {acc:>10.1f}%")

    print("-" * 48)
    print(f"{'TỔNG QUÁT (ONNX)':<18} | {correct:>4} / {total:<4} | {(correct / max(1, total))*100:>10.1f}%")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
        session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        probs = predict_file(session, Path(sys.argv[1]))
        print(f"Kết quả file: {sys.argv[1]}")
        for idx in np.argsort(probs)[::-1]:
            print(f"  - {labels[idx]:<15}: {probs[idx]*100:.2f}%")
    else:
        eval_validation_set()