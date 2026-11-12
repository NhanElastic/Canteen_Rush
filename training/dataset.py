import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from training.config import NUM_SAMPLES, VALID_EXTS
from training.utils import fit_to_window, load_and_trim_audio

class CanteenAudioDataset(Dataset):
    def __init__(self, split_dir: Path, class_names: list[str], is_train: bool = True):
        self.class_name = class_names
        self.is_train = is_train
        self.file_paths = []
        self.samples: list[tuple[np.ndarray, int, bool]] = []
        self.noise_pool: list[np.ndarray] = []

        for cls_name in class_names:
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                continue
            is_unknown = cls_name == "unknown"

            for audio_file in sorted(cls_dir.iterdir()):
                if not audio_file.is_file() or audio_file.suffix.lower() not in VALID_EXTS:
                    continue
                wav = load_and_trim_audio(audio_file, is_unknown=is_unknown)
                if wav is None:
                    continue
                if is_unknown:
                    self.noise_pool.append(wav)
                self.samples.append((wav, self.class_to_idx[cls_name], is_unknown))

    def __len__(self):
        return len(self.samples)

    def _augment(self, chunk: np.ndarray, is_unknown: bool) -> np.ndarray:
        chunk = chunk * random.uniform(0.6, 1.4)

        if not is_unknown and self.noise_pool and random.random() < 0.5:
            noise = random.choice(self.noise_pool)
            noise_chunk = fit_to_window(noise, is_unknown=True, is_train=True)
            chunk = chunk + noise_chunk * random.uniform(0.1, 0.35)

        if random.random() < 0.6:
            noise_std = random.uniform(0.001, 0.008)
            chunk = chunk + np.random.normal(0.0, noise_std, size=NUM_SAMPLES).astype(np.float32)

        return np.clip(chunk, -1.0, 1.0)

    def __getitem__(self, index: int):
        wav, label, is_unknown = self.samples[index]
        chunk = fit_to_window(wav, is_unknown=is_unknown, is_train=self.is_train)

        if self.is_train:
            chunk = self._argument(chunk, is_unknown=is_unknown)

        return torch.from_numpy(chunk), torch.tensor(label, dtype=torch.long)