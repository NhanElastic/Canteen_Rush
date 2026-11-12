import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import random

class LogMelExtractor(nn.Module):
    def __init__(self, sr=16000, n_fft=512, win_length=400, hop_length=160, n_mels=64):
        super().__init__()
        self.hop_length = hop_length
        self.pad = win_length // 2

        win = torch.hann_window(win_length)
        t = torch.arange(win_length, dtype=torch.float32)
        freqs = torch.arange(n_fft // 2 + 1, dtype=torch.float32).unsqueeze(1)
        angles = 2.0 * math.pi * freqs * t.unsqueeze(0) / n_fft

        self.register_buffer("real_kernel", (torch.cos(angles) * win.unsqueeze(0)).unsqueeze(1))
        self.register_buffer("imag_kernel", (-torch.sin(angles) * win.unsqueeze(0)).unsqueeze(1))

        mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=80.0, fmax=7600.0)
        self.register_buffer("mel_fb", torch.from_numpy(mel_fb).float())
        self.inst_norm = nn.InstanceNorm2d(1, affine=True)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = waveform.unsqueeze(1)
        real = F.conv1d(x, self.real_kernel, stride=self.hop_length, padding=self.pad)
        imag = F.conv1d(x, self.imag_kernel, stride=self.hop_length, padding=self.pad)
        power_spec = real.pow(2) + imag.pow(2)

        mel_spec = torch.matmul(self.mel_fb, power_spec)
        log_mel = torch.log(mel_spec + 1e-6).unsqueeze(1)  # [B, 1, 64 (Freq), 121 (Time)]
        return self.inst_norm(log_mel)

class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=8, time_mask_param=15):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return spec
        _, _, n_mels, n_steps = spec.shape
        out = spec.clone()

        f = random.randint(0, self.freq_mask_param)
        f0 = random.randint(0, max(0, n_mels - f))
        out[:, :, f0 : f0 + f, :] = 0.0

        t = random.randint(0, self.time_mask_param)
        t0 = random.randint(0, max(0, n_steps - t))
        out[:, :, :, t0 : t0 + t] = 0.0
        return out