import torch
import torch.nn as nn
from training.models.audio_frontend import LogMelExtractor, SpecAugment

class DSConvBlock(nn.Module):
    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class DSCNN(nn.Module):
    def __init__(self, num_classes: int, channels: int = 128):
        super().__init__()
        self.frontend = LogMelExtractor()
        self.spec_aug = SpecAugment()

        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(4, 10), stride=(2, 2), padding=(1, 4), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # 4 khối DS-CNN xếp tầng
        self.blocks = nn.Sequential(
            DSConvBlock(channels, stride=1),
            DSConvBlock(channels, stride=2),
            DSConvBlock(channels, stride=1),
            DSConvBlock(channels, stride=1),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(channels, num_classes),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.frontend(waveform)
        spec = self.spec_aug(spec)
        feat = self.stem(spec)
        feat = self.blocks(feat)
        return self.classifier(feat)