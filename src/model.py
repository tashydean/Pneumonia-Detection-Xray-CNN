"""
model.py
PneumoniaResNet: ResNet18 + pre_conv block.

pre_conv 설계 의도:
  자연 이미지에 최적화된 ResNet 앞단에,
  흉부 X-ray 특유의 저대비·음영 패턴을 먼저 정제하는 블록 추가.
  채널: 3 → 16 → 3 (ResNet 입력 shape 보존)
"""

import torch.nn as nn
from torchvision import models


class PneumoniaResNet(nn.Module):
    def __init__(self, pretrained=True, dropout_rate=0.5):
        super(PneumoniaResNet, self).__init__()

        # X-ray 특화 초기 특징 추출
        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

        # ResNet18 backbone
        self.model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        num_ftrs = self.model.fc.in_features  # 512

        # FC Head: dropout_rate 인자로 실험마다 조정 가능
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.model(x)
        return x.squeeze(1)
