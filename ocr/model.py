"""
Definicja architektury sieci CNN do rozpoznawania znaków.
"""

import torch.nn as nn

from .config import NUM_CLASSES



class SimpleCNN(nn.Module):
    """sieć CNN do rozpoznawania znaków."""

    def __init__(self, num_classes=46):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Blok 1: 1x28x28 -> 32x14x14 (grayscale)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),

            # Blok 2: 32x14x14 -> 64x7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),

            # Blok 3: 64x7x7 -> 128x3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),

            # Blok 4: 128x3x3 -> 256x3x3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
        )
        

        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=256,
            bidirectional=True,
            num_layers=2
        )
        
        self.fc = nn.Linear(512, num_classes)  # + blank

    def forward(self, x):
        x = self.features(x)          # (B, C, H, W)
        x = x.mean(2)                # (B, C, W)  ← redukcja wysokości
        x = x.permute(2, 0, 1)       # (W, B, C)

        x, _ = self.rnn(x)           # (W, B, 512)
        x = self.fc(x)               # (W, B, num_classes)
        return x
