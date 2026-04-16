"""
Klasa do zarządzania zestawem danych
"""

import os

import torch
from torch.utils.data import Dataset

from PIL import Image


class OCRDataset(torch.utils.data.Dataset):
    def __init__(self, json_data, images_dir, transform=None):
        self.data = json_data
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_path = item["image_path"]
        image = Image.open(img_path).convert("L")

        text = item["text"]

        if self.transform:
            image = self.transform(image)

        return image, text
