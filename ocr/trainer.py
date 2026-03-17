"""
Moduł trenowania modelu OCR.

Odpowiedzialności:
  - pobieranie i rozpakowywanie datasetu Chars74K
  - trening sieci SimpleCNN z walidacją
  - zapis najlepszego modelu na dysk
"""

import os
import tarfile
import urllib.request

import torch
import torch.nn as nn
from torchvision import datasets

from .config import (
    DATA_URL, DATA_DIR, ARCHIVE_PATH, EXTRACTED_DIR, MODEL_PATH
)
from .model import SimpleCNN
from .utils import get_train_transform


# ── Dataset ────────────────────────────────────────────────────────────────────

def download_dataset() -> None:
    """Pobiera archiwum datasetu, jeśli jeszcze go nie ma."""
    if not os.path.exists(ARCHIVE_PATH):
        print(f"Pobieranie datasetu z {DATA_URL}...")
        os.makedirs(DATA_DIR, exist_ok=True)

        def _progress(block_num, block_size, total_size):
            percent = min(100, block_num * block_size * 100 / total_size)
            print(f"\rPostęp: {percent:.1f}%", end="")

        urllib.request.urlretrieve(DATA_URL, ARCHIVE_PATH, _progress)
        print("\nPobrano!")

    if not os.path.exists(EXTRACTED_DIR):
        print("Rozpakowywanie...")
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)
        print("Rozpakowano!")


# ── Trening ────────────────────────────────────────────────────────────────────

def train_model(epochs: int = 10, batch_size: int = 32) -> None:
    """Trenuje model na datasecie Chars74K i zapisuje najlepszy checkpoint."""
    print("\n" + "=" * 60)
    print("TRENOWANIE MODELU")
    print("=" * 60)

    download_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    # Dataset i podział 80/20
    dataset = datasets.ImageFolder(root=EXTRACTED_DIR, transform=get_train_transform())
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    print(f"Zbiór treningowy:  {len(train_dataset)} obrazów")
    print(f"Zbiór walidacyjny: {len(val_dataset)} obrazów")
    print(f"Liczba klas:       {len(dataset.classes)}")

    model = SimpleCNN(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    for epoch in range(epochs):
        # ── Faza treningowa ──
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}]  "
                    f"Step [{i+1}/{len(train_loader)}]  "
                    f"Loss: {loss.item():.4f}"
                )

        # ── Faza walidacyjna ──
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}  Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  → Zapisano najlepszy model (acc: {best_acc:.2f}%)")

    print(f"\nTrenowanie zakończone! Najlepsza dokładność: {best_acc:.2f}%")
    print(f"Model zapisany do: {MODEL_PATH}")
