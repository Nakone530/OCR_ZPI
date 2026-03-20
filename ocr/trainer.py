"""
Moduł trenowania modelu OCR.

Odpowiedzialności:
  - pobieranie i rozpakowywanie datasetu Chars74K
  - trening sieci SimpleCNN z walidacją
  - zapis najlepszego modelu na dysk
"""

import os
import sys
import tarfile
import urllib.request
import signal

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
        print(f"brak datasetu")

        print("\nPobrano!")

    if not os.path.exists(EXTRACTED_DIR):
        print("Rozpakowywanie...")
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)
        print("Rozpakowano!")


# ── Trening ────────────────────────────────────────────────────────────────────

def handler(signum, frame):
    print(f"Odebrano sygnał: {signum}")
    print("Program działa.")

    sys.exit(0)

# Rejestracja obsługi sygnału SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, handler)




def train_model(epochs: int = 10, batch_size: int = 32):
    """Trenuje model na datasecie Chars74K."""
    print("\n" + "=" * 60)
    print("TRENOWANIE MODELU")
    print("=" * 60)

    # Pobierz dataset jeśli brak
    download_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    # Transformacje
    train_transform = get_train_transform()

    # Dataset
    dataset = datasets.ImageFolder(root=EXTRACTED_DIR, transform=train_transform)

    # Podział na train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Zbiór treningowy: {len(train_dataset)} obrazów")
    print(f"Zbiór walidacyjny: {len(val_dataset)} obrazów")
    print(f"Liczba klas: {len(dataset.classes)}")

    # Model
    model = SimpleCNN(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Trenowanie
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Walidacja
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Zapisano najlepszy model (acc: {best_acc:.2f}%)")

    print(f"\nTrenowanie zakończone! Najlepsza dokładność: {best_acc:.2f}%")
    print(f"Model zapisany do: {MODEL_PATH}")

