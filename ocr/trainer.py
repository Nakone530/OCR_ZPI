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
    DATA_URL, DATA_DIR, ARCHIVE_PATH, EXTRACTED_DIR, MODEL_PATH, CHECKPOINT_PATH
)
from .model import SimpleCNN
from .utils import get_train_transform


# -- globalne
GLOBAL_MODEL = None
GLOBAL_OPTIMIZER = None
GLOBAL_CRITERION = None
GLOBAL_DEVICE = None

GLOBAL_EPOCH = 0
GLOBAL_BEST_ACC = 0.0

# -- Dataset

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


# -- Trening

checkpoint_path = "checkpoint.pth"
current_state = {}

def handler(signum, frame):
    print(f"Odebrano sygnał: {signum}")
    checkpoint_path = save_model(CHECKPOINT_PATH)
    print("Program działa.")
    paused = True
    while paused:
        print("\n--- MENU ---")
        print("1. Wznów")
        print("2. Zapisz najlepszy")
        print("3. Wyjście bez zapisu")

        choice = input("Wybierz opcję: ")

        if choice == "1":
            print("Wznawianie")
            train_model(10, 32, CHECKPOINT_PATH)
        elif choice == "2":
            print("Zapisywanie najlepszego (obecnie nie do końca działa, zapisuje ostatni stan)")
            save_model(MODEL_PATH)
        elif choice == "3":
            print("Zamykanie programu...")
            paused = False

        else:
            print("Nieprawidłowy wybór!")
    sys.exit(0)

# Rejestracja obsługi sygnału SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, handler)

def save_model(path):
    global GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_EPOCH, GLOBAL_BEST_ACC

    if GLOBAL_MODEL is None:
        raise RuntimeError("Model nie jest zainicjalizowany")

    torch.save({
        "epoch": GLOBAL_EPOCH,
        "model_state_dict": GLOBAL_MODEL.state_dict(),
        "optimizer_state_dict": GLOBAL_OPTIMIZER.state_dict(),
        "best_acc": GLOBAL_BEST_ACC
    }, path)

    print(f"Model zapisany do: {path}")

    return path


def init_or_load_model(num_classes, model_path=None):
    global GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_CRITERION
    global GLOBAL_DEVICE, GLOBAL_EPOCH, GLOBAL_BEST_ACC

    GLOBAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    GLOBAL_MODEL = SimpleCNN(num_classes=num_classes).to(GLOBAL_DEVICE)
    GLOBAL_CRITERION = nn.CrossEntropyLoss()
    GLOBAL_OPTIMIZER = torch.optim.Adam(GLOBAL_MODEL.parameters(), lr=0.001)

    if model_path and os.path.exists(model_path):
        print(f"Wczytywanie modelu z: {model_path}")
        checkpoint = torch.load(model_path, map_location=GLOBAL_DEVICE)

        if "model_state_dict" in checkpoint:
            GLOBAL_MODEL.load_state_dict(checkpoint["model_state_dict"])
            GLOBAL_OPTIMIZER.load_state_dict(checkpoint["optimizer_state_dict"])
            GLOBAL_EPOCH = checkpoint.get("epoch", 0)
            GLOBAL_BEST_ACC = checkpoint.get("best_acc", 0.0)
        else:
            GLOBAL_MODEL.load_state_dict(checkpoint)


def train_model(epochs=10, batch_size=32, model_path=None):
    global GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_CRITERION
    global GLOBAL_DEVICE, GLOBAL_EPOCH, GLOBAL_BEST_ACC

    download_dataset()

    train_transform = get_train_transform()
    dataset = datasets.ImageFolder(root=EXTRACTED_DIR, transform=train_transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # init albo load
    if GLOBAL_MODEL is None:
        init_or_load_model(len(dataset.classes), model_path)
        
    total_steps = len(train_loader)
    for epoch in range(GLOBAL_EPOCH, GLOBAL_EPOCH + epochs):
        GLOBAL_MODEL.train()
        running_loss = 0.0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(GLOBAL_DEVICE), labels.to(GLOBAL_DEVICE)

            GLOBAL_OPTIMIZER.zero_grad()
            outputs = GLOBAL_MODEL(images)
            loss = GLOBAL_CRITERION(outputs, labels)
            loss.backward()
            GLOBAL_OPTIMIZER.step()

            running_loss += loss.item()

            # 🔹 log co 50 kroków
            if step % 50 == 0 or step == total_steps:
                print(f"Epoch [{epoch+1}] Step [{step}/{total_steps}] Loss: {loss.item():.4f}")
            
        # walidacja
        GLOBAL_MODEL.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(GLOBAL_DEVICE), labels.to(GLOBAL_DEVICE)
                outputs = GLOBAL_MODEL(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}] - Val Acc: {val_acc:.2f}%")

        GLOBAL_EPOCH = epoch + 1

        if val_acc > GLOBAL_BEST_ACC:
            GLOBAL_BEST_ACC = val_acc
            save_model(MODEL_PATH)

