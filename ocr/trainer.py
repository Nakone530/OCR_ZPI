"""
Moduł trenowania modelu OCR.

Odpowiedzialności:
  - pobieranie i rozpakowywanie datasetu Chars74K
  - trening sieci SimpleCNN z walidacją
  - zapis najlepszego modelu na dysk
  - nieskończony trening z możliwością przerwania i kontynuacji
"""

import os
import sys
import tarfile
import urllib.request
import re
import signal
import threading
import time
import gc
from datetime import datetime

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
GLOBAL_CLASS_NAMES = None

GLOBAL_EPOCH = 0
GLOBAL_BEST_ACC = 0.0

# -- flagi kontrolne dla nieskończonego treningu
TRAINING_PAUSED = False
TRAINING_STOP = False
BEST_MODEL_STATE = None  # przechowuje stan najlepszego modelu

# -- Dataset

def download_dataset() -> None:
    """Pobiera archiwum datasetu, jeśli jeszcze go nie ma."""
    if not os.path.exists(ARCHIVE_PATH):
        print(f"Brak archiwum datasetu: {ARCHIVE_PATH}")
        os.makedirs(os.path.dirname(ARCHIVE_PATH), exist_ok=True)

        def _show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                print(f"\rPobieranie: {percent:5.1f}%", end="")

        print(f"Pobieranie datasetu z: {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, ARCHIVE_PATH, _show_progress)
        print("\nPobrano!")

    if not os.path.exists(EXTRACTED_DIR):
        print("Rozpakowywanie...")
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)
        print("Rozpakowano!")


def _dataset_has_lowercase_classes(class_names: list[str]) -> bool:
    return any(len(name) == 1 and name.islower() for name in class_names)


def _dataset_looks_like_chars74k(class_names: list[str]) -> bool:
    return bool(class_names) and all(re.fullmatch(r"Sample\d+", str(name)) for name in class_names)


def _print_dataset_class_diagnostics(class_names: list[str]) -> None:
    print(f"Wykryto {len(class_names)} klas w datasecie.")

    if _dataset_has_lowercase_classes(class_names):
        print("Dataset zawiera małe litery.")
        return

    if _dataset_looks_like_chars74k(class_names):
        print("[UWAGA] Wykryto klasy w formacie Chars74K (SampleXXX).")
        print("        Upewnij się, że używasz pełnego zestawu liter (A-Z + a-z).")
        return

    print("[UWAGA] Dataset nie zawiera małych liter jako osobnych klas.")
    print("        W tym stanie model nie nauczy się rozpoznawać a-z.")
    print("        Sprawdź katalog treningowy:")
    print(f"        {EXTRACTED_DIR}")


# -- Trening

checkpoint_path = "checkpoint.pth"
current_state = {}


def _robust_torch_save(payload: dict, path: str, retries: int = 3, delay_s: float = 0.4) -> str:
    """Próbuje zapisać plik kilka razy; przy blokadzie używa pliku awaryjnego."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            torch.save(payload, path)
            return path
        except Exception as exc:
            last_exc = exc
            print(f"[WARN] Nie udało się zapisać '{path}' (próba {attempt}/{retries}): {exc}")
            if attempt < retries:
                time.sleep(delay_s)

    base, ext = os.path.splitext(path)
    fallback = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
    print(f"[WARN] Zapis awaryjny do: {fallback}")
    torch.save(payload, fallback)
    return fallback


def _state_dict_to_cpu(state_dict: dict) -> dict:
    """Kopiuje state_dict na CPU, aby nie trzymać snapshotów w VRAM."""
    cpu_state = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cpu_state[key] = value.detach().cpu().clone()
        else:
            cpu_state[key] = value
    return cpu_state


def _optimizer_state_to_cpu(optimizer_state: dict) -> dict:
    """Przenosi stany optymalizatora na CPU (rekurencyjnie)."""
    if torch.is_tensor(optimizer_state):
        return optimizer_state.detach().cpu().clone()
    if isinstance(optimizer_state, dict):
        return {k: _optimizer_state_to_cpu(v) for k, v in optimizer_state.items()}
    if isinstance(optimizer_state, list):
        return [_optimizer_state_to_cpu(v) for v in optimizer_state]
    if isinstance(optimizer_state, tuple):
        return tuple(_optimizer_state_to_cpu(v) for v in optimizer_state)
    return optimizer_state

def handler(signum, frame):
    """Handler dla zwykłego treningu (nie-nieskończonego)."""
    print(f"\nOdebrano sygnał: {signum}")
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


def infinite_handler(signum, frame):
    """Handler dla nieskończonego treningu - ustawia flagę pauzy."""
    global TRAINING_PAUSED
    TRAINING_PAUSED = True
    print("\n\n" + "="*60)
    print("  PRZERWANIE TRENINGU (Ctrl+C)")
    print("  Trening zostanie wstrzymany po aktualnym kroku...")
    print("="*60 + "\n")


# Rejestracja obsługi sygnału SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, handler)

def save_model(path):
    """Zapisuje aktualny stan modelu do pliku."""
    global GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_EPOCH, GLOBAL_BEST_ACC, GLOBAL_CLASS_NAMES

    if GLOBAL_MODEL is None:
        raise RuntimeError("Model nie jest zainicjalizowany")

    payload = {
        "epoch": GLOBAL_EPOCH,
        "model_state_dict": _state_dict_to_cpu(GLOBAL_MODEL.state_dict()),
        "optimizer_state_dict": _optimizer_state_to_cpu(GLOBAL_OPTIMIZER.state_dict()),
        "best_acc": GLOBAL_BEST_ACC,
        "class_names": GLOBAL_CLASS_NAMES,
    }

    saved_path = _robust_torch_save(payload, path)

    print(f"Model zapisany do: {saved_path}")

    return saved_path


def save_best_model(path):
    """Zapisuje najlepszy model (jeśli został zachowany) do pliku."""
    global BEST_MODEL_STATE, GLOBAL_BEST_ACC
    
    if BEST_MODEL_STATE is None:
        print("Brak zapisanego najlepszego modelu - zapisuję aktualny stan.")
        return save_model(path)
    
    saved_path = _robust_torch_save(BEST_MODEL_STATE, path)
    print(f"Najlepszy model (acc: {BEST_MODEL_STATE.get('best_acc', 0):.2f}%) zapisany do: {saved_path}")
    return saved_path


def capture_best_model():
    """Przechwytuje aktualny stan modelu jako najlepszy."""
    global BEST_MODEL_STATE, GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_EPOCH, GLOBAL_BEST_ACC, GLOBAL_CLASS_NAMES
    
    if GLOBAL_MODEL is None:
        return
    
    BEST_MODEL_STATE = {
        "epoch": GLOBAL_EPOCH,
        "model_state_dict": _state_dict_to_cpu(GLOBAL_MODEL.state_dict()),
        "optimizer_state_dict": _optimizer_state_to_cpu(GLOBAL_OPTIMIZER.state_dict()),
        "best_acc": GLOBAL_BEST_ACC,
        "class_names": GLOBAL_CLASS_NAMES,
    }
    # Zapisz też automatycznie do pliku
    save_best_model(MODEL_PATH)


def init_or_load_model(num_classes, model_path=None):
    global GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_CRITERION
    global GLOBAL_DEVICE, GLOBAL_EPOCH, GLOBAL_BEST_ACC, GLOBAL_CLASS_NAMES

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
            GLOBAL_CLASS_NAMES = checkpoint.get("class_names", GLOBAL_CLASS_NAMES)
        else:
            GLOBAL_MODEL.load_state_dict(checkpoint)


def train_model(epochs=10, batch_size=32, model_path=None):
    global GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_CRITERION
    global GLOBAL_DEVICE, GLOBAL_EPOCH, GLOBAL_BEST_ACC, GLOBAL_CLASS_NAMES

    download_dataset()

    train_transform = get_train_transform()
    dataset = datasets.ImageFolder(root=EXTRACTED_DIR, transform=train_transform)
    GLOBAL_CLASS_NAMES = list(dataset.classes)
    _print_dataset_class_diagnostics(GLOBAL_CLASS_NAMES)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # init albo load
    if GLOBAL_MODEL is None:
        init_or_load_model(len(dataset.classes), model_path)
        
    total_steps = len(train_loader)
    training_start = time.time()
    target_epoch = GLOBAL_EPOCH + epochs

    print("\n" + "=" * 60)
    print("  TRENING OCR")
    print("=" * 60)
    print(f"  Urządzenie: {GLOBAL_DEVICE}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epoki do wykonania: {epochs}")
    print(f"  Zakres epok: {GLOBAL_EPOCH + 1} -> {target_epoch}")
    print("=" * 60 + "\n")

    for epoch in range(GLOBAL_EPOCH, target_epoch):
        epoch_start = time.time()
        GLOBAL_MODEL.train()
        running_loss = 0.0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(GLOBAL_DEVICE), labels.to(GLOBAL_DEVICE)

            GLOBAL_OPTIMIZER.zero_grad(set_to_none=True)
            outputs = GLOBAL_MODEL(images)
            loss = GLOBAL_CRITERION(outputs, labels)
            loss.backward()
            GLOBAL_OPTIMIZER.step()

            running_loss += loss.item()

            # log co 50 kroków
            if step % 50 == 0 or step == total_steps:
                print(f"Epoch [{epoch+1}/{target_epoch}] Step [{step}/{total_steps}] Loss: {loss.item():.4f}")

            # Nie trzymaj referencji do tensorów dłużej niż trzeba.
            del outputs, loss, images, labels
            
        # walidacja
        GLOBAL_MODEL.eval()
        correct, total = 0, 0

        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(GLOBAL_DEVICE), labels.to(GLOBAL_DEVICE)
                outputs = GLOBAL_MODEL(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            del outputs, predicted, images, labels

        val_acc = 100 * correct / total
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / total_steps
        print(
            f"Epoch [{epoch+1}/{target_epoch}] - Val Acc: {val_acc:.2f}% "
            f"| Avg Loss: {avg_loss:.4f} | Czas epoki: {epoch_time:.1f}s"
        )

        GLOBAL_EPOCH = epoch + 1

        if val_acc > GLOBAL_BEST_ACC:
            GLOBAL_BEST_ACC = val_acc
            save_model(MODEL_PATH)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    total_training_time = time.time() - training_start
    print("\n" + "=" * 60)
    print("  PODSUMOWANIE TRENINGU")
    print("=" * 60)
    print(f"  Zakończona epoka: {GLOBAL_EPOCH}")
    print(f"  Najlepsza dokładność: {GLOBAL_BEST_ACC:.2f}%")
    print(f"  Całkowity czas treningu: {total_training_time:.1f}s")
    print("=" * 60)


def show_infinite_menu():
    """Wyświetla interaktywne menu po przerwaniu nieskończonego treningu."""
    global TRAINING_PAUSED, TRAINING_STOP, GLOBAL_EPOCH, GLOBAL_BEST_ACC
    
    while True:
        print("\n" + "="*60)
        print("  MENU NIESKOŃCZONEGO TRENINGU")
        print("="*60)
        print(f"  Aktualny stan:")
        print(f"    - Epoka: {GLOBAL_EPOCH}")
        print(f"    - Najlepsza dokładność: {GLOBAL_BEST_ACC:.2f}%")
        print("-"*60)
        print("  1. Kontynuuj trening")
        print("  2. Zapisz najlepszy model i kontynuuj")
        print("  3. Zapisz najlepszy model i zakończ")
        print("  4. Zapisz checkpoint i zakończ")
        print("  5. Zakończ bez zapisywania")
        print("="*60)
        
        try:
            choice = input("\nWybierz opcję (1-5): ").strip()
        except EOFError:
            choice = "5"
        
        if choice == "1":
            print("\nWznawianie treningu...")
            TRAINING_PAUSED = False
            return True  # kontynuuj
            
        elif choice == "2":
            print("\nZapisywanie najlepszego modelu...")
            save_best_model(MODEL_PATH)
            print("Wznawianie treningu...")
            TRAINING_PAUSED = False
            return True  # kontynuuj
            
        elif choice == "3":
            print("\nZapisywanie najlepszego modelu...")
            save_best_model(MODEL_PATH)
            print("Zakańczanie treningu...")
            TRAINING_STOP = True
            TRAINING_PAUSED = False
            return False  # zakończ
            
        elif choice == "4":
            print("\nZapisywanie checkpointu...")
            save_model(CHECKPOINT_PATH)
            print("Zakańczanie treningu...")
            TRAINING_STOP = True
            TRAINING_PAUSED = False
            return False  # zakończ
            
        elif choice == "5":
            print("\nZakańczanie bez zapisywania...")
            TRAINING_STOP = True
            TRAINING_PAUSED = False
            return False  # zakończ
            
        else:
            print("\nNieprawidłowy wybór! Wybierz 1-5.")


def infinite_train(batch_size=32, model_path=None, checkpoint_interval=5):
    """
    Nieskończony trening modelu OCR.
    
    Trening trwa do momentu przerwania przez użytkownika (Ctrl+C).
    Po przerwaniu wyświetlane jest menu z opcjami:
    - kontynuacji treningu
    - zapisania najlepszego modelu i kontynuacji
    - zapisania najlepszego modelu i zakończenia
    - zapisania checkpointu i zakończenia
    - zakończenia bez zapisywania
    
    Args:
        batch_size: Rozmiar batcha (domyślnie 32)
        model_path: Ścieżka do modelu do wczytania (opcjonalne)
        checkpoint_interval: Co ile epok zapisywać checkpoint (domyślnie 5)
    """
    global GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_CRITERION
    global GLOBAL_DEVICE, GLOBAL_EPOCH, GLOBAL_BEST_ACC, GLOBAL_CLASS_NAMES
    global TRAINING_PAUSED, TRAINING_STOP
    
    # Reset flag
    TRAINING_PAUSED = False
    TRAINING_STOP = False
    
    # Ustaw handler dla nieskończonego treningu
    old_handler = signal.signal(signal.SIGINT, infinite_handler)
    
    try:
        download_dataset()
        
        train_transform = get_train_transform()
        dataset = datasets.ImageFolder(root=EXTRACTED_DIR, transform=train_transform)
        GLOBAL_CLASS_NAMES = list(dataset.classes)
        _print_dataset_class_diagnostics(GLOBAL_CLASS_NAMES)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # init albo load
        if GLOBAL_MODEL is None:
            init_or_load_model(len(dataset.classes), model_path)
        
        print("\n" + "="*60)
        print("  NIESKOŃCZONY TRENING OCR")
        print("="*60)
        print(f"  Urządzenie: {GLOBAL_DEVICE}")
        print(f"  Batch size: {batch_size}")
        print(f"  Klasy: {len(dataset.classes)}")
        print(f"  Próbki treningowe: {len(train_dataset)}")
        print(f"  Próbki walidacyjne: {len(val_dataset)}")
        print(f"  Checkpoint co: {checkpoint_interval} epok")
        print("-"*60)
        print("  Naciśnij Ctrl+C aby wstrzymać i wyświetlić menu")
        print("="*60 + "\n")
        
        total_steps = len(train_loader)
        start_time = datetime.now()
        
        epoch = GLOBAL_EPOCH
        while not TRAINING_STOP:
            # Sprawdź czy pauza
            if TRAINING_PAUSED:
                should_continue = show_infinite_menu()
                if not should_continue:
                    break
                continue
            
            epoch_start = time.time()
            GLOBAL_MODEL.train()
            running_loss = 0.0
            
            for step, (images, labels) in enumerate(train_loader, start=1):
                # Sprawdź czy pauza w trakcie epoki
                if TRAINING_PAUSED:
                    break
                    
                images, labels = images.to(GLOBAL_DEVICE), labels.to(GLOBAL_DEVICE)
                
                GLOBAL_OPTIMIZER.zero_grad(set_to_none=True)
                outputs = GLOBAL_MODEL(images)
                loss = GLOBAL_CRITERION(outputs, labels)
                loss.backward()
                GLOBAL_OPTIMIZER.step()
                
                running_loss += loss.item()
                
                # log co 50 kroków
                if step % 50 == 0 or step == total_steps:
                    elapsed = datetime.now() - start_time
                    print(f"Epoch [{epoch+1}] Step [{step}/{total_steps}] "
                          f"Loss: {loss.item():.4f} | Czas: {elapsed}")

                # Zwolnij referencje po każdym kroku dla stabilności długiego treningu.
                del outputs, loss, images, labels
            
            # Jeśli pauza podczas kroku - wróć do początku pętli
            if TRAINING_PAUSED:
                continue
            
            # walidacja
            GLOBAL_MODEL.eval()
            correct, total = 0, 0
            
            with torch.inference_mode():
                for images, labels in val_loader:
                    images, labels = images.to(GLOBAL_DEVICE), labels.to(GLOBAL_DEVICE)
                    outputs = GLOBAL_MODEL(images)
                    _, predicted = torch.max(outputs, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    del outputs, predicted, images, labels
            
            val_acc = 100 * correct / total
            epoch_time = time.time() - epoch_start
            avg_loss = running_loss / total_steps
            
            print(f"\n>>> Epoch [{epoch+1}] zakończona")
            print(f"    Val Acc: {val_acc:.2f}% | Avg Loss: {avg_loss:.4f} | Czas: {epoch_time:.1f}s")
            
            GLOBAL_EPOCH = epoch + 1
            epoch = GLOBAL_EPOCH
            
            # Zapisz najlepszy model jeśli poprawa
            if val_acc > GLOBAL_BEST_ACC:
                old_best = GLOBAL_BEST_ACC
                GLOBAL_BEST_ACC = val_acc
                capture_best_model()
                print(f"    [NEW BEST] Poprawa: {old_best:.2f}% -> {val_acc:.2f}%")
            
            # Okresowy checkpoint
            if epoch % checkpoint_interval == 0:
                save_model(CHECKPOINT_PATH)
                print(f"    [CHECKPOINT] Zapisano checkpoint (epoka {epoch})")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            print()
        
        # Podsumowanie końcowe
        total_time = datetime.now() - start_time
        print("\n" + "="*60)
        print("  TRENING ZAKOŃCZONY")
        print("="*60)
        print(f"  Całkowity czas: {total_time}")
        print(f"  Epoki: {GLOBAL_EPOCH}")
        print(f"  Najlepsza dokładność: {GLOBAL_BEST_ACC:.2f}%")
        print("="*60)
        
    finally:
        # Przywróć oryginalny handler
        signal.signal(signal.SIGINT, old_handler)

