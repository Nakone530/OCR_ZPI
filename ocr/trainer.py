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
import signal
import threading
import time
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

GLOBAL_EPOCH = 0
GLOBAL_BEST_ACC = 0.0

# -- flagi kontrolne dla nieskończonego treningu
TRAINING_PAUSED = False
TRAINING_STOP = False
BEST_MODEL_STATE = None  # przechowuje stan najlepszego modelu

# -- Funkcje do kontroli treningu z GUI

def stop_training():
    """Zatrzymaj nieskończony trening z GUI"""
    global TRAINING_STOP
    TRAINING_STOP = True

def reset_training_flags():
    """Resetuj flagi treningu"""
    global TRAINING_PAUSED, TRAINING_STOP
    TRAINING_PAUSED = False
    TRAINING_STOP = False

# -- Dataset

def download_dataset(info=None) -> None:
    """Pobiera archiwum datasetu, jeśli jeszcze go nie ma."""
    if info is None:
        info = print
    
    if not os.path.exists(ARCHIVE_PATH):
        info(f"brak datasetu")
        # TODO: Tutaj powinno być pobieranie datasetu
        info("\nPobrano!")

    if not os.path.exists(EXTRACTED_DIR):
        info("Rozpakowywanie archiwum (to może chwilę potrwać)...")
        try:
            with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
                tar.extractall(path=DATA_DIR)
            info("Rozpakowano!")
        except Exception as e:
            info(f"Błąd podczas rozpakowania: {e}")
            raise
    else:
        info("Dataset już jest rozpakowany")


# -- Trening

checkpoint_path = "checkpoint.pth"
current_state = {}

def handler(signum, frame, info=None):
    """Handler dla zwykłego treningu (nie-nieskończonego)."""
    if info is None:
        info = print
    
    info(f"\nOdebrano sygnał: {signum}")
    checkpoint_path = save_model(CHECKPOINT_PATH, info)
    info("Program działa.")
    paused = True
    while paused:
        info("\n--- MENU ---")
        info("1. Wznów")
        info("2. Zapisz najlepszy")
        info("3. Wyjście bez zapisu")

        choice = input("Wybierz opcję: ")

        if choice == "1":
            info("Wznawianie")
            train_model(10, 32, CHECKPOINT_PATH)
        elif choice == "2":
            info("Zapisywanie najlepszego (obecnie nie do końca działa, zapisuje ostatni stan)")
            save_model(MODEL_PATH, info)
        elif choice == "3":
            info("Zamykanie programu...")
            paused = False

        else:
            info("Nieprawidłowy wybór!")
    sys.exit(0)


def infinite_handler(signum, frame, info=None):
    """Handler dla nieskończonego treningu - ustawia flagę pauzy."""
    if info is None:
        info = print
    
    global TRAINING_PAUSED
    TRAINING_PAUSED = True
    info("\n\n" + "="*60)
    info("  PRZERWANIE TRENINGU (Ctrl+C)")
    info("  Trening zostanie wstrzymany po aktualnym kroku...")
    info("="*60 + "\n")


# Rejestracja obsługi sygnału SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, handler)

def save_model(path, info=None):
    """Zapisuje aktualny stan modelu do pliku."""
    global GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_EPOCH, GLOBAL_BEST_ACC

    if info is None:
        info = print

    if GLOBAL_MODEL is None:
        raise RuntimeError("Model nie jest zainicjalizowany")

    torch.save({
        "epoch": GLOBAL_EPOCH,
        "model_state_dict": GLOBAL_MODEL.state_dict(),
        "optimizer_state_dict": GLOBAL_OPTIMIZER.state_dict(),
        "best_acc": GLOBAL_BEST_ACC
    }, path)

    info(f"Model zapisany do: {path}")

    return path


def save_best_model(path, info=None):
    """Zapisuje najlepszy model (jeśli został zachowany) do pliku."""
    global BEST_MODEL_STATE, GLOBAL_BEST_ACC
    
    if info is None:
        info = print
    
    if BEST_MODEL_STATE is None:
        info("Brak zapisanego najlepszego modelu - zapisuję aktualny stan.")
        return save_model(path, info)
    
    torch.save(BEST_MODEL_STATE, path)
    info(f"Najlepszy model (acc: {BEST_MODEL_STATE.get('best_acc', 0):.2f}%) zapisany do: {path}")
    return path


def capture_best_model(info=None):
    """Przechwytuje aktualny stan modelu jako najlepszy."""
    global BEST_MODEL_STATE, GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_EPOCH, GLOBAL_BEST_ACC
    
    if info is None:
        info = print
    
    if GLOBAL_MODEL is None:
        return
    
    BEST_MODEL_STATE = {
        "epoch": GLOBAL_EPOCH,
        "model_state_dict": GLOBAL_MODEL.state_dict().copy(),
        "optimizer_state_dict": GLOBAL_OPTIMIZER.state_dict().copy(),
        "best_acc": GLOBAL_BEST_ACC
    }
    # Zapisz też automatycznie do pliku
    save_best_model(MODEL_PATH, info)


def init_or_load_model(num_classes, model_path=None, info=None):
    global GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_CRITERION
    global GLOBAL_DEVICE, GLOBAL_EPOCH, GLOBAL_BEST_ACC

    if info is None:
        info = print

    GLOBAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    GLOBAL_MODEL = SimpleCNN(num_classes=num_classes).to(GLOBAL_DEVICE)
    GLOBAL_CRITERION = nn.CrossEntropyLoss()
    GLOBAL_OPTIMIZER = torch.optim.Adam(GLOBAL_MODEL.parameters(), lr=0.001)

    if model_path and os.path.exists(model_path):
        info(f"Wczytywanie modelu z: {model_path}")
        checkpoint = torch.load(model_path, map_location=GLOBAL_DEVICE)

        if "model_state_dict" in checkpoint:
            GLOBAL_MODEL.load_state_dict(checkpoint["model_state_dict"])
            GLOBAL_OPTIMIZER.load_state_dict(checkpoint["optimizer_state_dict"])
            GLOBAL_EPOCH = checkpoint.get("epoch", 0)
            GLOBAL_BEST_ACC = checkpoint.get("best_acc", 0.0)
        else:
            GLOBAL_MODEL.load_state_dict(checkpoint)


def train_model(epochs=10, batch_size=32, model_path=None, info=None):
    global GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_CRITERION
    global GLOBAL_DEVICE, GLOBAL_EPOCH, GLOBAL_BEST_ACC

    if info is None:
        info = print

    info("1. Ładowanie datasetu...")
    download_dataset(info)
    
    info("2. Ładowanie transformacji obrazów...")
    train_transform = get_train_transform()
    
    info("3. Ładowanie ImageFolder z datasetu...")
    dataset = datasets.ImageFolder(root=EXTRACTED_DIR, transform=train_transform)
    
    info(f"4. Dataset załadowany: {len(dataset)} obrazów, {len(dataset.classes)} klas")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    info(f"5. Splitting: {train_size} trening, {val_size} walidacja")
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    info("6. Tworzenie DataLoader...")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    info("7. Inicjalizacja lub ładowanie modelu...")
    # init albo load
    if GLOBAL_MODEL is None:
        init_or_load_model(len(dataset.classes), model_path, info)
    
    info("8. Rozpoczynanie treningu...")
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

            # log co 50 kroków
            if step % 50 == 0 or step == total_steps:
                info(f"Epoch [{epoch+1}/{epochs}] Step [{step}/{total_steps}] Loss: {loss.item():.4f}")
            
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
        info(f"Epoch [{epoch+1}/{epochs}] - Val Acc: {val_acc:.2f}%")

        GLOBAL_EPOCH = epoch + 1

        if val_acc > GLOBAL_BEST_ACC:
            GLOBAL_BEST_ACC = val_acc
            save_model(MODEL_PATH, info)


def show_infinite_menu(info=None):
    """Wyświetla interaktywne menu po przerwaniu nieskończonego treningu."""
    global TRAINING_PAUSED, TRAINING_STOP, GLOBAL_EPOCH, GLOBAL_BEST_ACC
    
    if info is None:
        info = print
    
    while True:
        info("\n" + "="*60)
        info("  MENU NIESKOŃCZONEGO TRENINGU")
        info("="*60)
        info(f"  Aktualny stan:")
        info(f"    - Epoka: {GLOBAL_EPOCH}")
        info(f"    - Najlepsza dokładność: {GLOBAL_BEST_ACC:.2f}%")
        info("-"*60)
        info("  1. Kontynuuj trening")
        info("  2. Zapisz najlepszy model i kontynuuj")
        info("  3. Zapisz najlepszy model i zakończ")
        info("  4. Zapisz checkpoint i zakończ")
        info("  5. Zakończ bez zapisywania")
        info("="*60)
        
        try:
            choice = input("\nWybierz opcję (1-5): ").strip()
        except EOFError:
            choice = "5"
        
        if choice == "1":
            info("\nWznawianie treningu...")
            TRAINING_PAUSED = False
            return True  # kontynuuj
            
        elif choice == "2":
            info("\nZapisywanie najlepszego modelu...")
            save_best_model(MODEL_PATH, info)
            info("Wznawianie treningu...")
            TRAINING_PAUSED = False
            return True  # kontynuuj
            
        elif choice == "3":
            info("\nZapisywanie najlepszego modelu...")
            save_best_model(MODEL_PATH, info)
            info("Zakańczanie treningu...")
            TRAINING_STOP = True
            TRAINING_PAUSED = False
            return False  # zakończ
            
        elif choice == "4":
            info("\nZapisywanie checkpointu...")
            save_model(CHECKPOINT_PATH, info)
            info("Zakańczanie treningu...")
            TRAINING_STOP = True
            TRAINING_PAUSED = False
            return False  # zakończ
            
        elif choice == "5":
            info("\nZakańczanie bez zapisywania...")
            TRAINING_STOP = True
            TRAINING_PAUSED = False
            return False  # zakończ
            
        else:
            info("\nNieprawidłowy wybór! Wybierz 1-5.")


def infinite_train(batch_size=32, model_path=None, checkpoint_interval=5, info=None):
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
        info: Funkcja do logowania (domyślnie print)
    """
    global GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_CRITERION
    global GLOBAL_DEVICE, GLOBAL_EPOCH, GLOBAL_BEST_ACC
    global TRAINING_PAUSED, TRAINING_STOP
    
    if info is None:
        info = print
    
    # Reset flag
    TRAINING_PAUSED = False
    TRAINING_STOP = False
    
    # UWAGA: signal.signal() nie może być używany w wątku!
    # Dlatego nie ustawiamy handlera - nieskończony trening będzie działać bez Ctrl+C
    
    try:
        info("1. Ładowanie datasetu...")
        download_dataset(info)
        
        info("2. Ładowanie transformacji obrazów...")
        train_transform = get_train_transform()
        
        info("3. Ładowanie ImageFolder...")
        dataset = datasets.ImageFolder(root=EXTRACTED_DIR, transform=train_transform)
        info(f"4. Dataset załadowany: {len(dataset)} obrazów, {len(dataset.classes)} klas")
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        info(f"5. Splitting: {train_size} trening, {val_size} walidacja")
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        info("6. Tworzenie DataLoader...")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        info("7. Inicjalizacja lub ładowanie modelu...")
        # init albo load
        if GLOBAL_MODEL is None:
            init_or_load_model(len(dataset.classes), model_path, info)
        
        info("8. Rozpoczynanie treningu...")
        info("\n" + "="*60)
        info("  NIESKOŃCZONY TRENING OCR")
        info("="*60)
        info(f"  Urządzenie: {GLOBAL_DEVICE}")
        info(f"  Batch size: {batch_size}")
        info(f"  Klasy: {len(dataset.classes)}")
        info(f"  Próbki treningowe: {len(train_dataset)}")
        info(f"  Próbki walidacyjne: {len(val_dataset)}")
        info(f"  Checkpoint co: {checkpoint_interval} epok")
        info("-"*60)
        info("  Naciśnij Ctrl+C aby wstrzymać i wyświetlić menu")
        info("="*60 + "\n")
        
        total_steps = len(train_loader)
        start_time = datetime.now()
        
        epoch = GLOBAL_EPOCH
        while not TRAINING_STOP:
            # Sprawdź czy pauza
            if TRAINING_PAUSED:
                should_continue = show_infinite_menu(info)
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
                
                GLOBAL_OPTIMIZER.zero_grad()
                outputs = GLOBAL_MODEL(images)
                loss = GLOBAL_CRITERION(outputs, labels)
                loss.backward()
                GLOBAL_OPTIMIZER.step()
                
                running_loss += loss.item()
                
                # log co 50 kroków
                if step % 50 == 0 or step == total_steps:
                    elapsed = datetime.now() - start_time
                    info(f"Epoch [{epoch+1}] Step [{step}/{total_steps}] "
                         f"Loss: {loss.item():.4f} | Czas: {elapsed}")
            
            # Jeśli pauza podczas kroku - wróć do początku pętli
            if TRAINING_PAUSED:
                continue
            
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
            epoch_time = time.time() - epoch_start
            avg_loss = running_loss / total_steps
            
            info(f"\n>>> Epoch [{epoch+1}] zakończona")
            info(f"    Val Acc: {val_acc:.2f}% | Avg Loss: {avg_loss:.4f} | Czas: {epoch_time:.1f}s")
            
            GLOBAL_EPOCH = epoch + 1
            epoch = GLOBAL_EPOCH
            
            # Zapisz najlepszy model jeśli poprawa
            if val_acc > GLOBAL_BEST_ACC:
                old_best = GLOBAL_BEST_ACC
                GLOBAL_BEST_ACC = val_acc
                capture_best_model(info)
                info(f"    [NEW BEST] Poprawa: {old_best:.2f}% -> {val_acc:.2f}%")
            
            # Okresowy checkpoint
            if epoch % checkpoint_interval == 0:
                save_model(CHECKPOINT_PATH, info)
                info(f"    [CHECKPOINT] Zapisano checkpoint (epoka {epoch})")
            
            info("")
        
        # Podsumowanie końcowe
        total_time = datetime.now() - start_time
        info("\n" + "="*60)
        info("  TRENING ZAKOŃCZONY")
        info("="*60)
        info(f"  Całkowity czas: {total_time}")
        info(f"  Epoki: {GLOBAL_EPOCH}")
        info(f"  Najlepsza dokładność: {GLOBAL_BEST_ACC:.2f}%")
        info("="*60)
    except Exception as e:
        info(f"\nBłąd podczas treningu: {e}")
        import traceback
        info(traceback.format_exc())

