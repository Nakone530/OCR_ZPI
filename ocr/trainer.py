"""
Moduł trenowania modelu OCR.

Odpowiedzialności:
  - trening sieci MainModel z walidacją
  - zapis najlepszego modelu na dysk
  - nieskończony trening z możliwością przerwania i kontynuacji
"""

import difflib
import os
import sys
import re
import shutil
import signal
import threading
import time
import gc
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

from .config import (
    DATA_DIR, MODEL_PATH, CHECKPOINT_PATH,
    MODEL_ARCHIVE_DIR, MODEL_ARCHIVE_KEEP_COUNT, IMAGES_DIR, CHARS, char2idx, idx2char, DATA_ROOT_DIR,
    PHSF_DATA_DIR,
)
from .model import MainModel, CNNClassifier
from .utils import get_train_transform, get_cnn_train_transform, load_all_datasets, load_phsf_znaki
from .model_archive import ModelArchiver
from .OCRDataset import OCRDataset
from . import info


# ── Konfiguracje treningowe ───────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Konfiguracja jednego przebiegu treningowego."""
    name: str
    description: str
    epochs: Optional[int] = None        # None = dziedziczy z CLI --epochs
    batch_size: Optional[int] = None    # None = dziedziczy z CLI --batch-size
    denoise_prob: float = 0.3           # prawdopodobieństwo RandomDenoise (0.0-1.0)
    max_padding: int = 20               # maks. padding w pikselach (RandomPadding)
    model_path: Optional[str] = None    # ścieżka zapisu (None = models/model_<name>.pth)
    resume_path: Optional[str] = None   # ścieżka checkpointu do wznowienia


TRAINING_PRESETS: dict = {
    "baseline": TrainingConfig(
        name="baseline",
        description="Model bazowy – brak modyfikacji transformacji",
        denoise_prob=0.0,
        max_padding=0,
    ),
    "p_denoise": TrainingConfig(
        name="p_denoise",
        description="Trening z odszumianiem – RandomDenoise zawsze aktywny (prob=1.0)",
        denoise_prob=1.0,
        max_padding=20,
    ),
    "padding": TrainingConfig(
        name="padding",
        description="Trening z ulepszonym paddingiem – RandomPadding do 40px",
        denoise_prob=0.0,
        max_padding=40,
    ),
    "denoise": TrainingConfig(
        name="denoise",
        description="Trening z odszumianiem – RandomDenoise zawsze aktywny (prob=1.0), bez paddingu",
        denoise_prob=0.0,
        max_padding=40,
    ),
    "padding": TrainingConfig(
        name="padding",
        description="Trening z umiarkowanym paddingiem i denoise",
        denoise_prob=0.5,
        max_padding=30,
    ),
    "p_baseline": TrainingConfig(
        name="p_baseline",
        description="Model bazowy – brak modyfikacji transformacji poza lekkim paddingiem",
        denoise_prob=0.0,
        max_padding=20,
    ),
}


# -- globalne
GLOBAL_MODEL = None
GLOBAL_OPTIMIZER = None
GLOBAL_CRITERION = None
GLOBAL_DEVICE = None
GLOBAL_SCHEDULER = None
GLOBAL_CLASS_NAMES = None

GLOBAL_EPOCH = 0
GLOBAL_BEST_ACC = 0.0
GLOBAL_VAL_ACCURACY = 0.0

_GLOBAL_MAJOR_VERSION = None
_GLOBAL_MINOR_VERSION = None
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


def pad_images(images):
    """
    Padding obrazów do tej samej szerokości (OCR/CRNN).
    Zakłada: obrazy są tensorami formatu (C, H, W) o tym samym H, różnym W.
    """
    # Konwertuj na tensory jeśli potrzeba
    tensor_images = []
    for img in images:
        if isinstance(img, torch.Tensor):
            tensor_images.append(img)
        else:
            # Jeśli PIL Image, konwertuj do tensora
            import torchvision.transforms as T
            img_tensor = T.ToTensor()(img)
            tensor_images.append(img_tensor)
    
    images = tensor_images
    max_w = max(img.shape[-1] for img in images)

    padded = []
    for img in images:
        if img.dim() == 2:  # (H, W) -> dodaj kanał
            img = img.unsqueeze(0)
        
        _, h, w = img.shape
        pad_w = max_w - w

        # (left, right, top, bottom)
        img = F.pad(img, (0, pad_w, 0, 0), value=255)
        padded.append(img)

    return torch.stack(padded)

def _tight_crop(gray: np.ndarray) -> np.ndarray:
    """Przycina obraz do obszaru zawierającego piksele znaku."""
    rows = np.where(np.sum(gray < 128, axis=1) > 0)[0]
    cols = np.where(np.sum(gray < 128, axis=0) > 0)[0]
    if len(rows) == 0 or len(cols) == 0:
        return gray
    return gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def collate_fn(batch):
    images, texts = zip(*batch)

    images = list(images)
    #images = pad_images(images)
    processed_images = []

    for img in images:
        # jeśli to PIL → numpy
        if hasattr(img, "convert"):
            img = np.array(img.convert("L"))
        elif img.ndim == 3:
            img = np.array(img)

        img = _tight_crop(img)
        processed_images.append(torch.tensor(img, dtype=torch.float32))

    images = torch.stack(processed_images)
    
    targets = []
    target_lengths = []

    for t in texts:
        t = t.lower()  # ważne!
        encoded = [char2idx[c] for c in t if c in char2idx]
        targets.extend(encoded)
        target_lengths.append(len(encoded))

    targets = torch.tensor(targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)


    return images, targets, target_lengths


def collate_fn_cnn(batch):
    """
    Collate dla treningu CNN na pojedynczych znakach.
    Zakłada: każdy element to (tensor 1×32×32, tekst pojedynczego znaku).
    Zwraca: (images: B×1×32×32, labels: B) – etykiety 0-indeksowane dla CrossEntropy.
    """
    images, texts = zip(*batch)
    images = torch.stack(list(images))
    labels = torch.tensor(
        [char2idx[text] - 1 for text in texts],
        dtype=torch.long,
    )
    return images, labels


# -- Trening

def get_runtime_major_version(folder):
    global _GLOBAL_MAJOR_VERSION

    if _GLOBAL_MAJOR_VERSION is not None:
        return _GLOBAL_MAJOR_VERSION

    pattern = re.compile(r"v(\d+)\.")
    majors = []

    if os.path.exists(folder):
        for name in os.listdir(folder):
            match = pattern.match(name)
            if match:
                majors.append(int(match.group(1)))

    if not majors:
        _GLOBAL_MAJOR_VERSION = 1
    else:
        _GLOBAL_MAJOR_VERSION = max(majors) + 1

    return _GLOBAL_MAJOR_VERSION



def get_runtime_minor_version(folder, major):
    global _GLOBAL_MINOR_VERSION

    if _GLOBAL_MINOR_VERSION is not None:
        return _GLOBAL_MINOR_VERSION

    pattern = re.compile(
        rf"v{major}\.(\d+)"
    )

    minors = []

    if os.path.exists(folder):

        for name in os.listdir(folder):

            match = pattern.match(name)

            if match:
                minors.append(int(match.group(1)))

    if not minors:
        _GLOBAL_MINOR_VERSION = 1
    else:
        _GLOBAL_MINOR_VERSION = max(minors) + 1

    return _GLOBAL_MINOR_VERSION

checkpoint_path = "checkpoint.pth"
current_state = {}


def _robust_torch_save(payload: dict, path: str, retries: int = 3, delay_s: float = 0.4) -> str:
    """Próbuje zapisać plik kilka razy; przy blokadzie używa pliku awaryjnego."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            torch.save(payload, path)
            return path
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(delay_s)

    base, ext = os.path.splitext(path)
    fallback = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
    os.makedirs(os.path.dirname(os.path.abspath(fallback)), exist_ok=True)
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
            info("Zapisywanie najlepszego")
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

def _compute_val_char_accuracy(model, val_loader, device) -> float:
    """Oblicza dokładność znakową modelu na zbiorze walidacyjnym.

    Używa CTC greedy decoding i SequenceMatcher do porównania z referencją.
    Zwraca średnią ważoną długością tekstu referencyjnego (0–100%).
    """
    total_weight = 0.0
    weighted_sum = 0.0

    model.eval()
    with torch.inference_mode():
        for images, targets, target_lengths in val_loader:
            images = images.to(device)
            outputs = model(images)
            log_probs = outputs.log_softmax(2)
            preds = log_probs.argmax(2)  # (T, B)

            target_offset = 0
            for b in range(images.size(0)):
                tl = target_lengths[b].item()
                gt_indices = targets[target_offset:target_offset + tl].tolist()
                gt_text = ''.join(idx2char.get(i, '') for i in gt_indices)
                target_offset += tl

                prev = 0
                pred_chars = []
                for t in range(preds.size(0)):
                    p = int(preds[t, b].item())
                    if p != prev and p != 0:
                        pred_chars.append(idx2char.get(p, ''))
                    prev = p
                pred_text = ''.join(pred_chars)

                weight = len(gt_text)
                if weight > 0:
                    ratio = difflib.SequenceMatcher(None, pred_text.lower(), gt_text.lower()).ratio()
                    weighted_sum += ratio * weight
                    total_weight += weight

    return (weighted_sum / total_weight * 100) if total_weight > 0 else 0.0


def get_stored_accuracy(model_path=None) -> float:
    """Zwraca val_char_accuracy zapisaną w checkpoincie lub -1.0 gdy brak."""
    path = model_path or MODEL_PATH
    if not os.path.exists(path):
        return -1.0
    try:
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(checkpoint, dict):
            return checkpoint.get('val_char_accuracy', -1.0)
    except Exception:
        return -1.0
    return -1.0


def _make_backup(path: str) -> str | None:
    """Tworzy kopię zapasową pliku modelu (.bak) przed nadpisaniem."""
    if os.path.exists(path):
        backup = path + ".bak"
        shutil.copy2(path, backup)
        return backup
    return None


def _make_backup_as(src: str, dst: str) -> bool:
    """Kopiuje plik src do dst (backup pod wskazaną nazwą). Zwraca True jeśli sukces."""
    if os.path.exists(src):
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def restore_from_backup(path: str = None, info=None) -> bool:
    """Przywraca model z kopii zapasowej (.bak). Zwraca True jeśli sukces."""
    if info is None:
        info = print
    path = path or MODEL_PATH
    backup = path + ".bak"
    if os.path.exists(backup):
        shutil.copy2(backup, path)
        info(f"Model przywrócony z: {backup}")
        return True
    info(f"Brak kopii zapasowej: {backup}")
    return False


def _archive_previous_model(current_model_path):
    """Archiwizuje poprzedni model jeśli istnieje."""
    try:
        if os.path.exists(current_model_path):
            archiver = ModelArchiver(MODEL_ARCHIVE_DIR)
            archiver.archive_model(
                current_model_path,
                accuracy=GLOBAL_BEST_ACC,
                epoch=GLOBAL_EPOCH,
                tags=["auto-archived"]
            )
            # Czyszczenie starych archiwów
            archiver.cleanup_old_archives(keep_count=MODEL_ARCHIVE_KEEP_COUNT)
    except Exception as e:
        info(f"[WARN] Błąd podczas archiwizacji modelu: {e}")

def save_model(path, info=None):
    """Zapisuje aktualny stan modelu do pliku."""
    global GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_EPOCH, GLOBAL_BEST_ACC, GLOBAL_CLASS_NAMES, GLOBAL_VAL_ACCURACY

    if info is None:
        info = print

    if GLOBAL_MODEL is None:
        raise RuntimeError("Model nie jest zainicjalizowany")

    if path == MODEL_PATH:
        _make_backup(path)

    payload = {
        "epoch": GLOBAL_EPOCH,
        "model_state_dict": _state_dict_to_cpu(GLOBAL_MODEL.state_dict()),
        "optimizer_state_dict": _optimizer_state_to_cpu(GLOBAL_OPTIMIZER.state_dict()),
        "best_loss": GLOBAL_BEST_ACC,
        "class_names": GLOBAL_CLASS_NAMES,
        "val_char_accuracy": GLOBAL_VAL_ACCURACY,
    }

    saved_path = _robust_torch_save(payload, path)

    info(f"Model zapisany do: {saved_path}")

    # Archiwizuj poprzedni model jeśli jest to główny model
    if path == MODEL_PATH:
        _archive_previous_model(saved_path)

    return saved_path


def save_best_model(path, info=None):
    """Zapisuje najlepszy model (jeśli został zachowany) do pliku."""
    global BEST_MODEL_STATE, GLOBAL_BEST_ACC

    if info is None:
        info = print

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if BEST_MODEL_STATE is None:
        info("Brak zapisanego najlepszego modelu - zapisuję aktualny stan.")
        return save_model(path, info)

    _make_backup(path)
    torch.save(BEST_MODEL_STATE, path)
    acc = BEST_MODEL_STATE.get('val_char_accuracy', None)
    acc_str = f", acc: {acc:.2f}%" if acc is not None else ""
    info(f"Najlepszy model (loss: {BEST_MODEL_STATE.get('best_loss', 0):.4f}{acc_str}) zapisany do: {path}")
    return path


def capture_best_model(info=None, path=None):
    """Przechwytuje aktualny stan modelu jako najlepszy i zapisuje go pod wskazaną ścieżką."""
    global BEST_MODEL_STATE, GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_EPOCH, GLOBAL_BEST_ACC, GLOBAL_CLASS_NAMES, GLOBAL_VAL_ACCURACY

    if info is None:
        info = print

    if GLOBAL_MODEL is None:
        return

    BEST_MODEL_STATE = {
        "epoch": GLOBAL_EPOCH,
        "model_state_dict": _state_dict_to_cpu(GLOBAL_MODEL.state_dict()),
        "optimizer_state_dict": _optimizer_state_to_cpu(GLOBAL_OPTIMIZER.state_dict()),
        "best_loss": GLOBAL_BEST_ACC,
        "class_names": GLOBAL_CLASS_NAMES,
        "val_char_accuracy": GLOBAL_VAL_ACCURACY,
    }
    save_best_model(path or MODEL_PATH, info)


def init_or_load_model(num_classes, model_path=None, info=None):
    global GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_CRITERION
    global GLOBAL_DEVICE, GLOBAL_EPOCH, GLOBAL_BEST_ACC, GLOBAL_CLASS_NAMES, GLOBAL_VAL_ACCURACY

    if info is None:
        info = print

    GLOBAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    GLOBAL_MODEL = MainModel(num_classes=num_classes).to(GLOBAL_DEVICE)
    GLOBAL_CRITERION = nn.CTCLoss(zero_infinity=True)
    GLOBAL_OPTIMIZER = torch.optim.Adam(GLOBAL_MODEL.parameters(), lr=0.0001)

    if model_path and os.path.exists(model_path):
        info(f"Wczytywanie modelu z: {model_path}")
        checkpoint = torch.load(model_path, map_location=GLOBAL_DEVICE)

        if "model_state_dict" in checkpoint:
            GLOBAL_MODEL.load_state_dict(checkpoint["model_state_dict"])
            GLOBAL_OPTIMIZER.load_state_dict(checkpoint["optimizer_state_dict"])
            GLOBAL_EPOCH = checkpoint.get("epoch", 0)
            GLOBAL_BEST_ACC = checkpoint.get("best_acc", 0.0)
            GLOBAL_CLASS_NAMES = checkpoint.get("class_names", GLOBAL_CLASS_NAMES)
            GLOBAL_VAL_ACCURACY = checkpoint.get("val_char_accuracy", 0.0)
        else:
            GLOBAL_MODEL.load_state_dict(checkpoint)


def train_model(epochs=10, batch_size=32, model_path=None, info=None, args=None,
                config: Optional['TrainingConfig'] = None, save_path: Optional[str] = None,
                preloaded_data: Optional[list] = None):
    """
    Trenuje model OCR.

    Args:
        config: TrainingConfig z parametrami transformacji (denoise_prob, max_padding).
                Gdy None – używane są wartości domyślne.
        save_path: ścieżka zapisu wytrenowanego modelu.
                   Gdy None – używana jest stała MODEL_PATH.
    """
    global GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_CRITERION, GLOBAL_SCHEDULER
    global GLOBAL_DEVICE, GLOBAL_EPOCH, GLOBAL_BEST_ACC, GLOBAL_CLASS_NAMES, GLOBAL_VAL_ACCURACY

    if info is None:
        info = print

    effective_save_path = save_path or MODEL_PATH

    # Parametry transformacji z konfiguracji lub wartości domyślne
    denoise_prob = config.denoise_prob if config is not None else 0.3
    max_padding = config.max_padding if config is not None else 20
    checkpoint_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
    checkpoint_saved = {r: False for r in checkpoint_ratios}
    # Odczyt dokładności istniejącego modelu i kopia zapasowa
    prev_accuracy = get_stored_accuracy(effective_save_path)
    backup_path = effective_save_path + ".backup"
    _make_backup_as(effective_save_path, backup_path)
    if prev_accuracy >= 0:
        info(f"Poprzednia dokładność modelu: {prev_accuracy:.2f}%")
    else:
        info("Brak poprzedniego modelu – pierwszy trening.")

    if preloaded_data is not None:
        info(f"3. Używam przekazanego datasetu ({len(preloaded_data)} próbek).")
        data = preloaded_data
    else:
        info("3. Ładowanie datasetu OCR...")
        data = load_phsf_znaki(PHSF_DATA_DIR)
        info(f"   ttData: {len(data)} próbek")

##        if os.path.isdir(PHSF_DATA_DIR):
##            phsf_data = load_phsf_znaki(PHSF_DATA_DIR)
##            info(f"   PHSF znaki: {len(phsf_data)} próbek")
##            data = data + phsf_data
##        else:
##            info(f"   PHSF pominięty (brak katalogu: {PHSF_DATA_DIR})")

    dataset = OCRDataset(
        json_data=data,
        images_dir=None,
        transform=get_train_transform(args, denoise_prob=denoise_prob, max_padding=max_padding)
    )

    info(f"4. Dataset załadowany: {len(dataset)} obrazów łącznie")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    info(f"5. Splitting: {train_size} trening, {val_size} walidacja")
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    info("6. Tworzenie DataLoader...")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    info("7. Inicjalizacja lub ładowanie modelu...")
    # init albo load
    if GLOBAL_MODEL is None:
        init_or_load_model(len(CHARS) + 1, model_path, info)

    if args is not None and getattr(args, "cycle_lr", False):
        step_size = getattr(args, "cycle_step_size", None)
        if step_size is None:
            step_size = max(1, len(train_loader))
        GLOBAL_SCHEDULER = torch.optim.lr_scheduler.CyclicLR(
            GLOBAL_OPTIMIZER,
            base_lr=getattr(args, "cycle_base_lr", 1e-5),
            max_lr=getattr(args, "cycle_max_lr", 1e-4),
            step_size_up=step_size,
            mode="triangular2",
            cycle_momentum=False,
        )
        info(
            f"CyclicLR aktywny: base_lr={getattr(args, 'cycle_base_lr', 1e-5)} "
            f"max_lr={getattr(args, 'cycle_max_lr', 1e-4)} step_size_up={step_size}"
        )
    else:
        GLOBAL_SCHEDULER = None
    
    info("8. Rozpoczynanie treningu...")
    total_steps = len(train_loader)
    training_start = time.time()
    target_epoch = GLOBAL_EPOCH + epochs

    info("\n" + "=" * 60)
    info("  TRENING OCR")
    info("=" * 60)
    info(f"  Urządzenie: {GLOBAL_DEVICE}")
    info(f"  Batch size: {batch_size}")
    info(f"  Epoki do wykonania: {epochs}")
    info(f"  Zakres epok: {GLOBAL_EPOCH + 1} -> {target_epoch}")
    info("=" * 60 + "\n")

    for epoch in range(GLOBAL_EPOCH, target_epoch):
        epoch_start = time.time()
        GLOBAL_MODEL.train()
        running_loss = 0.0

        for step, (images, targets, target_lengths) in enumerate(train_loader, start=1):
            images = images.to(GLOBAL_DEVICE)
            targets = targets.to(GLOBAL_DEVICE)
            target_lengths = target_lengths.to(GLOBAL_DEVICE)

            GLOBAL_OPTIMIZER.zero_grad(set_to_none=True)

            outputs = GLOBAL_MODEL(images)  # (T, B, C)
            log_probs = outputs.log_softmax(2)

            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=outputs.size(0),
                dtype=torch.long,
                device=GLOBAL_DEVICE,
            )

            loss = GLOBAL_CRITERION(
                log_probs,
                targets,
                input_lengths,
                target_lengths
            )

            loss.backward()
            GLOBAL_OPTIMIZER.step()
            if GLOBAL_SCHEDULER is not None:
                GLOBAL_SCHEDULER.step()

            running_loss += loss.item()

            if step % 50 == 0:
                if args is not None and getattr(args, "log_lr", False):
                    lr = GLOBAL_OPTIMIZER.param_groups[0]["lr"]
                    info(f"Epoch {epoch+1} Step {step} Loss: {loss.item():.4f} | LR: {lr:.6g}")
                else:
                    info(f"Epoch {epoch+1} Step {step} Loss: {loss.item():.4f}")

        # WALIDACJA
        GLOBAL_MODEL.eval()
        val_loss = 0.0

        with torch.inference_mode():
            for images, targets, target_lengths in val_loader:
                images = images.to(GLOBAL_DEVICE)
                targets = targets.to(GLOBAL_DEVICE)
                target_lengths = target_lengths.to(GLOBAL_DEVICE)

                outputs = GLOBAL_MODEL(images)
                log_probs = outputs.log_softmax(2)

                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=outputs.size(0),
                    dtype=torch.long,
                    device=GLOBAL_DEVICE,
                )

                loss = GLOBAL_CRITERION(
                    log_probs,
                    targets,
                    input_lengths,
                    target_lengths
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        epoch_time = time.time() - epoch_start
        info(f"Epoch {epoch+1} | train_loss={running_loss:.4f} | val_loss={val_loss:.4f} | time={epoch_time:.1f}s")
        progress = (epoch + 1) / target_epoch

        for ratio in checkpoint_ratios:
            if not checkpoint_saved[ratio] and progress >= ratio:
                info(f"[CHECKPOINT] Saving model at {int(ratio*100)}% (epoch {epoch+1})")
                save_checkpoint_model(GLOBAL_MODEL, epoch + 1, ratio)
                checkpoint_saved[ratio] = True
        if torch.isnan(loss):
            print("NaN detected!")
            print("targets:", target_lengths)
            print("input:", input_lengths)
            print("outputs shape:", outputs.shape)
            continue
        GLOBAL_EPOCH = epoch + 1

    total_training_time = time.time() - training_start

    # Oblicz dokładność nowego modelu i porównaj z poprzednim
    info("\nObliczanie dokładności znakowej na zbiorze walidacyjnym...")
    new_accuracy = _compute_val_char_accuracy(GLOBAL_MODEL, val_loader, GLOBAL_DEVICE)
    GLOBAL_VAL_ACCURACY = new_accuracy
    info(f"Nowa dokładność: {new_accuracy:.2f}%")

    if prev_accuracy < 0:
        info("Pierwszy model – zapisuję jako punkt odniesienia.")
        capture_best_model(info, path=effective_save_path)
    elif new_accuracy >= prev_accuracy:
        info(f"Poprawa: {prev_accuracy:.2f}% -> {new_accuracy:.2f}% – zapisuję nowy model.")
        capture_best_model(info, path=effective_save_path)
    else:
        info(f"Brak poprawy: {prev_accuracy:.2f}% -> {new_accuracy:.2f}% – przywracam poprzedni model.")
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, effective_save_path)

    # Usuń plik backup
    if os.path.exists(backup_path):
        os.remove(backup_path)

    info("\n" + "=" * 60)
    info("  PODSUMOWANIE TRENINGU")
    info("=" * 60)
    info(f"  Zakończona epoka: {GLOBAL_EPOCH}")
    info(f"  Dokładność znakowa (val): {GLOBAL_VAL_ACCURACY:.2f}%")
    info(f"  Całkowity czas treningu: {total_training_time:.1f}s")
    info("=" * 60)

def save_checkpoint_model(model, epoch, ratio):
    f_models = folder = "models"
    major = get_runtime_major_version(f_models)
    minor = get_runtime_minor_version(f_models, major)
    folder = os.path.join(folder, f"v{major}.{minor}")
    os.makedirs(folder, exist_ok=True)
    
    subfolder = os.path.join(folder, f"v{major}.{int(ratio*10) - 4}")
    os.makedirs(subfolder, exist_ok=True)
    mPath = os.path.join(subfolder, f"model.pth")
    
    torch.save(model.state_dict(), mPath)
    dPath = os.path.join(subfolder, f"model_v{major}.{int(ratio*10) - 4}_epoch{epoch}.txt")
    with open(dPath, "w", encoding="utf-8") as f:
        f.write(f"epoch: {epoch}\n")
        f.write(f"ratio: {ratio}\n")
        f.write(f"model_version: v{major}.{int(ratio*10) - 4}\n")
        
def train_crnn(epochs=10, batch_size=32, model_path=None, info=None, args=None,
               config=None, save_path=None, preloaded_data=None):
    """
    Trening modelu CRNN (CNN + LSTM + CTC).
    Dane: ttData – obrazy słów/sekwencji z etykietami tekstowymi.
    Strata: CTCLoss.
    """
    global GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_CRITERION, GLOBAL_SCHEDULER
    global GLOBAL_DEVICE, GLOBAL_EPOCH, GLOBAL_BEST_ACC, GLOBAL_CLASS_NAMES, GLOBAL_VAL_ACCURACY

    if info is None:
        info = print

    effective_save_path = save_path or MODEL_PATH
    denoise_prob = config.denoise_prob if config is not None else 0.3
    max_padding = config.max_padding if config is not None else 20
    checkpoint_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
    checkpoint_saved = {r: False for r in checkpoint_ratios}

    prev_accuracy = get_stored_accuracy(effective_save_path)
    backup_path = effective_save_path + ".backup"
    _make_backup_as(effective_save_path, backup_path)
    if prev_accuracy >= 0:
        info(f"Poprzednia dokładność modelu: {prev_accuracy:.2f}%")
    else:
        info("Brak poprzedniego modelu – pierwszy trening.")

    if preloaded_data is not None:
        info(f"3. Używam przekazanego datasetu ({len(preloaded_data)} próbek).")
        data = preloaded_data
    else:
        info("3. Ładowanie datasetu CRNN (ttData)...")
        data = load_all_datasets(DATA_ROOT_DIR)
        info(f"   ttData: {len(data)} próbek")

    dataset = OCRDataset(
        json_data=data,
        images_dir=None,
        transform=get_train_transform(args, denoise_prob=denoise_prob, max_padding=max_padding),
    )
    info(f"4. Dataset załadowany: {len(dataset)} obrazów łącznie")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    info(f"5. Splitting: {train_size} trening, {val_size} walidacja")
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    info("6. Tworzenie DataLoader...")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    info("7. Inicjalizacja lub ładowanie modelu CRNN...")
    if GLOBAL_MODEL is None:
        init_or_load_model(len(CHARS) + 1, model_path, info)

    if args is not None and getattr(args, "cycle_lr", False):
        step_size = getattr(args, "cycle_step_size", None)
        if step_size is None:
            step_size = max(1, len(train_loader))
        GLOBAL_SCHEDULER = torch.optim.lr_scheduler.CyclicLR(
            GLOBAL_OPTIMIZER,
            base_lr=getattr(args, "cycle_base_lr", 1e-5),
            max_lr=getattr(args, "cycle_max_lr", 1e-4),
            step_size_up=step_size,
            mode="triangular2",
            cycle_momentum=False,
        )
        info(
            f"CyclicLR aktywny: base_lr={getattr(args, 'cycle_base_lr', 1e-5)} "
            f"max_lr={getattr(args, 'cycle_max_lr', 1e-4)} step_size_up={step_size}"
        )
    else:
        GLOBAL_SCHEDULER = None

    info("8. Rozpoczynanie treningu CRNN...")
    total_steps = len(train_loader)
    training_start = time.time()
    target_epoch = GLOBAL_EPOCH + epochs

    info("\n" + "=" * 60)
    info("  TRENING CRNN (ttData)")
    info("=" * 60)
    info(f"  Urządzenie: {GLOBAL_DEVICE}")
    info(f"  Batch size: {batch_size}")
    info(f"  Epoki do wykonania: {epochs}")
    info(f"  Zakres epok: {GLOBAL_EPOCH + 1} -> {target_epoch}")
    info("=" * 60 + "\n")

    for epoch in range(GLOBAL_EPOCH, target_epoch):
        epoch_start = time.time()
        GLOBAL_MODEL.train()
        running_loss = 0.0

        for step, (images, targets, target_lengths) in enumerate(train_loader, start=1):
            images = images.to(GLOBAL_DEVICE)
            targets = targets.to(GLOBAL_DEVICE)
            target_lengths = target_lengths.to(GLOBAL_DEVICE)

            GLOBAL_OPTIMIZER.zero_grad(set_to_none=True)
            outputs = GLOBAL_MODEL(images)
            log_probs = outputs.log_softmax(2)
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=outputs.size(0),
                dtype=torch.long,
                device=GLOBAL_DEVICE,
            )
            loss = GLOBAL_CRITERION(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            GLOBAL_OPTIMIZER.step()
            if GLOBAL_SCHEDULER is not None:
                GLOBAL_SCHEDULER.step()
            running_loss += loss.item()

            if step % 50 == 0:
                if args is not None and getattr(args, "log_lr", False):
                    lr = GLOBAL_OPTIMIZER.param_groups[0]["lr"]
                    info(f"Epoch {epoch+1} Step {step} Loss: {loss.item():.4f} | LR: {lr:.6g}")
                else:
                    info(f"Epoch {epoch+1} Step {step} Loss: {loss.item():.4f}")

        GLOBAL_MODEL.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for images, targets, target_lengths in val_loader:
                images = images.to(GLOBAL_DEVICE)
                targets = targets.to(GLOBAL_DEVICE)
                target_lengths = target_lengths.to(GLOBAL_DEVICE)
                outputs = GLOBAL_MODEL(images)
                log_probs = outputs.log_softmax(2)
                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=outputs.size(0),
                    dtype=torch.long,
                    device=GLOBAL_DEVICE,
                )
                loss = GLOBAL_CRITERION(log_probs, targets, input_lengths, target_lengths)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        epoch_time = time.time() - epoch_start
        info(f"Epoch {epoch+1} | train_loss={running_loss:.4f} | val_loss={val_loss:.4f} | time={epoch_time:.1f}s")

        progress = (epoch + 1) / target_epoch
        for ratio in checkpoint_ratios:
            if not checkpoint_saved[ratio] and progress >= ratio:
                info(f"[CHECKPOINT] Saving model at {int(ratio*100)}% (epoch {epoch+1})")
                save_checkpoint_model(GLOBAL_MODEL, epoch + 1, ratio)
                checkpoint_saved[ratio] = True

        GLOBAL_EPOCH = epoch + 1

    total_training_time = time.time() - training_start

    info("\nObliczanie dokładności znakowej na zbiorze walidacyjnym...")
    new_accuracy = _compute_val_char_accuracy(GLOBAL_MODEL, val_loader, GLOBAL_DEVICE)
    GLOBAL_VAL_ACCURACY = new_accuracy
    info(f"Nowa dokładność: {new_accuracy:.2f}%")

    if prev_accuracy < 0:
        info("Pierwszy model – zapisuję jako punkt odniesienia.")
        capture_best_model(info, path=effective_save_path)
    elif new_accuracy >= prev_accuracy:
        info(f"Poprawa: {prev_accuracy:.2f}% -> {new_accuracy:.2f}% – zapisuję nowy model.")
        capture_best_model(info, path=effective_save_path)
    else:
        info(f"Brak poprawy: {prev_accuracy:.2f}% -> {new_accuracy:.2f}% – przywracam poprzedni model.")
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, effective_save_path)

    if os.path.exists(backup_path):
        os.remove(backup_path)

    info("\n" + "=" * 60)
    info("  PODSUMOWANIE TRENINGU CRNN")
    info("=" * 60)
    info(f"  Zakończona epoka: {GLOBAL_EPOCH}")
    info(f"  Dokładność znakowa (val): {GLOBAL_VAL_ACCURACY:.2f}%")
    info(f"  Całkowity czas treningu: {total_training_time:.1f}s")
    info("=" * 60)


def train_cnn(epochs=10, batch_size=32, model_path=None, info=None, args=None,
              save_path=None, preloaded_data=None):
    """
    Trening modelu CNN do klasyfikacji pojedynczych znaków.
    Dane: phsf – obrazy pojedynczych znaków podzielone na klasy.
    Strata: CrossEntropyLoss.
    """
    global GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_CRITERION, GLOBAL_SCHEDULER
    global GLOBAL_DEVICE, GLOBAL_EPOCH, GLOBAL_BEST_ACC, GLOBAL_CLASS_NAMES, GLOBAL_VAL_ACCURACY

    if info is None:
        info = print

    effective_save_path = save_path or MODEL_PATH

    if preloaded_data is not None:
        info(f"3. Używam przekazanego datasetu ({len(preloaded_data)} próbek).")
        data = preloaded_data
    else:
        info("3. Ładowanie datasetu CNN (phsf)...")
        data = load_phsf_znaki(PHSF_DATA_DIR)
        info(f"   phsf: {len(data)} próbek")

    dataset = OCRDataset(
        json_data=data,
        images_dir=None,
        transform=get_cnn_train_transform(),
    )
    info(f"4. Dataset załadowany: {len(dataset)} obrazów łącznie")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    info(f"5. Splitting: {train_size} trening, {val_size} walidacja")
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    info("6. Tworzenie DataLoader...")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_cnn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_cnn)

    info("7. Inicjalizacja lub ładowanie modelu CNN...")
    GLOBAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GLOBAL_MODEL = CNNClassifier(num_classes=len(CHARS)).to(GLOBAL_DEVICE)
    GLOBAL_CRITERION = nn.CrossEntropyLoss()
    GLOBAL_OPTIMIZER = torch.optim.Adam(GLOBAL_MODEL.parameters(), lr=0.0001)
    GLOBAL_EPOCH = 0

    if args is not None and getattr(args, "cycle_lr", False):
        step_size = getattr(args, "cycle_step_size", None)
        if step_size is None:
            step_size = max(1, len(train_loader))
        GLOBAL_SCHEDULER = torch.optim.lr_scheduler.CyclicLR(
            GLOBAL_OPTIMIZER,
            base_lr=getattr(args, "cycle_base_lr", 1e-5),
            max_lr=getattr(args, "cycle_max_lr", 1e-4),
            step_size_up=step_size,
            mode="triangular2",
            cycle_momentum=False,
        )
        info(
            f"CyclicLR aktywny: base_lr={getattr(args, 'cycle_base_lr', 1e-5)} "
            f"max_lr={getattr(args, 'cycle_max_lr', 1e-4)} step_size_up={step_size}"
        )
    else:
        GLOBAL_SCHEDULER = None

    if model_path and os.path.exists(model_path):
        info(f"Wczytywanie modelu z: {model_path}")
        checkpoint = torch.load(model_path, map_location=GLOBAL_DEVICE)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        GLOBAL_MODEL.load_state_dict(state_dict)
        GLOBAL_EPOCH = checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0

    info("8. Rozpoczynanie treningu CNN...")
    target_epoch = GLOBAL_EPOCH + epochs
    training_start = time.time()
    best_val_acc = 0.0

    info("\n" + "=" * 60)
    info("  TRENING CNN (phsf – klasyfikacja znaków)")
    info("=" * 60)
    info(f"  Urządzenie: {GLOBAL_DEVICE}")
    info(f"  Batch size: {batch_size}")
    info(f"  Epoki do wykonania: {epochs}")
    info(f"  Zakres epok: {GLOBAL_EPOCH + 1} -> {target_epoch}")
    info("=" * 60 + "\n")

    for epoch in range(GLOBAL_EPOCH, target_epoch):
        epoch_start = time.time()
        GLOBAL_MODEL.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(GLOBAL_DEVICE)
            labels = labels.to(GLOBAL_DEVICE)

            GLOBAL_OPTIMIZER.zero_grad(set_to_none=True)
            outputs = GLOBAL_MODEL(images)
            loss = GLOBAL_CRITERION(outputs, labels)
            loss.backward()
            GLOBAL_OPTIMIZER.step()
            if GLOBAL_SCHEDULER is not None:
                GLOBAL_SCHEDULER.step()

            running_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if step % 50 == 0:
                if args is not None and getattr(args, "log_lr", False):
                    lr = GLOBAL_OPTIMIZER.param_groups[0]["lr"]
                    info(f"Epoch {epoch+1} Step {step} Loss: {loss.item():.4f} | LR: {lr:.6g}")
                else:
                    info(f"Epoch {epoch+1} Step {step} Loss: {loss.item():.4f}")

        train_acc = correct / total * 100

        GLOBAL_MODEL.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(GLOBAL_DEVICE)
                labels = labels.to(GLOBAL_DEVICE)
                outputs = GLOBAL_MODEL(images)
                loss = GLOBAL_CRITERION(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100
        val_loss /= len(val_loader)
        epoch_time = time.time() - epoch_start
        info(f"Epoch {epoch+1} | train_acc={train_acc:.2f}% | val_acc={val_acc:.2f}% | val_loss={val_loss:.4f} | time={epoch_time:.1f}s")

        GLOBAL_EPOCH = epoch + 1
        GLOBAL_VAL_ACCURACY = val_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            capture_best_model(info, path=effective_save_path)

    total_training_time = time.time() - training_start

    info("\n" + "=" * 60)
    info("  PODSUMOWANIE TRENINGU CNN")
    info("=" * 60)
    info(f"  Zakończona epoka: {GLOBAL_EPOCH}")
    info(f"  Najlepsza dokładność (val): {best_val_acc:.2f}%")
    info(f"  Całkowity czas treningu: {total_training_time:.1f}s")
    info("=" * 60)


def multi_train(
    preset_names: Optional[List[str]] = None,
    epochs: int = 10,
    batch_size: int = 32,
    info=None,
):
    """
    Uruchamia wiele konfiguracji treningowych kolejno po sobie.

    Args:
        preset_names: lista nazw presetów z TRAINING_PRESETS; None lub [] = wszystkie
        epochs: domyślna liczba epok (używana gdy preset nie nadpisuje)
        batch_size: domyślny rozmiar batcha (używany gdy preset nie nadpisuje)
        info: funkcja logowania

    Przykład użycia w CLI:
        python -m ocr.main --multi-train                 # wszystkie presety
        python -m ocr.main --multi-train baseline denoise
    """
    global GLOBAL_MODEL, GLOBAL_EPOCH, GLOBAL_BEST_ACC, GLOBAL_VAL_ACCURACY, BEST_MODEL_STATE, _GLOBAL_MAJOR_VERSION, _GLOBAL_MINOR_VERSION

    if info is None:
        info = print

    if not preset_names:
        configs = list(TRAINING_PRESETS.values())
    else:
        unknown = [n for n in preset_names if n not in TRAINING_PRESETS]
        if unknown:
            info(f"Nieznane presety: {', '.join(unknown)}")
            info(f"Dostępne: {', '.join(TRAINING_PRESETS)}")
            return
        configs = [TRAINING_PRESETS[n] for n in preset_names]

    total = len(configs)
    info("\n" + "=" * 60)
    info(f"  MULTI-TRENING – {total} konfiguracji w kolejce")
    info("=" * 60)
    for i, cfg in enumerate(configs, 1):
        info(f"  {i}. [{cfg.name}] {cfg.description}")
    info("=" * 60 + "\n")

    info("\nŁadowanie datasetu (raz dla wszystkich presetów)...")
    shared_data = load_all_datasets(DATA_ROOT_DIR)
    info(f"  ttData: {len(shared_data)} próbek")
    if os.path.isdir(PHSF_DATA_DIR):
        phsf_data = load_phsf_znaki(PHSF_DATA_DIR)
        info(f"  PHSF znaki: {len(phsf_data)} próbek")
        shared_data = shared_data + phsf_data
    else:
        info(f"  PHSF pominięty (brak katalogu: {PHSF_DATA_DIR})")
    info(f"  Łącznie: {len(shared_data)} próbek\n")

    results = []
    overall_start = time.time()

    for i, config in enumerate(configs, 1):
        info("\n" + "=" * 60)
        info(f"  [{i}/{total}] START: {config.name}")
        info(f"  {config.description}")
        info("=" * 60 + "\n")

        # Resetuj globalny stan modelu – każdy preset zaczyna od zera
        GLOBAL_MODEL = None
        GLOBAL_EPOCH = 0
        GLOBAL_BEST_ACC = 0.0
        GLOBAL_VAL_ACCURACY = 0.0
        BEST_MODEL_STATE = None
        _GLOBAL_MAJOR_VERSION = None
        _GLOBAL_MINOR_VERSION = None

        cfg_epochs = config.epochs if config.epochs is not None else epochs
        cfg_batch_size = config.batch_size if config.batch_size is not None else batch_size
        base_dir = config.model_path or "./models"
        base_name = f"model_{config.name}"

        save_path = get_versioned_model_path(base_dir, base_name)
        run_start = time.time()
        status = "OK"
        try:
            train_model(
                epochs=cfg_epochs,
                batch_size=cfg_batch_size,
                model_path=config.resume_path,
                info=info,
                config=config,
                save_path=save_path,
                preloaded_data=shared_data,
            )
        except Exception as exc:
            status = "BŁĄD"
            info(f"Błąd podczas treningu '{config.name}': {exc}")
            import traceback
            info(traceback.format_exc())

        run_time = time.time() - run_start
        results.append((config.name, status, GLOBAL_VAL_ACCURACY, save_path, run_time))
        info(f"\n  [{i}/{total}] KONIEC: {config.name} | {status} | czas: {run_time:.1f}s\n")

    total_time = time.time() - overall_start
    info("\n" + "=" * 60)
    info("  PODSUMOWANIE MULTI-TRENINGU")
    info("=" * 60)
    info(f"  {'PRESET':<14} {'STATUS':<8} {'DOKŁADNOŚĆ':>12}  ŚCIEŻKA")
    info("  " + "-" * 58)
    for name, status, acc, path, _ in results:
        info(f"  {name:<14} {status:<8} {acc:>10.2f}%  {path}")
    info("=" * 60)
    info(f"  Łączny czas: {total_time:.1f}s")
    info("=" * 60 + "\n")

def get_versioned_model_path(base_dir, base_name):
    pattern = re.compile(rf"{re.escape(base_name)}_v(\d+)\.pth$")
    
    existing_versions = []

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for f in os.listdir(base_dir):
        match = pattern.match(f)
        if match:
            existing_versions.append(int(match.group(1)))

    if not existing_versions:
        version = 1
    else:
        version = max(existing_versions) + 1

    filename = f"{base_name}_v{version}.pth"
    return os.path.join(base_dir, filename)

def get_preset_names() -> List[str]:
    """Zwraca listę dostępnych nazw presetów treningowych."""
    return list(TRAINING_PRESETS.keys())

    
    
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


def infinite_train(batch_size=32, model_path=None, checkpoint_interval=5, info=None, args=None):
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
    global GLOBAL_MODEL, GLOBAL_OPTIMIZER, GLOBAL_CRITERION, GLOBAL_SCHEDULER
    global GLOBAL_DEVICE, GLOBAL_EPOCH, GLOBAL_BEST_ACC, GLOBAL_CLASS_NAMES
    global TRAINING_PAUSED, TRAINING_STOP
    
    if info is None:
        info = print
    
    # Reset flag
    TRAINING_PAUSED = False
    TRAINING_STOP = False
    GLOBAL_BEST_ACC = float('inf')

    signal.signal(signal.SIGINT, infinite_handler)

    try:
        info("1. Ładowanie datasetu...")
        
        info("2. Ładowanie transformacji obrazów...")
        train_transform = get_train_transform()
        
        info("3. Ładowanie datasetu OCR...")
        data = load_all_datasets(DATA_ROOT_DIR)
        info(f"   ttData: {len(data)} próbek")

        if os.path.isdir(PHSF_DATA_DIR):
            phsf_data = load_phsf_znaki(PHSF_DATA_DIR)
            info(f"   PHSF znaki: {len(phsf_data)} próbek")
            data = data + phsf_data
        else:
            info(f"   PHSF pominięty (brak katalogu: {PHSF_DATA_DIR})")

        dataset = OCRDataset(
            json_data=data,
            images_dir=None,
            transform=get_train_transform()
        )

        info(f"4. Dataset załadowany: {len(dataset)} obrazów łącznie")
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        info(f"5. Splitting: {train_size} trening, {val_size} walidacja")
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        info("6. Tworzenie DataLoader...")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        info("7. Inicjalizacja lub ładowanie modelu...")
        # init albo load
        if GLOBAL_MODEL is None:
            init_or_load_model(len(CHARS) + 1, model_path, info)

        if args is not None and getattr(args, "cycle_lr", False):
            step_size = getattr(args, "cycle_step_size", None)
            if step_size is None:
                step_size = max(1, len(train_loader))
            GLOBAL_SCHEDULER = torch.optim.lr_scheduler.CyclicLR(
                GLOBAL_OPTIMIZER,
                base_lr=getattr(args, "cycle_base_lr", 1e-5),
                max_lr=getattr(args, "cycle_max_lr", 1e-4),
                step_size_up=step_size,
                mode="triangular2",
                cycle_momentum=False,
            )
            info(
                f"CyclicLR aktywny: base_lr={getattr(args, 'cycle_base_lr', 1e-5)} "
                f"max_lr={getattr(args, 'cycle_max_lr', 1e-4)} step_size_up={step_size}"
            )
        else:
            GLOBAL_SCHEDULER = None
        

        info("8. Rozpoczynanie treningu...")
        info("\n" + "="*60)
        info("  NIESKOŃCZONY TRENING OCR")
        info("="*60)
        info(f"  Urządzenie: {GLOBAL_DEVICE}")
        info(f"  Batch size: {batch_size}")
        info(f"  Klasy: {len(CHARS) + 1}")
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
            
            for step, (images, targets, target_lengths) in enumerate(train_loader, start=1):
                # Sprawdź czy pauza w trakcie epoki
                if TRAINING_PAUSED:
                    break
                    
                images = images.to(GLOBAL_DEVICE)
                targets = targets.to(GLOBAL_DEVICE)
                target_lengths = target_lengths.to(GLOBAL_DEVICE)
                
                GLOBAL_OPTIMIZER.zero_grad(set_to_none=True)
                outputs = GLOBAL_MODEL(images)
                log_probs = outputs.log_softmax(2)
                
                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=outputs.size(0),
                    dtype=torch.long,
                    device=GLOBAL_DEVICE,
                )

                loss = GLOBAL_CRITERION(
                    log_probs,
                    targets,
                    input_lengths,
                    target_lengths
                )
                loss.backward()
                GLOBAL_OPTIMIZER.step()
                if GLOBAL_SCHEDULER is not None:
                    GLOBAL_SCHEDULER.step()
                
                running_loss += loss.item()
                
                # log co 50 kroków
                if step % 50 == 0 or step == total_steps:
                    elapsed = datetime.now() - start_time
                    if args is not None and getattr(args, "log_lr", False):
                        lr = GLOBAL_OPTIMIZER.param_groups[0]["lr"]
                        info(f"Epoch [{epoch+1}] Step [{step}/{total_steps}] "
                             f"Loss: {loss.item():.4f} | LR: {lr:.6g} | Czas: {elapsed}")
                    else:
                        info(f"Epoch [{epoch+1}] Step [{step}/{total_steps}] "
                             f"Loss: {loss.item():.4f} | Czas: {elapsed}")

                # Zwolnij referencje po każdym kroku dla stabilności długiego treningu.
                del outputs, loss, images, targets, target_lengths, log_probs, input_lengths
                
            # Jeśli pauza podczas kroku - wróć do początku pętli
            if TRAINING_PAUSED:
                continue
            
            # walidacja
            GLOBAL_MODEL.eval()
            val_loss = 0.0
            
            with torch.inference_mode():
                for images, targets, target_lengths in val_loader:
                    images = images.to(GLOBAL_DEVICE)
                    targets = targets.to(GLOBAL_DEVICE)
                    target_lengths = target_lengths.to(GLOBAL_DEVICE)
                    
                    outputs = GLOBAL_MODEL(images)
                    log_probs = outputs.log_softmax(2)
                    
                    input_lengths = torch.full(
                        size=(images.size(0),),
                        fill_value=outputs.size(0),
                        dtype=torch.long,
                        device=GLOBAL_DEVICE,
                    )

                    loss = GLOBAL_CRITERION(
                        log_probs,
                        targets,
                        input_lengths,
                        target_lengths
                    )

                    val_loss += loss.item()
                    del outputs, log_probs, input_lengths, images, targets, target_lengths
            
            val_loss /= len(val_loader)

            epoch_time = time.time() - epoch_start
            avg_loss = running_loss / total_steps

            info(f"\n>>> Epoch [{epoch+1}] zakończona")
            info(f"    Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Czas: {epoch_time:.1f}s")

            GLOBAL_EPOCH = epoch + 1
            epoch = GLOBAL_EPOCH

            # Zapisz najlepszy model jeśli poprawa (mniejszy loss)
            if val_loss < GLOBAL_BEST_ACC:  # GLOBAL_BEST_ACC przechowuje teraz best_loss
                old_best = GLOBAL_BEST_ACC
                GLOBAL_BEST_ACC = val_loss
                capture_best_model()
                info(f"    [NEW BEST] Poprawa loss: {old_best:.4f} -> {val_loss:.4f}")

            # Okresowy checkpoint
            if epoch % checkpoint_interval == 0:
                save_model(CHECKPOINT_PATH)
                info(f"    [CHECKPOINT] Zapisano checkpoint (epoka {epoch})")

            info("")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


        
        # Końcowa dokładność znakowa po zakończeniu pętli
        info("\nObliczanie końcowej dokładności znakowej...")
        final_acc = _compute_val_char_accuracy(GLOBAL_MODEL, val_loader, GLOBAL_DEVICE)
        GLOBAL_VAL_ACCURACY = final_acc

        # Podsumowanie końcowe
        total_time = datetime.now() - start_time
        info("\n" + "="*60)
        info("  TRENING ZAKOŃCZONY")
        info("="*60)
        info(f"  Całkowity czas: {total_time}")
        info(f"  Epoki: {GLOBAL_EPOCH}")
        info(f"  Najlepszy loss: {GLOBAL_BEST_ACC:.4f}")
        info(f"  Dokładność znakowa (val): {GLOBAL_VAL_ACCURACY:.2f}%")
        info("="*60)

    except Exception as e:
        info(f"\nBłąd podczas treningu: {e}")
        import traceback
        info(traceback.format_exc())
    finally:
        signal.signal(signal.SIGINT, handler)
