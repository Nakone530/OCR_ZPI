"""
Centralna konfiguracja projektu OCR.
Wszystkie stałe używane przez pozostałe moduły.

Moduł definiuje:
    - Ścieżki do datasetu i archiwum
    - Ścieżki do plików modelu
    - Parametry przetwarzania obrazów
    - Zestaw znaków do rozpoznawania

Attributes:
    DATA_DIR (str): Lokalny katalog na dane.
    MODEL_PATH (str): Ścieżka do zapisanego modelu (.pth).
    CHECKPOINT_PATH (str): Ścieżka do checkpointu treningu.
    IMAGE_SIZE (int): Rozmiar obrazu wejściowego (28x28 pikseli).
    MEAN (list): Wartości średnie do normalizacji (ImageNet).
    STD (list): Odchylenia standardowe do normalizacji (ImageNet).
    CHARS (str): Alfabet znaków do rozpoznawania (A-Z).
    NUM_CLASSES (int): Liczba klas (26 liter alfabetu).
"""

import os
import re
# ── Dataset ──────────────────────────────────────────────────────────────────

# Katalog lokalny na przechowywanie danych
DATA_DIR = "./data"
DATA_ROOT_DIR = "./ttData"
PHSF_DATA_DIR = "./data/phsf"

IMAGES_DIR = "./ttData"

ÐICT_PATH = "./data/dictionary.json"
# ── Model ─────────────────────────────────────────────────────────────────────
# Ścieżka do folderu modeli
MODEL_DIR = "./models"
# Ścieżka do zapisanego wytrenowanego modelu
MODEL_PATH = os.path.join(MODEL_DIR, "model_ocr.pth")
OCR_MODEL_PATH = os.path.join(MODEL_DIR, "model_ocr_old.pth")
# Ścieżka do checkpointu (do wznawiania treningu)
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.pth")
# Ścieżka do archiwum starych modeli
MODEL_ARCHIVE_DIR = "./model_archive"

# Liczba ostatnich modeli do zachowania przed czyszczeniem
MODEL_ARCHIVE_KEEP_COUNT = 10

# ── Obraz ─────────────────────────────────────────────────────────────────────
# Rozmiar obrazu wejściowego dla modelu CNN OCR (szerokość i wysokość)
IMAGE_SIZE = 24

# Wartości średnie do normalizacji obrazu (format ImageNet)
MEAN = [0.5, 0.5, 0.5]

# Odchylenia standardowe do normalizacji obrazu (format ImageNet)
STD  = [0.5, 0.5, 0.5]

# ── Klasy ( A-z - alfabet ze znakami polskimi) ──────────────────────────────────────
# Zestaw znaków obsługiwanych przez model (wielkie litery A-z)
CHARS = "ABCDEFGHIJKLMNOPRSTUWYZĄĆĘŁŃÓŚŹŻabcdefghijklmnoprstuwyząćęłńóśźż"
AuxCHARS = "ABCDEFGHIJKLMNOPRSTUWYZabcderghijklmnoprstuwyz"
NUM_CLASSES = len(CHARS)

char2idx = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 = blank!
idx2char = {i + 1: c for i, c in enumerate(CHARS)}

BLANK = 0
#--format nazw modeli
VERSION_RE = re.compile(r"v\.?(\d+)(?:\.(\d+))?$")
