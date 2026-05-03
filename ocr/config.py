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
    ARCHIVE_PATH (str): Ścieżka do pobranego archiwum .tgz.
    EXTRACTED_DIR (str): Ścieżka do rozpakowanych danych.
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

# Ścieżka do pobranego archiwum
ARCHIVE_PATH = os.path.join(DATA_DIR, "EnglishFnt.tgz")

# Ścieżka do rozpakowanych obrazów datasetu
EXTRACTED_DIR = os.path.join(DATA_DIR, "English", "Fnt")

IMAGES_DIR = "./ttData"
# ── Model ─────────────────────────────────────────────────────────────────────
# Ścieżka do zapisanego wytrenowanego modelu
MODEL_PATH = "./models/model_ocr.pth"

# Ścieżka do checkpointu (do wznawiania treningu)
CHECKPOINT_PATH = "./models/checkpoint.pth"

# Ścieżka do archiwum starych modeli
MODEL_ARCHIVE_DIR = "./model_archive"

# Liczba ostatnich modeli do zachowania przed czyszczeniem
MODEL_ARCHIVE_KEEP_COUNT = 10

# ── Obraz ─────────────────────────────────────────────────────────────────────
# Rozmiar obrazu wejściowego dla modelu CNN (szerokość i wysokość)
IMAGE_SIZE = 28

# Wartości średnie do normalizacji obrazu (format ImageNet)
MEAN = [0.5, 0.5, 0.5]

# Odchylenia standardowe do normalizacji obrazu (format ImageNet)
STD  = [0.5, 0.5, 0.5]

# ── Klasy (46: A-z - alfabet bez znaków polskich) ──────────────────────────────────────
# Zestaw znaków obsługiwanych przez model OCR (wielkie litery A-z)
CHARS = "ABCDEFGHIJKLMNOPRSTUWYZĄĆĘŁŃÓŚŹŻabcdefghijklmnoprstuwyząćęłńóśźż"
NUM_CLASSES = len(CHARS)

char2idx = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 = blank!
idx2char = {i + 1: c for i, c in enumerate(CHARS)}

BLANK = 0
#--format nazw modeli
VERSION_RE = re.compile(r"v\.?(\d+)(?:\.(\d+))?$")
