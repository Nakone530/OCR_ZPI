"""
Centralna konfiguracja projektu OCR.
Wszystkie stałe używane przez pozostałe moduły.
"""

import os

# ── Dataset ──────────────────────────────────────────────────────────────────
DATA_URL = "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz"
DATA_DIR = "./data"
ARCHIVE_PATH = os.path.join(DATA_DIR, "EnglishFnt.tgz")
EXTRACTED_DIR = os.path.join(DATA_DIR, "English", "Fnt")

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "./model_ocr.pth"

# ── Obraz ─────────────────────────────────────────────────────────────────────
IMAGE_SIZE = 28
MEAN = [0.485, 0.456, 0.406]   # ImageNet mean
STD  = [0.229, 0.224, 0.225]   # ImageNet std

# ── Klasy (26: A-Z) ──────────────────────────────────────────────────────────
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(CHARS)
