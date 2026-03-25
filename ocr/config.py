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
CHECKPOINT_PATH = "./checkpoint.pth"

# ── Obraz ─────────────────────────────────────────────────────────────────────
IMAGE_SIZE = 28
MEAN = [0.5, 0.5, 0.5]   # ImageNet mean
STD  = [0.5, 0.5, 0.5]   # ImageNet std

# ── Klasy (52: A-Z + a-z - alfabet angielski) ────────────────────────────────
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
NUM_CLASSES = len(CHARS)
