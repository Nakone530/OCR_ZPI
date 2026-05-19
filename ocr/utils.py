"""
Narzędzia pomocnicze:
  - transformacje obrazów
  - odszumianie filtrem bilateralnym (OpenCV)
  - ładowanie obrazów (PNG/JPG/PDF) z opcjonalnym odszumianiem
  - zapis obrazu do folderu z dzisiejszą datą
"""

import os
from datetime import date
from pathlib import Path
import random
from difflib import SequenceMatcher
import json
try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None
import numpy as np
from PIL import Image
try:
    from pdf2image import convert_from_path  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    convert_from_path = None
try:
    import fitz  # type: ignore  # PyMuPDF
except ModuleNotFoundError:  # pragma: no cover
    fitz = None
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F
import csv
from datetime import datetime
from typing import Any
from datetime import date

from pathlib import Path
import matplotlib.pyplot as plt
import itertools
from .config import IMAGE_SIZE, MEAN, STD, CHARS, AuxCHARS, MODEL_PATH, NUM_CLASSES, char2idx, idx2char, VERSION_RE, ÐICT_PATH, DATA_ROOT_DIR
from .model import MainModel, AuxModel
from . import info

def load_dictionary(json_path: str = ÐICT_PATH) -> list[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON musi zawierać listę stringów")

    return [str(word) for word in data]


_TRAINING_TRIGRAMS_CACHE: list[tuple[str, str, str]] | None = None


def _load_training_trigrams(root_dir: str = DATA_ROOT_DIR) -> list[tuple[str, str, str]]:
    """Load (prev, middle, next) trigrams from training boxes.jsonl files."""
    global _TRAINING_TRIGRAMS_CACHE
    if _TRAINING_TRIGRAMS_CACHE is not None:
        return _TRAINING_TRIGRAMS_CACHE

    trigrams: list[tuple[str, str, str]] = []
    dataset = load_all_datasets(root_dir)

    if not dataset:
        _TRAINING_TRIGRAMS_CACHE = trigrams
        return trigrams

    dataset.sort(key=lambda item: (item.get("source_image", ""), item.get("id", 0)))

    prev_row = None
    curr_row = None
    for row in dataset:
        text = (row.get("text") or "").strip()
        if not text:
            prev_row = curr_row
            curr_row = row
            continue

        if prev_row is not None and curr_row is not None:
            prev_text = (prev_row.get("text") or "").strip()
            curr_text = (curr_row.get("text") or "").strip()
            if prev_text and curr_text:
                trigrams.append((prev_text, curr_text, text))

        prev_row = curr_row
        curr_row = row

    _TRAINING_TRIGRAMS_CACHE = trigrams
    return trigrams


def suggest_word_between_from_training(
    prev_word: str,
    next_word: str,
    threshold: float = 0.0,
    root_dir: str = DATA_ROOT_DIR,
) -> str:
    """
    Suggest a word between two words using training data order.

    Uses trigrams from boxes.jsonl where ids follow reading order.
    Returns the most frequent middle word for exact (prev, next).
    If no exact matches, uses similarity to find the closest trigram pair.
    Returns empty string if best score is below threshold.
    """
    prev_word = (prev_word or "").strip()
    next_word = (next_word or "").strip()

    if not prev_word or not next_word:
        return ""

    trigrams = _load_training_trigrams(root_dir)
    if not trigrams:
        return ""

    prev_lower = prev_word.lower()
    next_lower = next_word.lower()

    counts: dict[str, int] = {}
    for prev_text, mid_text, next_text in trigrams:
        if prev_text.lower() == prev_lower and next_text.lower() == next_lower:
            counts[mid_text] = counts.get(mid_text, 0) + 1

    if counts:
        best_word = max(counts.items(), key=lambda item: item[1])[0]
        return best_word

    best_match = ""
    best_score = 0.0

    for prev_text, mid_text, next_text in trigrams:
        score_prev = SequenceMatcher(None, prev_lower, prev_text.lower()).ratio()
        score_next = SequenceMatcher(None, next_lower, next_text.lower()).ratio()
        score = (score_prev + score_next) / 2.0

        if score > best_score:
            best_score = score
            best_match = mid_text

    if best_score >= threshold:
        return best_match

    return ""
# ── Transformacje ──────────────────────────────────────────────────────────────

def aux_transform():
    """
    pipeline transformacji obrazu do inferencji (bez augmentacji danych).
    
    Pipeline zawiera:
      - Konwersja do skali szarości (1 kanał)
      - Zmiana rozmiaru do IMAGE_SIZE x IMAGE_SIZE
      - Konwersja do tensora PyTorch
      - Normalizacja wartości pikseli (mean=0.5, std=0.5)
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((24, 24)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

def base_transform():
    """
    pipeline transformacji obrazu do inferencji (bez augmentacji danych).
    
    Pipeline zawiera:
      - Konwersja do skali szarości (1 kanał)
      - Zmiana rozmiaru do IMAGE_SIZE x IMAGE_SIZE
      - Konwersja do tensora PyTorch
      - Normalizacja wartości pikseli (mean=0.5, std=0.5)
    """
    return [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]

def get_inf_transform(args) -> transforms.Compose:
    """
    Zwraca pipeline transformacji obrazu do inferencji (bez augmentacji danych).
    
    Pipeline zawiera:
      - Konwersja do skali szarości (1 kanał)
      - Zmiana rozmiaru do IMAGE_SIZE x IMAGE_SIZE
      - Konwersja do tensora PyTorch
      - Normalizacja wartości pikseli (mean=0.5, std=0.5)
    
    Zwraca:
        transforms.Compose: Złożona transformacja gotowa do użycia
                            na obrazach PIL podczas predykcji.
    
    Przykład:
        >>> transform = get_transform()
        >>> tensor = transform(pil_image)
    """
    pack = [ResizeWithAspect(),
            RandomOtsu(Otsu()),
            TightCrop(),]
    if getattr(args, "denoise", False):
        pack.append(trans_denoise_bil())

    pack.extend(base_transform())
    return transforms.Compose(pack)


def get_train_transform(args=None, denoise_prob: float = 0.3, max_padding: int = 20) -> transforms.Compose:
    """
    Zwraca pipeline transformacji obrazu do trenowania modelu (z augmentacją danych).

    Args:
        args: nieużywany, zachowany dla kompatybilności wstecznej
        denoise_prob: prawdopodobieństwo losowego odszumiania (0.0 = wyłączone, 1.0 = zawsze)
        max_padding: maksymalny padding w pikselach dla RandomPadding
    """
    pack = [
        transforms.RandomRotation(5),
        RandomDenoise(p=denoise_prob),
        RandomPadding(max_pad=max_padding),
        ResizeWithAspect(),
        RandomOtsu(Otsu()),
        TightCrop(),
    ]
    pack.extend(base_transform())
    return transforms.Compose(pack)


class RandomPadding:
    def __init__(self, max_pad=20):
        self.max_pad = max_pad

    def __call__(self, img):
        left = random.randint(0, self.max_pad)
        top = random.randint(0, self.max_pad)
        right = random.randint(0, self.max_pad)
        bottom = random.randint(0, self.max_pad)
        return F.pad(img, (left, top, right, bottom), fill=255)

class RandomDenoise:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return trans_denoise_bil()(img)
        return img


class ResizeWithAspect:
    def __init__(self, size=(32, 128), fill=255):
        self.h, self.w = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = min(self.w / w, self.h / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = F.resize(img, (new_h, new_w))

        pad_w = self.w - new_w
        pad_h = self.h - new_h

        left = pad_w // 2
        top = pad_h // 2

        return F.pad(img, (left, top, pad_w - left, pad_h - top), fill=self.fill)

class Otsu:
    def __call__(self, img):
        arr = np.array(img.convert("L"))
        _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(th)


class RandomOtsu:
    def __init__(self, otsu, p=0.1):
        self.otsu = otsu
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return self.otsu(img)
        return img

class TightCrop:
    def __call__(self, img):
        # zakładamy PIL Image lub tensor -> konwersja do numpy
        img_np = np.array(img)

        if img_np.ndim == 3:  # RGB → grayscale
            img_np = img_np.mean(axis=2)

        # maska nie-tła (próg można dostosować)
        mask = img_np < 250  # dla jasnego tła

        coords = np.argwhere(mask)

        if coords.size == 0:
            return img  # fallback

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        cropped = img.crop((x0, y0, x1, y1))
        return cropped
    
def preprocess_letter(img: np.ndarray) -> np.ndarray:
    """
    Przetwarza obraz pojedynczej litery przed klasyfikacją.
    
    Operacje wykonywane na obrazie:
      1. Dodaje biały padding (10px) wokół obrazu
      2. Tworzy kwadratowy canvas o rozmiarze max(wysokość, szerokość)
      3. Centruje literę na canvasie z białym tłem
      4. Skaluje wynikowy obraz do rozmiaru 24x24 pikseli
    
    Argumenty:
        img (np.ndarray): Obraz litery w skali szarości jako tablica numpy.
                          Oczekiwany format: (wysokość, szerokość), dtype uint8.
    
    Zwraca:
        np.ndarray: Przetworzony obraz litery o wymiarach 28x28 pikseli.
    
    Przykład:
        >>> letter = preprocess_letter(letter_array)
        >>> letter.shape
        (24, 24)
    """
    pad = 10
    img = np.pad(img, pad, mode='constant', constant_values=255)

    h, w = img.shape
    size = max(h, w)
    
    new_img = np.full((size, size), 255, dtype=img.dtype)
    
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    
    new_img[y_offset:y_offset+h, x_offset:x_offset+w] = img
    
    new_img = cv2.resize(new_img, (24, 24))
    return new_img

# ── Odszumianie ────────────────────────────────────────────────────────────────

def denoise_pil(img_pil: Image.Image) -> Image.Image:
    """
    Odszumia obraz PIL filtrem bilateralnym.

    Argumenty:
        img_pil (Image.Image): Obraz wejściowy w formacie PIL.
    
    Zwraca:
        Image.Image: Odszumiony obraz w formacie PIL.
    
    Przykład:
        >>> denoised = denoise_pil(img)
    """
    if cv2 is None:
        raise ModuleNotFoundError(
            "Brak modułu 'cv2'. Zainstaluj: pip install opencv-python "
            "(wymagane tylko dla opcji --denoise)."
        )
    rgb = img_pil.convert("RGB")
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    out = cv2.bilateralFilter(bgr, 9, 75, 75)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

class trans_denoise_bil:
    def __init__(self, d=9, sigma_color=75, sigma_space=75):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, img: Image.Image):
        rgb = img.convert("RGB")
        arr = np.array(rgb)

        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        out = cv2.bilateralFilter(bgr, self.d, self.sigma_color, self.sigma_space)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        return Image.fromarray(out)
# ── Ładowanie obrazu ───────────────────────────────────────────────────────────

def load_image(image_path: str, mode: str = "L") -> Image.Image:
    """
    Ładuje obraz z pliku i konwertuje do wybranego trybu kolorów.
    
    Obsługuje formaty PNG, JPG/JPEG oraz PDF (pierwsza strona).
    Dla plików PDF wykorzystuje bibliotekę pdf2image z DPI=300.
    
    Argumenty:
        image_path (str): Ścieżka do pliku obrazu (PNG, JPG, JPEG, PDF).
        mode (str, opcjonalnie): Tryb kolorów PIL. Domyślnie "L" (skala szarości).
            Dostępne tryby: "L" (grayscale), "RGB", "RGBA", "1" (binarny).
    
    Zwraca:
        Image.Image: Załadowany obraz w formacie PIL w wybranym trybie.
    
    Przykład:
        >>> img = load_image("letter.png", mode="L")
        >>> img_rgb = load_image("document.pdf", mode="RGB")
    """
    suffix = Path(image_path).suffix.lower()
    if suffix == ".pdf":
        # Prefer pdf2image when available (higher DPI control),
        # but on Windows it requires Poppler binaries in PATH.
        if convert_from_path is not None:
            try:
                img = convert_from_path(image_path, dpi=300)[0]
                return img.convert(mode)
            except Exception:
                # fall back to PyMuPDF if Poppler is missing or pdf2image fails
                pass

        if fitz is None:
            raise ModuleNotFoundError(
                "Obsługa PDF wymaga dodatkowych zależności.\n"
                "- Opcja A (najprostsza): zainstaluj PyMuPDF: pip install pymupdf\n"
                "- Opcja B: użyj pdf2image + zainstaluj Poppler i dodaj do PATH.\n"
                "Błąd wygląda na brak Popplera (pdfinfo/pdftoppm) w systemie."
            )

        # PyMuPDF fallback: render first page to a bitmap
        doc = fitz.open(image_path)
        try:
            page = doc.load_page(0)
            # ~300 DPI equivalent: scale 300/72
            zoom = 300.0 / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        finally:
            doc.close()
    else:
        img = Image.open(image_path)

    return img.convert(mode)


def load_and_optionally_denoise(image_path: str, args, mode: str = "L") -> Image.Image:
    """
    Ładuje obraz i opcjonalnie stosuje odszumianie na podstawie argumentów CLI.
    
    Funkcja łączy ładowanie obrazu z opcjonalnym preprocessingiem (odszumianiem).
    Jeśli args.denoise jest True, stosowany jest filtr bilateralny.
    
    Argumenty:
        image_path (str): Ścieżka do pliku obrazu (PNG, JPG, PDF).
        args: Obiekt argparse.Namespace z parametrami. Oczekiwane atrybuty:
            - denoise (bool): Czy stosować odszumianie
        mode (str, opcjonalnie): Tryb kolorów wyjściowych. Domyślnie "L".
            Używaj "RGB" dla klasyfikacji lub "L" dla segmentacji.
    
    Zwraca:
        Image.Image: Załadowany (i opcjonalnie odszumiony) obraz w formacie PIL.
    
    Przykład:
        >>> img = load_and_optionally_denoise("document.png", args, mode="L")
    """
    suffix = Path(image_path).suffix.lower()
    if suffix == ".pdf":
        img = convert_from_path(image_path, dpi=300)[0]
    else:
        img = Image.open(image_path)

    if getattr(args, "denoise", False):
        img = denoise_pil(img)

    return img.convert(mode)

# ── ładowanie datasetu ────────────────────────────────────────────────────

def load_all_datasets(root_dir):
    all_data = []

    for author in os.listdir(root_dir):
        author_path = os.path.join(root_dir, author)

        if not os.path.isdir(author_path):
            continue

        json_path = os.path.join(author_path, "boxes.jsonl")

        if not os.path.exists(json_path):
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                # KLUCZOWE: dodaj pełną ścieżkę do obrazu
                item["image_path"] = os.path.join(author_path, item["crop_file"])

                all_data.append(item)

    return all_data

# ── Zapis do folderu z datą ────────────────────────────────────────────────────

def get_today_folder() -> str:
    """
    Zwraca ścieżkę do folderu z dzisiejszą datą, tworząc go jeśli nie istnieje.
    
    Folder jest tworzony w bieżącym katalogu roboczym z nazwą w formacie
    YYYY-MM-DD (np. "2024-01-15").
    
    Argumenty:
        Brak argumentów.
    
    Zwraca:
        str: Ścieżka do folderu z dzisiejszą datą (np. "./2024-01-15").
    
    Przykład:
        >>> folder = get_today_folder()
        >>> print(folder)
        './2024-01-15'
    """
    folder = os.path.join("./outputs/", date.today().strftime("%Y-%m-%d"))
    os.makedirs(folder, exist_ok=True)
    return folder

def save_image_to_today_folder(image_path: str, info=None) -> str:
    """
    Kopiuje obraz do folderu z dzisiejszą datą z automatyczną numeracją.
    
    Obrazy są zapisywane jako pliki PNG z kolejnymi numerami (1.png, 2.png, ...).
    Funkcja automatycznie wykrywa istniejące pliki i nadaje następny numer.
    
    Argumenty:
        image_path (str): Ścieżka do obrazu źródłowego do skopiowania.
        info (callable, opcjonalnie): Funkcja do logowania. Domyślnie print.
    
    Zwraca:
        str: Ścieżka do zapisanego pliku (np. "./2024-01-15/3.png").
    
    Efekty uboczne:
        - Tworzy folder z dzisiejszą datą (jeśli nie istnieje)
        - Zapisuje obraz jako plik PNG
        - Wypisuje komunikat o zapisie na konsolę
    
    Przykład:
        >>> path = save_image_to_today_folder("input/letter.jpg")
        Zapisano zdjęcie jako: ./2024-01-15/1.png
    """
    if info is None:
        info = print
    
    folder = get_today_folder()
    existing = [
        int(os.path.splitext(f)[0])
        for f in os.listdir(folder)
        if f.endswith(".png") and os.path.splitext(f)[0].isdigit()
    ]
    dest = os.path.join(folder, f"{max(existing, default=0) + 1}.png")
    load_image(image_path, mode="L").save(dest, format="PNG")
    info(f"Zapisano zdjęcie jako: {dest}")
    return dest


def save_image_to_temp_folder(image: Any, order: str) -> str:
    """
    Zapisuje obraz do folderu tymczasowego z timestampem w nazwie.
    
    Funkcja tworzy folder ./temp/ (jeśli nie istnieje) i zapisuje obraz
    z nazwą zawierającą timestamp i podany suffix (order).
    
    Argumenty:
        image: Obraz do zapisania. Może być:
            - PIL.Image.Image: Obraz PIL
            - np.ndarray: Tablica numpy (zostanie skonwertowana do PIL)
        order (str): Suffix dodawany do nazwy pliku przed rozszerzeniem.
                     Używany do oznaczenia kolejności lub typu obrazu.
    
    Zwraca:
        str: Ścieżka do zapisanego pliku (np. "./temp/20240115_143022_1.png").
    
    Efekty uboczne:
        - Tworzy folder ./temp/ jeśli nie istnieje
        - Zapisuje obraz jako plik PNG
    
    Przykład:
        >>> path = save_image_to_temp_folder(pil_image, "_letter1")
        >>> path = save_image_to_temp_folder(numpy_array, "_segment")
    """
    os.makedirs("./temp", exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + order + ".png"
    filepath = os.path.join("./temp", filename)

    if isinstance(image, Image.Image):
        image.save(filepath)
    else:
        # assume numpy array
        img = Image.fromarray(image)
        img.save(filepath)

    return filepath

# ── Ładowanie modelu ───────────────────────────────────────────────────────────

ACTIVE_CHARS = list(CHARS)


def get_active_chars() -> list[str]:
    """Zwraca aktualne mapowanie indeks->znak używane przez model."""
    return list(ACTIVE_CHARS)


def _set_active_chars(checkpoint: object | None = None) -> None:
    """Ustawia mapowanie indeks->znak na podstawie checkpointa lub domyślnej konfiguracji."""
    global ACTIVE_CHARS


    ACTIVE_CHARS = list(CHARS)


def _label_for_idx(idx: int) -> str:
    if 0 <= idx < len(ACTIVE_CHARS):
        return ACTIVE_CHARS[idx]
    return f"<UNK:{idx}>"

#print(matplotlib.get_backend())

def load_model(model_path: str = MODEL_PATH, device: torch.device = None, info=None, model_type = 1) -> nn.Module:
    if info is None:
        info = print

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(model_path):
        info(f"Wczytywanie modelu z {model_path}...")

        checkpoint = torch.load(model_path, map_location=device)

        _set_active_chars(checkpoint)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            info("Wczytano checkpoint")
        else:
            state_dict = checkpoint
            info("Wczytano state_dict")
        if(model_type == 1):
            model = MainModel(num_classes=len(CHARS) + 1)
        else:
            model = AuxModel(num_classes=len(AuxCHARS))
        model.load_state_dict(state_dict)
        info("Model wczytany!")

    else:
        info(f"UWAGA: Nie znaleziono modelu {model_path}")
        if(model_type == 1):
            model = MainModel(num_classes=len(CHARS) + 1)
        else:
            model = AuxModel(num_classes=len(AuxCHARS))

    model.to(device)
    model.eval()

    return model


#------Zarządzanie transkrypcjami------

def load_transcription(path):
    mapping = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue

            key, text = parts

            # usuń rozszerzenie jeśli jest
            key = key.replace(".png", "")

            mapping[key] = text

    return mapping

#------Zarządzanie modelami------------
def list_models(models_dir: str, version: str | None = None):
    models = [
        name for name in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, name))
    ]

    if version:
        models = [
            m for m in models
            if m.startswith(f"v{version}")
        ]

    models.sort()
    return models

def select_version():
    version = input("Wybierz wersję (ENTER = wszystkie): ").strip()
    return version if version else None

def select_models(models):
    print("\nDostępne modele:")
    for i, m in enumerate(models):
        print(f"[{i}] {m}")

    default_idx = int(input("\nWybierz DEFAULT model (index): "))
    selected = input("Wybierz ensemble (np. 0,1,2) lub ENTER = wszystkie: ")

    if selected.strip() == "":
        ensemble = models
    else:
        ensemble = [models[int(i)] for i in selected.split(",")]

    default_model = models[default_idx]

    return default_model, ensemble



def load_models(models_dir, selected_models, device, info):
    loaded = {}

    for m in selected_models:
        path = os.path.join(models_dir, m, "model.pth")
        info(f"Ładowanie modelu: {m}")
        loaded[m] = load_model(path, device, info)

    return loaded

from collections import Counter

def aggregate(results, default_model):
    votes = Counter()
    confidence_sum = {}

    for model_name, res in results.items():
        text = res["text"]

        conf = res["confidence"]
        if hasattr(conf, "item"):  # torch / numpy
            conf = conf.item()

        votes[text] += 1
        confidence_sum[text] = confidence_sum.get(text, 0) + conf

    top_text, top_count = votes.most_common(1)[0]

    if top_count >= 2:
        return top_text

    return results[default_model]["text"]

def generate_model_ensembles(models, min_size=3, max_size=5, mode=1):
    if max_size is None:
        max_size = len(models)

    max_size = min(max_size, len(models))

    if mode == 1 :
        for r in range(min_size, max_size + 1):
            for combo in itertools.combinations(models, r):
                yield combo
    elif mode == 2 :
        for r in range(min_size, max_size + 1):
            for combo in itertools.permutations(models, r):
                yield combo

#---cache----------
def parse_version(name: str):
    m = VERSION_RE.match(name)
    if not m:
        return None
    major = int(m.group(1))
    minor = int(m.group(2) or 0)
    return (major, minor)


def version_str(v):
    major, minor = v
    return f"v{major}" if minor == 0 else f"v{major}.{minor}"


def resolve_cache_path(models_dir="./models", cache_dir="./cache"):
    versions = []

    for d in os.listdir(models_dir):
        full = os.path.join(models_dir, d)
        if os.path.isdir(full):
            v = parse_version(d)
            if v:
                versions.append(v)

    if not versions:
        raise ValueError("Brak poprawnych wersji w ./models")

    versions.sort()

    v_min = versions[0]
    v_max = versions[-1]

    base = f"{version_str(v_min)}_{version_str(v_max)}.json"
    path = os.path.join(cache_dir, base)

    if not os.path.exists(path):
        return path

    i = 1
    while True:
        suffix = f"_a{i:02d}"
        new_path = os.path.join(cache_dir, base.replace(".json", f"{suffix}.json"))
        if not os.path.exists(new_path):
            return new_path
        i += 1




def load_cache(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist() if obj.ndim > 0 else obj.item()

    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]

    return obj


def save_results_csv(rows, path="results.csv"):
    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "file",
                "ensemble",
                "text",
                "confidence",
                "accuracy"
            ]
        )

        if not file_exists:
            writer.writeheader()

        for r in rows:

            writer.writerow({
                "key": r["key"],
                "file": r["file"],
                "ensemble": r["ensemble"],
                "text": r["text"],
                "confidence": r["confidence"],
                "accuracy": r["accuracy"]
            })
#--- Autokorekta------
def DictCorrect(
    text: str,
    threshold: float = 0.8,
) -> str:
    """
    Szuka najbardziej podobnego słowa w słowniku.
    
    Jeśli podobieństwo >= threshold:
        zwraca słowo ze słownika
    W przeciwnym razie:
        zwraca oryginalny tekst
    """
    dictionary = load_dictionary()
    text = text.strip()

    if not text:
        return text

    best_match = None
    best_score = 0.0

    text_lower = text.lower()

    for word in dictionary:
        score = SequenceMatcher(
            None,
            text_lower,
            word.lower()
        ).ratio()

        if score > best_score:
            best_score = score
            best_match = word

    if best_score >= threshold:
        return best_match

    return text


def suggest_word_between(
    prev_word: str,
    next_word: str,
    threshold: float = 0.8,
    candidates: list[str] | None = None,
) -> str:
    """
    Suggest a word that best fits between two words.

    Scores each candidate by similarity to prev and next word and returns the
    best match if it meets the threshold. Returns an empty string otherwise.
    """
    prev_word = (prev_word or "").strip()
    next_word = (next_word or "").strip()

    if not prev_word or not next_word:
        return ""

    dictionary = candidates or load_dictionary()

    best_match = ""
    best_score = 0.0

    prev_lower = prev_word.lower()
    next_lower = next_word.lower()

    for word in dictionary:
        word_lower = word.lower()
        score_prev = SequenceMatcher(None, prev_lower, word_lower).ratio()
        score_next = SequenceMatcher(None, word_lower, next_lower).ratio()
        score = (score_prev + score_next) / 2.0

        if score > best_score:
            best_score = score
            best_match = word

    if best_score >= threshold:
        return best_match

    return ""
