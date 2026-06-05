"""
Narzędzia pomocnicze:
  - transformacje obrazów
  - odszumianie filtrem bilateralnym (OpenCV)
  - ładowanie obrazów (PNG/JPG/PDF) z opcjonalnym odszumianiem
  - zapis obrazu do folderu z dzisiejszą datą
"""

import os
try:
    import pyphen
    PYPHEN_AVAILABLE = True
except ModuleNotFoundError:
    pyphen = None  # type: ignore
    PYPHEN_AVAILABLE = False
import re
import csv
import math
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
from datetime import datetime
from typing import Any
from datetime import date

from pathlib import Path
import matplotlib.pyplot as plt
import itertools
from .config import IMAGE_SIZE, MEAN, STD, CHARS, AuxCHARS, MODEL_PATH, NUM_CLASSES, char2idx, idx2char, VERSION_RE, ÐICT_PATH
from .model import MainModel, AuxModel
from .bbox_annotator import load_boxes_from_annotations, sort_boxes_reading_order, detect_word_boxes_auto, edit_boxes_interactive
from . import info


if PYPHEN_AVAILABLE:
    try:
        dic = pyphen.Pyphen(lang="pl_PL")
    except Exception:
        dic = None
else:
    dic = None

def load_dictionary(json_path: str = ÐICT_PATH) -> list[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON musi zawierać listę stringów")

    return [str(word) for word in data]


def _load_char_folder_map(numeracja_path: str) -> dict:
    mapping = {}
    try:
        with open(numeracja_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)

            for row_num, row in enumerate(reader, start=1):
                left = row[1]
                right = row[0]
                folder_id = left.strip()
                syl = right.strip()
                if folder_id and syl:
                    mapping[syl] = folder_id
    except OSError:
        return {}
    return mapping


def word_to_folder_paths(word: str, data_root: str = "data", info: callable = None) -> list:
    numeracja_path = os.path.join(data_root, "numeracja.csv")
    if not os.path.exists(numeracja_path):
        numeracja_path = os.path.join(data_root, "phsf", "syllables", "numeracja.csv")
    base_dir = os.path.join(data_root, "syllables")
    if not os.path.exists(base_dir):
        base_dir = os.path.join(data_root, "phsf", "syllables")
    char_map = _load_char_folder_map(numeracja_path)

    if os.path.isfile(word) and word.lower().endswith(".txt"):
        with open(word, encoding="utf-8") as handle:
            words = [token for line in handle for token in line.split()]
        out_path = os.path.splitext(word)[0] + "_folders.txt"
        all_results = []
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(f"Foldery znakow z pliku: {word}\n")
            _log(info, f"\nFoldery znakow z pliku: {word}")
            for single_word in words:
                out.write(f"\nSlowo: {single_word}\n")
                _log(info, f"\nSlowo: {single_word}")
                pairs = _word_to_folder_syllables(single_word, char_map, base_dir)
                for char, path in pairs:
                    all_results.append((char, path))
                    if path:
                        out.write(f"{char} -> {path}\n")
                        _log(info, f"{char} -> {path}")
                    else:
                        out.write(f"{char} -> BRAK\n")
                        _log(info, f"{char} -> BRAK")
        _log(info, f"Zapisano wyniki do: {out_path}")
        return all_results

    # pojedynczy wyraz
    _log(info, f"\nFoldery znakow dla slowa: {word}")
    results = _word_to_folder_syllables(word, char_map, base_dir)
    print(results)
    for char, path in results:
        if path:
            _log(info, f"{char} -> {path}")
        else:
            _log(info, f"{char} -> BRAK")
    return results


def _word_to_folder_syllables(word: str, char_map: dict, base_dir: str) -> list:
    results = []
    if dic is not None:
        try:
            syllables = dic.inserted(word).split("-")
        except Exception:
            syllables = [word]
    else:
        syllables = [word]
    for syllab in syllables:
        folder_id = char_map.get(syllab)
        if not folder_id:
            results.append((syllab, None))
            continue
        folder_path = os.path.join(base_dir, folder_id)
        results.append((syllab, folder_path))
    return results


def _log(info: callable, msg: str):
    if info:
        info(msg)


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


def get_cnn_train_transform(denoise_prob: float = 0.3) -> transforms.Compose:
    """
    Transformacja do treningu CNN na pojedynczych znakach (phsf).
    Wyjście: tensor (1, 32, 32) – stały rozmiar kwadratowy.
    """
    return transforms.Compose([
        transforms.RandomRotation(5),
        RandomDenoise(p=denoise_prob),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


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
        
        # Obsługa augmentowanych plików w folderach _aug* obok oryginalnego folderu
        # Zbierz mapę oryginalnych wpisów dla tego autora: base_name -> item
        original_map = {}
        for item in list(all_data):
            # uwzględnij tylko wpisy z tego autora
            img_path = item.get("image_path", "")
            if img_path.startswith(author_path):
                key = os.path.splitext(item.get("crop_file", ""))[0]
                if key:
                    original_map[key] = item

        # Znajdź wszystkie katalogi w root_dir pasujące do author + '_aug'
        for d in os.listdir(root_dir):
            if not d.startswith(author + "_aug"):
                continue
            aug_path = os.path.join(root_dir, d)
            if not os.path.isdir(aug_path):
                continue

            for f in os.listdir(aug_path):
                if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                base_name = os.path.splitext(f)[0]
                if "_aug_" not in base_name:
                    continue
                original_base = base_name[: base_name.rfind("_aug_")]
                orig_item = original_map.get(original_base)
                if not orig_item:
                    continue
                aug_item = orig_item.copy()
                aug_item["image_path"] = os.path.join(aug_path, f)
                aug_item["crop_file"] = f
                all_data.append(aug_item)

    return all_data


def load_phsf_znaki(phsf_dir: str) -> list:
    """
    Ładuje dataset znaków PHSF (znaki/png/0..88).

    Parsuje numeracja.txt, dla każdego folderu zbiera pliki PNG
    i buduje listę {"image_path": ..., "text": znak}.
    Pomija znaki nieobsługiwane przez model (spoza char2idx po lowercase).
    """
    numeracja_path = os.path.join(phsf_dir, "numeracja.txt")
    znaki_dir = os.path.join(phsf_dir, "znaki", "png")

    folder_to_char: dict[int, str] = {}
    with open(numeracja_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"(\d+)\s*=\s*(.+)", line.strip())
            if m:
                folder_to_char[int(m.group(1))] = m.group(2).strip()

    items = []
    skipped_chars: set[str] = set()

    for folder_num, char in sorted(folder_to_char.items()):
        if char.lower() not in char2idx:
            skipped_chars.add(char)
            continue

        folder_path = os.path.join(znaki_dir, str(folder_num))
        if not os.path.isdir(folder_path):
            continue

        for fname in os.listdir(folder_path):
            if fname.lower().endswith(".png"):
                items.append({
                    "image_path": os.path.join(folder_path, fname),
                    "text": char,
                })

    return items


def load_folder8(root_dir: str) -> list:
    """
    Ładuje dataset ze struktury folder8 (podfolderów 1-174).

    Każdy podfolder zawiera obrazy słów i plik labels.txt
    w formacie: 'nazwa_pliku<TAB>słowo' (jedna para na linię).
    Zwraca listę {"image_path": ..., "text": ...} gotową dla OCRDataset.
    """
    items = []
    if not os.path.isdir(root_dir):
        return items

    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        labels_path = os.path.join(subfolder_path, "labels.txt")
        if not os.path.exists(labels_path):
            continue

        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if "\t" not in line:
                    continue
                fname, text = line.split("\t", 1)
                fname = fname.strip()
                text = text.strip()
                if not fname or not text:
                    continue
                img_path = os.path.join(subfolder_path, fname)
                if os.path.exists(img_path):
                    items.append({"image_path": img_path, "text": text})

    return items


def load_phsf_words(words_dir: str = None, gen_words_dir: str = None) -> list:
    """
    Ładuje dataset wyrazów PHSF z folderów words i gen_words.

    Struktura words/: podfolderów nazwanych słowem (np. words/boi/boi_000000.png).
    Struktura gen_words/: numerowane podfolderów z labels.txt (nazwa<TAB>słowo).

    Args:
        words_dir: ścieżka do folderu words; None = ./data/phsf/words
        gen_words_dir: ścieżka do gen_words; None = ./data/phsf/gen_words (pominięte gdy None)

    Returns:
        Lista {"image_path": str, "text": str} gotowa dla OCRDataset
    """
    if words_dir is None:
        words_dir = "./data/phsf/words"

    items = []
    skipped_words: set[str] = set()

    # -- words/: subfolder name = słowo, wszystkie PNG w podfolderze należą do tego słowa
    if os.path.isdir(words_dir):
        loaded_words = 0
        skipped_w = 0
        for word_folder in os.listdir(words_dir):
            word_path = os.path.join(words_dir, word_folder)
            if not os.path.isdir(word_path):
                continue
            text = word_folder  # nazwa folderu = transkrypcja
            valid = all(c in char2idx for c in text.lower())
            if not valid:
                skipped_words.add(text)
                skipped_w += 1
                continue
            for fname in os.listdir(word_path):
                if fname.lower().endswith(".png"):
                    items.append({"image_path": os.path.join(word_path, fname), "text": text})
                    loaded_words += 1
        print(f"[load_phsf_words] words: załadowano {loaded_words} obrazów, pominięto {skipped_w} słów")

    # -- gen_words/: numerowane podfolderów z labels.txt (filename<TAB>słowo)
    if gen_words_dir is not None:
        if gen_words_dir == "":
            gen_words_dir = "./data/phsf/gen_words"
        if os.path.isdir(gen_words_dir):
            loaded_gen = 0
            skipped_gen = 0
            for subfolder in os.listdir(gen_words_dir):
                subfolder_path = os.path.join(gen_words_dir, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue
                labels_path = os.path.join(subfolder_path, "labels.txt")
                if not os.path.exists(labels_path):
                    continue
                with open(labels_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        if "\t" not in line:
                            continue
                        fname, text = line.split("\t", 1)
                        fname, text = fname.strip(), text.strip()
                        if not fname or not text:
                            continue
                        if not all(c in char2idx for c in text.lower()):
                            skipped_words.add(text)
                            skipped_gen += 1
                            continue
                        img_path = os.path.join(subfolder_path, fname)
                        if os.path.exists(img_path):
                            items.append({"image_path": img_path, "text": text})
                            loaded_gen += 1
            print(f"[load_phsf_words] gen_words: załadowano {loaded_gen} obrazów, pominięto {skipped_gen} wpisów")

    if skipped_words:
        print(f"[load_phsf_words] Pominięto słowa z nieobsługiwanymi znakami: {len(skipped_words)} typów")
    print(f"[load_phsf_words] Łącznie załadowano: {len(items)} wyrazów")
    return items


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
            model = AuxModel(num_classes=len(CHARS) + 1)
        model.load_state_dict(state_dict)
        info("Model wczytany!")

    else:
        info(f"UWAGA: Nie znaleziono modelu {model_path}")
        if(model_type == 1):
            model = MainModel(num_classes=len(CHARS) + 1)
        else:
            model = AuxModel(num_classes=len(CHARS) + 1)

    model.to(device)
    model.eval()

    return model


#------Zarządzanie transkrypcjami------

def load_transcription(path, return_dict=False):
    """
    word_000 nic
    word_001 dwa
    """

    pairs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) == 1:
                key = parts[0].replace(".png", "")
                text = ""
            else:
                key, text = parts
                key = key.replace(".png", "")

            pairs.append((key, text))

    if return_dict:
        return dict(pairs)

    return [text for _, text in pairs]

def save_aligned_boxes_jsonl(
    page_path,
    trans_path,
    saveto_dir,
    box_dir,
):
    """
    Łączy boxy + transkrypcję i zapisuje jsonl.
    """

    os.makedirs(saveto_dir, exist_ok=True)

    img = cv2.imdecode(np.fromfile(page_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Nie udało się wczytać obrazu: {page_path}")

    img_h, img_w = img.shape[:2]

    edited_boxes = load_boxes_from_annotations(box_dir, img_w, img_h)

    transcript_words = load_transcription(trans_path)


    total = max(
        len(edited_boxes),
        len(transcript_words),
    )

    page_name = os.path.splitext(
        os.path.basename(page_path)
    )[0]

    jsonl_path = os.path.join(
        saveto_dir,
        "boxes.jsonl",
    )

    with open(
        jsonl_path,
        "w",
        encoding="utf-8",
    ) as f:

        for idx in range(total):

            # box
            box = None
            width = None
            height = None

            if idx < len(edited_boxes):

                box = edited_boxes[idx]["box"]

                x1, y1, x2, y2 = box

                width = int(x2 - x1)
                height = int(y2 - y1)

            # tekst
            text = ""

            if idx < len(transcript_words):
                text = transcript_words[idx]

            # entry
            entry = {
                "id": f"word_{idx:03d}",
                "text": text,
                "box": box,
                "width": width,
                "height": height,
                "page_file": os.path.basename(page_path),
            }

            f.write(
                json.dumps(
                    entry,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"[OK] zapisano: {jsonl_path}"
    )

    print(
        f"boxów={len(edited_boxes)} "
        f"| słów={len(transcript_words)} "
        f"| zapisano={total}"
    )

    return jsonl_path


def convert_aligned_to_ttdata(
    aligned_jsonl_path: str,
    page_image_path: str,
    ttdata_dir: str = "ttData",
) -> str:
    """
    Konwertuje aligned_jsonl na strukturę ttData gotową do treningu.

    Tworzy ttData/{name}/ z:
      - source_image.jpg
      - word_000.png, word_001.png, ... (wycinki słów)
      - boxes.jsonl (format kompatybilny z load_all_datasets)

    Pomija wpisy bez boxa lub bez tekstu.
    Nadpisuje istniejący folder.
    """
    entries = load_aligned_jsonl(aligned_jsonl_path)

    File, _ = os.path.splitext(os.path.basename(page_image_path))
    out_dir = os.path.join(ttdata_dir, File)
    out_dir = os.path.join("data", out_dir)
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imdecode(np.fromfile(page_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Nie udało się wczytać obrazu: {page_image_path}")

    dest_image = os.path.join(out_dir, "source_image.jpg")
    cv2.imwrite(dest_image, img)
    abs_source = os.path.relpath(dest_image, start=os.getcwd())

    img_h, img_w = img.shape[:2]
    
    jsonl_path = os.path.join(out_dir, "boxes.jsonl")

    written = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in entries:
            box = entry.get("box")
            text = entry.get("text", "").strip()

            if not box or not text:
                continue

            word_id = entry.get("id", f"word_{written:03d}")
            crop_filename = f"{word_id}.png"
            crop_path = os.path.join(out_dir, crop_filename)

            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            if x2 > x1 and y2 > y1:
                cv2.imwrite(crop_path, img[y1:y2, x1:x2])
            else:
                continue

            record = {
                "id": written,
                "crop_file": crop_filename,
                "bbox_xyxy": [x1, y1, x2, y2],
                "text": text,
                "suggested_text": text,
                "prediction_score": 0.0,
                "text_override": True,
                "prob": 1.0,
                "source_image": abs_source,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"[ttData] {out_dir}  ({written} wpisów)")
    return out_dir


def load_aligned_jsonl(jsonl_path):
    """
    Zwraca listę:
    [
        {
            "id": "word_000",
            "text": "nic",
            "box": [x1,y1,x2,y2],
            ...
        }
    ]
    """

    entries = []

    if not os.path.exists(jsonl_path):
        return entries

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            data = json.loads(line)

            entries.append(data)

    return entries

def save_aligned_jsonl(jsonl_path, entries):
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in entries:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

def aligned_to_editor_boxes(entries):
    boxes = []

    for item in entries:

        box = item.get("box")

        if not box:
            continue

        x1, y1, x2, y2 = box

        boxes.append(
            {
                "box": [x1, y1, x2, y2],
                "text": item.get("text", ""),
                "id": item.get("id"),
            }
        )

    return boxes

def merge_editor_changes(entries, edited_boxes):

    by_id = {
        e["id"]: e
        for e in entries
    }

    for box_data in edited_boxes:

        word_id = box_data["id"]

        if word_id not in by_id:
            continue

        item = by_id[word_id]

        item["box"] = box_data["box"]

        if "text" in box_data:
            item["text"] = box_data["text"]

        if item["box"]:

            x1, y1, x2, y2 = item["box"]

            item["width"] = int(x2 - x1)
            item["height"] = int(y2 - y1)

    return list(by_id.values())
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


DEFAULT_MODELS = [
    "v4", "v4.1", "v4.4",
    "v6.2", "v6.5",
    "v7.2", "v7.4", "v7.5",
    "v8.2",
    "v10.2", "v10.3", "v10.4", "v10.5",
    "v11.1", "v11.2", "v11.3", "v11.4", "v11.5",
]

DEFAULT_MODEL = "v11.5"


def auto_select_models(models_dir: str, version: str | None = None):
    all_models = list_models(models_dir)

    if version:
        candidates = [m for m in all_models if m.startswith(f"v{version}")]
    else:
        candidates = [m for m in DEFAULT_MODELS if m in all_models]

    if not candidates:
        if all_models:
            candidates = [all_models[-1]]
        else:
            return DEFAULT_MODEL, []

    if version:
        default = candidates[-1]
    else:
        default = DEFAULT_MODEL if DEFAULT_MODEL in candidates else candidates[-1]

    return default, candidates


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
# ── Embedding-based autokorekta HTR ──────────────────────────────────────────

_EMBED_DIM = 2048
_NGRAM_SIZES = (2, 3)

# Cache słownikowych embeddingów (ładowany raz na sesję)
_embed_cache: tuple[str, list[str], np.ndarray] | None = None

# Mapowanie polskich diakrytyków na ASCII — używane przed embeddingiem
# żeby HTR-owe "sloce" trafiało w to samo przestrzeń co słownikowe "słońce"
_DIACRITIC_MAP = str.maketrans(
    "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ",
    "acelnoszzACELNOSZZ",
)


def _to_ascii(word: str) -> str:
    """Usuwa polskie diakrytyki: 'słońce' → 'slonce', 'życiem' → 'zyciem'."""
    return word.translate(_DIACRITIC_MAP)


def _stable_hash(s: str, mod: int) -> int:
    """Deterministyczny hash stringa (niezależny od PYTHONHASHSEED)."""
    h = 0
    for c in s:
        h = (h * 31 + ord(c)) % mod
    return h


def _word_embedding(word: str, dim: int = _EMBED_DIM) -> np.ndarray:
    """
    Zamienia słowo na wektor character n-gram (styl FastText).
    Przed embeddingiem normalizuje diakrytyki do ASCII, dzięki czemu
    HTR-owe 'sloce' i słownikowe 'słońce' trafiają w podobną przestrzeń.
    """
    normalized = _to_ascii(word.lower())
    padded = f"<{normalized}>"
    vec = np.zeros(dim, dtype=np.float32)
    for n in _NGRAM_SIZES:
        for i in range(len(padded) - n + 1):
            gram = padded[i : i + n]
            vec[_stable_hash(gram, dim)] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def _get_dict_embeddings(dict_path: str = ÐICT_PATH) -> tuple[list[str], np.ndarray]:
    """
    Zwraca (words, matrix) z cache'em na poziomie modułu.
    matrix[i] to L2-znormalizowany embedding words[i].
    """
    global _embed_cache
    if _embed_cache is not None and _embed_cache[0] == dict_path:
        return _embed_cache[1], _embed_cache[2]

    words = load_dictionary(dict_path)
    matrix = np.stack([_word_embedding(w) for w in words])  # (N, dim)
    _embed_cache = (dict_path, words, matrix)
    return words, matrix


def _rerank_score(candidate: str, query: str, embed_sim: float) -> float:
    """
    Drugi etap scoringu (reranking) łączący kilka sygnałów:
      - cosine similarity z fazy 1 (embedding)
      - SequenceMatcher (dokładniejsze porównanie znaków)
      - podobieństwo długości
      - bonus za zgodność pierwszego znaku
    """
    # Porównanie po ASCII — żeby 'sloce' vs 'słońce' miało wysokie seq_sim
    q_ascii = _to_ascii(query.lower())
    c_ascii = _to_ascii(candidate.lower())
    seq_sim = SequenceMatcher(None, q_ascii, c_ascii).ratio()
    lq, lc = len(query), len(candidate)
    len_sim = 1.0 - abs(lq - lc) / max(lq, lc, 1)
    prefix = 0.1 if (q_ascii and c_ascii and q_ascii[0] == c_ascii[0]) else 0.0
    return 0.40 * embed_sim + 0.40 * seq_sim + 0.15 * len_sim + prefix


def _score_candidate_with_vectors(
    candidate: str,
    letter_vectors: list,
) -> float:
    """
    Ocenia kandydata ze słownika używając wektorów prawdopodobieństwa modelu.

    Używa DTW (Dynamic Time Warping) do optymalnego wyrównania liter modelu
    do liter kandydata — działa poprawnie nawet gdy długości się różnią
    (HTR odczytał za mało lub za dużo liter).

    letter_vectors: lista (litera, np.ndarray kształtu C) z predict_letter()
    Zwraca: score w zakresie (-inf, 0] — wyższy = lepszy kandydat
    """
    if not letter_vectors:
        return -999.0

    n_obs  = len(letter_vectors)
    n_cand = len(candidate)

    # Odrzuć kandydatów drastycznie różniących się długością (>2x)
    if n_cand == 0 or n_obs / n_cand > 2.5 or n_cand / max(n_obs, 1) > 2.5:
        return -999.0

    # Macierz kosztu: cost[i][j] = -log P(candidate[j] | wektor timestep i)
    cost = np.empty((n_obs, n_cand), dtype=np.float32)
    for i, (_, prob_vec) in enumerate(letter_vectors):
        for j in range(n_cand):
            c = candidate[j]
            c_ascii = _to_ascii(c)
            p = 1e-9
            for ch in (c, c_ascii, c.upper(), c_ascii.upper()):
                idx = char2idx.get(ch)
                if idx is not None and idx < len(prob_vec):
                    p = max(p, float(prob_vec[idx]))
            cost[i, j] = -math.log(p)

    # DTW: dp[i][j] = minimalny koszt wyrównania obs[0..i] → cand[0..j]
    dp = np.full((n_obs + 1, n_cand + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n_obs + 1):
        for j in range(1, n_cand + 1):
            dp[i, j] = cost[i - 1, j - 1] + min(
                dp[i - 1, j - 1],   # dopasowanie 1:1
                dp[i - 1, j],       # pominięcie obserwowanej litery (HTR za dużo)
                dp[i, j - 1],       # pominięcie litery kandydata (HTR za mało)
            )

    path_len = max(n_obs, n_cand)
    return -dp[n_obs, n_cand] / path_len  # normalizacja do [-inf, 0]


# ── Częstość słów polskich (prior językowy) ───────────────────────────────────
# Wartości znormalizowane do [0, 1]: 1.0 = najczęstsze słowa funkcyjne,
# 0.65 = częste słowa treściowe, 0.35 = umiarkowanie częste.
# Słowa spoza listy domyślnie: 0.05 (obecne w słowniku, ale rzadkie).
_WORD_FREQ: dict[str, float] = {
    # Tier 1 — stopwords i słowa funkcyjne (~60 słów)
    **{w: 1.0 for w in [
        "nie", "się", "to", "jest", "jak", "co", "do", "na", "że", "już",
        "też", "ale", "po", "tak", "czy", "go", "mi", "mu", "jej", "jego",
        "ich", "nas", "je", "tam", "tu", "ze", "przy", "przez", "dla", "we",
        "od", "za", "bo", "sobie", "wszystko", "kiedy", "gdzie", "jeszcze",
        "tylko", "więc", "bardzo", "tego", "tej", "ten", "ta", "te", "sam",
        "być", "mieć", "i", "w", "z", "o", "a", "u", "ni", "nic", "kto",
        "pan", "pani", "temu", "tą", "tych", "tymi", "tego",
    ]},
    # Tier 2 — częste słowa treściowe (~100 słów)
    **{w: 0.65 for w in [
        "dom", "człowiek", "rok", "czas", "dzień", "noc", "życie", "świat",
        "ręka", "głowa", "oko", "słowo", "droga", "serce", "praca", "woda",
        "ziemia", "ludzie", "chwila", "myśl", "twarz", "drzwi", "okno",
        "niebo", "nikt", "coś", "ktoś", "każdy", "razem", "teraz", "właśnie",
        "może", "pewnie", "chyba", "naprawdę", "jeden", "dwa", "trzy", "raz",
        "miejsce", "kraj", "miasto", "las", "góra", "morze", "księżyc",
        "słońce", "gwiazda", "kwiat", "drzewo", "ptak", "ryba", "kot", "pies",
        "miał", "mówić", "widzieć", "powiedzieć", "robić", "wziąć", "dać",
        "iść", "stać", "siedzieć", "czuć", "myśleć", "znać", "wracać",
        "nowe", "stare", "duże", "małe", "dobre", "złe", "wielkie", "długie",
        "pierwsze", "drugie", "całe", "własne", "ludzkie", "inne", "same",
        "często", "zawsze", "nigdy", "wszędzie", "gdzieś", "kiedyś", "czegoś",
        "domu", "czasu", "dnia", "nocy", "życia", "świata", "ręki", "głowy",
        "oczu", "słów", "drogi", "serca", "pracy", "wody", "ziemi", "ludzi",
        "chwili", "myśli", "twarzy", "nieba", "słońca", "gwiazd", "kwiatów",
    ]},
    # Tier 3 — umiarkowanie częste (~80 słów)
    **{w: 0.35 for w in [
        "przyczyna", "przyczyny", "przyczynie", "przyczynę",
        "zdarzać", "zdarza", "zdarzył", "zdarzyła", "zdarzyć", "zdarzę",
        "słoneczna", "słoneczny", "słoneczne", "słonecznej",
        "prawa", "prawem", "prawny", "prawo", "prawem", "prawda", "prawdy",
        "wróbel", "wróble", "wróbli", "wróblom",
        "drzewo", "drzewa", "drzewem", "drzew",
        "morze", "morza", "morzem", "mórz",
        "góra", "góry", "górze", "gór",
        "rzeka", "rzeki", "rzece", "rzek",
        "kamień", "kamienia", "kamieniu", "kamieni",
        "ogień", "ognia", "ogniu", "ogniem",
        "powietrze", "powietrza", "powietrzem",
        "cisza", "ciszy", "ciszą", "ciszę",
        "radość", "radości", "radością",
        "smutek", "smutku", "smutkiem",
        "miłość", "miłości", "miłością",
        "wieczór", "wieczoru", "wieczorem",
        "ranek", "ranka", "rankiem",
        "wiatr", "wiatru", "wiatrem",
        "deszcz", "deszczu", "deszczem",
        "śnieg", "śniegu", "śniegiem",
    ]},
}


def _freq_score(word: str) -> float:
    """Zwraca znormalizowaną częstość słowa [0, 1]. Default 0.05 dla słów spoza listy."""
    return _WORD_FREQ.get(word.lower(), 0.05)


#--- Autokorekta------
def DictCorrect(
    text: str,
    confidence: float = 100.0,
    threshold: float = 0.55,
    top_k: int = 30,
    letter_vectors: list | None = None,
) -> str:
    """
    Pipeline autokorekty HTR:
      embedding → cosine similarity → TOP-5 → reranking → wybór najlepszego.

    Etapy rerankingu (w kolejności priorytetu):
      1. _score_candidate_with_vectors — używa wektorów prawdopodobieństwa
         modelu per litera (jeśli letter_vectors dostarczone i długości zgodne)
      2. _rerank_score — embedding + SequenceMatcher + długość (fallback)

    confidence (0-100): pewność modelu HTR.
    letter_vectors: lista (litera, np.ndarray) z predict_letter() — opcjonalna.
    """
    text = text.strip()
    if not text:
        return text

    # Jeśli słowo jest już w słowniku — nie ma co korygować
    words_check, _ = _get_dict_embeddings()
    if text.lower() in {w.lower() for w in words_check}:
        return text

    # 1. Embedding słowa wejściowego (HTR output)
    query_vec = _word_embedding(text)

    # 2. Porównanie ze słownikiem przez cosine similarity
    words, dict_matrix = _get_dict_embeddings()
    similarities = dict_matrix @ query_vec  # (N,) — szybki iloczyn macierzowy

    # 3. TOP-K kandydatów według embeddingu
    k = min(top_k, len(words))
    top_idx = np.argpartition(similarities, -k)[-k:]
    top_idx = top_idx[np.argsort(similarities[top_idx])[::-1]]
    top_candidates = [(words[i], float(similarities[i])) for i in top_idx]

    # 4. Reranking: jeśli mamy letter_vectors — używamy wektorów modelu
    #    W przeciwnym razie fallback do SequenceMatcher + embedding
    use_vectors = bool(letter_vectors)

    def _combined_score(word: str, embed_sim: float) -> float:
        freq = _freq_score(word)
        if use_vectors:
            vec_score = _score_candidate_with_vectors(word, letter_vectors)
            if vec_score > -999.0:
                vec_norm = math.exp(vec_score)
                return 0.46 * vec_norm + 0.27 * embed_sim + 0.18 * _rerank_score(word, text, embed_sim) + 0.09 * freq
        return _rerank_score(word, text, embed_sim) + 0.09 * freq

    reranked = sorted(
        top_candidates,
        key=lambda x: _combined_score(x[0], x[1]),
        reverse=True,
    )

    # 5. Wybór najlepszego słowa z uwzględnieniem confidence modelu HTR
    conf_factor = confidence / 100.0
    effective_threshold = threshold + (0.90 - threshold) * conf_factor

    best_word, best_embed_sim = reranked[0]
    best_score = _combined_score(best_word, best_embed_sim)

    if best_score >= effective_threshold:
        if text and text[0].isupper():
            best_word = best_word[0].upper() + best_word[1:]
        return best_word
    return text
