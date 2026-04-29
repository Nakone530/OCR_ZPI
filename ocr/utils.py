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
from torchvision import transforms
import torchvision.transforms.functional as F

from datetime import datetime
from typing import Any
from datetime import date

from pathlib import Path
import matplotlib.pyplot as plt
from .config import IMAGE_SIZE, MEAN, STD
from . import info


# ── Transformacje ──────────────────────────────────────────────────────────────

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
            RandomOtsu(Otsu()),]
    if getattr(args, "denoise", False):
        pack.append(trans_denoise_bil())

    pack.extend(base_transform())
    return transforms.Compose(pack)


def get_train_transform(args) -> transforms.Compose:
    """
    Zwraca pipeline transformacji obrazu do trenowania modelu (z augmentacją danych).
    
    Pipeline zawiera:
      - Losowa rotacja obrazu o maksymalnie 10 stopni (augmentacja)
      - Konwersja do skali szarości (1 kanał)
      - Zmiana rozmiaru do IMAGE_SIZE x IMAGE_SIZE
      - Konwersja do tensora PyTorch
      - Normalizacja wartości pikseli (mean=0.5, std=0.5)
    
    Zwraca:
        transforms.Compose: Złożona transformacja z augmentacją
                            do użycia podczas trenowania modelu.
    
    Przykład:
        >>> train_transform = get_train_transform()
        >>> tensor = train_transform(pil_image)
    """
    pack = [
        transforms.RandomRotation(5),
        RandomDenoise(),
        RandomPadding(),
        ResizeWithAspect(),
        RandomOtsu(Otsu())
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
    
def preprocess_letter(img: np.ndarray) -> np.ndarray:
    """
    Przetwarza obraz pojedynczej litery przed klasyfikacją.
    
    Operacje wykonywane na obrazie:
      1. Dodaje biały padding (10px) wokół obrazu
      2. Tworzy kwadratowy canvas o rozmiarze max(wysokość, szerokość)
      3. Centruje literę na canvasie z białym tłem
      4. Skaluje wynikowy obraz do rozmiaru 28x28 pikseli
    
    Argumenty:
        img (np.ndarray): Obraz litery w skali szarości jako tablica numpy.
                          Oczekiwany format: (wysokość, szerokość), dtype uint8.
    
    Zwraca:
        np.ndarray: Przetworzony obraz litery o wymiarach 28x28 pikseli.
    
    Przykład:
        >>> letter = preprocess_letter(letter_array)
        >>> letter.shape
        (28, 28)
    """
    pad = 10
    img = np.pad(img, pad, mode='constant', constant_values=255)

    h, w = img.shape
    size = max(h, w)
    
    new_img = np.full((size, size), 255, dtype=img.dtype)
    
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    
    new_img[y_offset:y_offset+h, x_offset:x_offset+w] = img
    
    new_img = cv2.resize(new_img, (28, 28))
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
