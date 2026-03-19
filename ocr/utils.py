"""
Narzędzia pomocnicze:
  - transformacje obrazów
  - odszumianie (OpenCV)
  - ładowanie obrazów (PNG/JPG/PDF) z opcjonalnym odszumianiem
  - zapis obrazu do folderu z dzisiejszą datą
"""

import os
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from torchvision import transforms

from .config import IMAGE_SIZE, MEAN, STD


# ── Transformacje ──────────────────────────────────────────────────────────────

def get_transform() -> transforms.Compose:
    """Transformacja do inference (bez augmentacji)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def get_train_transform() -> transforms.Compose:
    """Transformacja do trenowania (z augmentacją)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def preprocess_letter(img):
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

# ── Konwersje PIL ↔ OpenCV ─────────────────────────────────────────────────────

def _pil_to_bgr(img_pil: Image.Image) -> np.ndarray:
    rgb = img_pil.convert("L")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ── Metody odszumiania ─────────────────────────────────────────────────────────

def _denoise_nlm_color(bgr: np.ndarray, h: int, hColor: int) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(bgr, None, h, hColor, 7, 21)


def _denoise_median(bgr: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.medianBlur(bgr, ksize)


def _denoise_bilateral(bgr: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(bgr, 9, 75, 75)


def _denoise_gaussian(bgr: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.GaussianBlur(bgr, (ksize, ksize), 0)


def denoise_pil(
    img_pil: Image.Image,
    method: str,
    h: int = 10,
    hColor: int = 10,
    ksize: int = 3,
) -> Image.Image:
    """
    Odszumia obraz PIL.

    method: 'nlm-color' | 'median' | 'bilateral' | 'gaussian'
    """
    bgr = _pil_to_bgr(img_pil)
    m = method.lower()
    if m == "nlm-color":
        out = _denoise_nlm_color(bgr, h, hColor)
    elif m == "median":
        out = _denoise_median(bgr, ksize)
    elif m == "bilateral":
        out = _denoise_bilateral(bgr)
    elif m == "gaussian":
        out = _denoise_gaussian(bgr, ksize)
    else:
        raise ValueError(f"Nieznana metoda odszumiania: {method}")
    return _bgr_to_pil(out)


# ── Ładowanie obrazu ───────────────────────────────────────────────────────────

def load_image(image_path: str, mode: str = "RGB") -> Image.Image:
    """Ładuje obraz (PNG/JPG/PDF) i konwertuje do podanego trybu."""
    suffix = Path(image_path).suffix.lower()
    if suffix == ".pdf":
        img = convert_from_path(image_path, dpi=300)[0]
    else:
        img = Image.open(image_path)
    return img.convert(mode)


def load_and_optionally_denoise(image_path: str, args, mode: str = "L") -> Image.Image:
    """
    Ładuje obraz i opcjonalnie odszumia go wg parametrów z args.
    mode: 'RGB' (klasyfikacja) lub 'L' (segmentacja).
    """
    suffix = Path(image_path).suffix.lower()

    if suffix == ".pdf":
        return load_image(image_path, mode)

    img = Image.open(image_path)

    if getattr(args, "denoise", False):
        img_rgb = img.convert("L")
        img_rgb = denoise_pil(
            img_rgb,
            method=args.denoise_method,
            h=getattr(args, "h", 10),
            hColor=getattr(args, "hColor", 10),
            ksize=getattr(args, "ksize", 3),
        )
        return img_rgb.convert("L")

    return img.convert(mode)


# ── Zapis do folderu z datą ────────────────────────────────────────────────────

def get_today_folder() -> str:
    """Zwraca ścieżkę do folderu z dzisiejszą datą (tworzy, jeśli brak)."""
    folder = os.path.join(".", date.today().strftime("%Y-%m-%d"))
    os.makedirs(folder, exist_ok=True)
    return folder


def save_image_to_today_folder(image_path: str) -> str:
    """
    Kopiuje obraz do folderu z dzisiejszą datą jako kolejny plik PNG
    (1.png, 2.png, …). Zwraca ścieżkę docelową.
    """
    folder = get_today_folder()
    existing = [
        int(os.path.splitext(f)[0])
        for f in os.listdir(folder)
        if f.endswith(".png") and os.path.splitext(f)[0].isdigit()
    ]
    dest = os.path.join(folder, f"{max(existing, default=0) + 1}.png")
    load_image(image_path, mode="L").save(dest, format="PNG")
    print(f"Zapisano zdjęcie jako: {dest}")
    return dest
