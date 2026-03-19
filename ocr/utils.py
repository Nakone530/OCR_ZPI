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

from .config import IMAGE_SIZE, MEAN, STD


# ── Transformacje ──────────────────────────────────────────────────────────────

def get_transform() -> transforms.Compose:
    """Transformacja do inference (bez augmentacji)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_train_transform() -> transforms.Compose:
    """Transformacja do trenowania (z augmentacją)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ── Konwersje PIL ↔ OpenCV ─────────────────────────────────────────────────────

def _pil_to_bgr(img_pil: Image.Image) -> np.ndarray:
    if cv2 is None:
        raise ModuleNotFoundError(
            "Brak modułu 'cv2'. Zainstaluj: pip install opencv-python "
            "(wymagane tylko dla opcji --denoise)."
        )
    rgb = img_pil.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    if cv2 is None:
        raise ModuleNotFoundError(
            "Brak modułu 'cv2'. Zainstaluj: pip install opencv-python "
            "(wymagane tylko dla opcji --denoise)."
        )
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ── Metody odszumiania ─────────────────────────────────────────────────────────

def _denoise_nlm_color(bgr: np.ndarray, h: int, hColor: int) -> np.ndarray:
    if cv2 is None:
        raise ModuleNotFoundError(
            "Brak modułu 'cv2'. Zainstaluj: pip install opencv-python "
            "(wymagane tylko dla opcji --denoise)."
        )
    return cv2.fastNlMeansDenoisingColored(bgr, None, h, hColor, 7, 21)


def _denoise_median(bgr: np.ndarray, ksize: int) -> np.ndarray:
    if cv2 is None:
        raise ModuleNotFoundError(
            "Brak modułu 'cv2'. Zainstaluj: pip install opencv-python "
            "(wymagane tylko dla opcji --denoise)."
        )
    return cv2.medianBlur(bgr, ksize)


def _denoise_bilateral(bgr: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise ModuleNotFoundError(
            "Brak modułu 'cv2'. Zainstaluj: pip install opencv-python "
            "(wymagane tylko dla opcji --denoise)."
        )
    return cv2.bilateralFilter(bgr, 9, 75, 75)


def _denoise_gaussian(bgr: np.ndarray, ksize: int) -> np.ndarray:
    if cv2 is None:
        raise ModuleNotFoundError(
            "Brak modułu 'cv2'. Zainstaluj: pip install opencv-python "
            "(wymagane tylko dla opcji --denoise)."
        )
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


def load_and_optionally_denoise(image_path: str, args, mode: str = "RGB") -> Image.Image:
    """
    Ładuje obraz i opcjonalnie odszumia go wg parametrów z args.
    mode: 'RGB' (klasyfikacja) lub 'L' (segmentacja).
    """
    suffix = Path(image_path).suffix.lower()

    if suffix == ".pdf":
        return load_image(image_path, mode)

    img = Image.open(image_path)

    if getattr(args, "denoise", False):
        img_rgb = img.convert("RGB")
        img_rgb = denoise_pil(
            img_rgb,
            method=args.denoise_method,
            h=getattr(args, "h", 10),
            hColor=getattr(args, "hColor", 10),
            ksize=getattr(args, "ksize", 3),
        )
        return img_rgb if mode == "RGB" else img_rgb.convert("L")

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
    load_image(image_path, mode="RGB").save(dest, format="PNG")
    print(f"Zapisano zdjęcie jako: {dest}")
    return dest
