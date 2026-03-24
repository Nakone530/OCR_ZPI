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

from datetime import datetime
from typing import Any
from datetime import date

from pathlib import Path
import matplotlib.pyplot as plt
from .config import IMAGE_SIZE, MEAN, STD


# ── Transformacje ──────────────────────────────────────────────────────────────

def get_transform() -> transforms.Compose:
    """
    Zwraca pipeline transformacji obrazu do inferencji (bez augmentacji danych).
    
    Pipeline zawiera:
      - Konwersja do skali szarości (1 kanał)
      - Zmiana rozmiaru do IMAGE_SIZE x IMAGE_SIZE
      - Konwersja do tensora PyTorch
      - Normalizacja wartości pikseli (mean=0.5, std=0.5)
    
    Argumenty:
        Brak argumentów.
    
    Zwraca:
        transforms.Compose: Złożona transformacja gotowa do użycia
                            na obrazach PIL podczas predykcji.
    
    Przykład:
        >>> transform = get_transform()
        >>> tensor = transform(pil_image)
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def get_train_transform() -> transforms.Compose:
    """
    Zwraca pipeline transformacji obrazu do trenowania modelu (z augmentacją danych).
    
    Pipeline zawiera:
      - Konwersja do skali szarości (1 kanał)
      - Zmiana rozmiaru do IMAGE_SIZE x IMAGE_SIZE
      - Losowa rotacja obrazu o maksymalnie 10 stopni (augmentacja)
      - Konwersja do tensora PyTorch
      - Normalizacja wartości pikseli (mean=0.5, std=0.5)
    
    Argumenty:
        Brak argumentów.
    
    Zwraca:
        transforms.Compose: Złożona transformacja z augmentacją
                            do użycia podczas trenowania modelu.
    
    Przykład:
        >>> train_transform = get_train_transform()
        >>> tensor = train_transform(pil_image)
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


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

# ── Konwersje PIL ↔ OpenCV ─────────────────────────────────────────────────────

def _pil_to_bgr(img_pil: Image.Image) -> np.ndarray:
    """
    Konwertuje obraz PIL do formatu BGR (OpenCV).
    
    Funkcja konwertuje obraz PIL na format skali szarości,
    następnie do tablicy numpy i finalnie do formatu BGR
    używanego przez OpenCV.
    
    Argumenty:
        img_pil (Image.Image): Obraz w formacie PIL (dowolny tryb kolorów).
    
    Zwraca:
        np.ndarray: Obraz w formacie BGR jako tablica numpy,
                    gotowy do przetwarzania przez funkcje OpenCV.
    
    Uwaga:
        Ta funkcja jest pomocnicza i używana wewnętrznie przez
        funkcje odszumiania.
    """
    rgb = img_pil.convert("L")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """
    Konwertuje obraz z formatu BGR (OpenCV) do formatu PIL.
    
    Argumenty:
        img_bgr (np.ndarray): Obraz w formacie BGR jako tablica numpy.
    
    Zwraca:
        Image.Image: Obraz w formacie PIL w trybie RGB.
    
    Uwaga:
        Ta funkcja jest pomocnicza i używana wewnętrznie przez
        funkcje odszumiania.
    """
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ── Metody odszumiania ─────────────────────────────────────────────────────────

def _denoise_nlm_color(bgr: np.ndarray, h: int, hColor: int) -> np.ndarray:
    """
    Odszumia obraz metodą Non-Local Means dla obrazów kolorowych.
    
    Metoda NLM porównuje podobieństwo bloków pikseli w całym obrazie,
    uśredniając piksele o podobnym otoczeniu. Skuteczna przy szumie Gaussowskim.
    
    Argumenty:
        bgr (np.ndarray): Obraz wejściowy w formacie BGR.
        h (int): Siła filtrowania dla kanału jasności.
                 Większa wartość = silniejsze odszumianie, ale utrata detali.
        hColor (int): Siła filtrowania dla kanałów koloru.
    
    Zwraca:
        np.ndarray: Odszumiony obraz w formacie BGR.
    """
    return cv2.fastNlMeansDenoisingColored(bgr, None, h, hColor, 7, 21)


def _denoise_median(bgr: np.ndarray, ksize: int) -> np.ndarray:
    """
    Odszumia obraz filtrem medianowym.
    
    Filtr medianowy zastępuje każdy piksel medianą wartości w oknie.
    Skuteczny przy usuwaniu szumu typu "sól i pieprz".
    
    Argumenty:
        bgr (np.ndarray): Obraz wejściowy w formacie BGR.
        ksize (int): Rozmiar okna filtra (musi być liczbą nieparzystą, np. 3, 5, 7).
    
    Zwraca:
        np.ndarray: Odszumiony obraz w formacie BGR.
    """
    return cv2.medianBlur(bgr, ksize)


def _denoise_bilateral(bgr: np.ndarray) -> np.ndarray:
    """
    Odszumia obraz filtrem bilateralnym.
    
    Filtr bilateralny wygładza obraz zachowując krawędzie.
    Uwzględnia zarówno odległość przestrzenną, jak i różnicę intensywności.
    
    Argumenty:
        bgr (np.ndarray): Obraz wejściowy w formacie BGR.
    
    Zwraca:
        np.ndarray: Odszumiony obraz w formacie BGR z zachowanymi krawędziami.
    
    Uwaga:
        Parametry filtra są ustalone: d=9, sigmaColor=75, sigmaSpace=75.
    """
    return cv2.bilateralFilter(bgr, 9, 75, 75)


def _denoise_gaussian(bgr: np.ndarray, ksize: int) -> np.ndarray:
    """
    Odszumia obraz rozmyciem Gaussowskim.
    
    Stosuje filtr Gaussowski który wygładza obraz poprzez
    uśrednianie ważone pikseli w oknie (wagi wg rozkładu Gaussa).
    
    Argumenty:
        bgr (np.ndarray): Obraz wejściowy w formacie BGR.
        ksize (int): Rozmiar jądra filtra (musi być liczbą nieparzystą).
    
    Zwraca:
        np.ndarray: Rozmyty obraz w formacie BGR.
    """
    return cv2.GaussianBlur(bgr, (ksize, ksize), 0)


def denoise_pil(
    img_pil: Image.Image,
    method: str,
    h: int = 10,
    hColor: int = 10,
    ksize: int = 3,
) -> Image.Image:
    """
    Odszumia obraz PIL wybraną metodą.
    
    Główna funkcja do odszumiania obrazów. Wspiera różne metody
    odszumiania dostępne w OpenCV.
    
    Argumenty:
        img_pil (Image.Image): Obraz wejściowy w formacie PIL.
        method (str): Metoda odszumiania do użycia. Dostępne opcje:
            - 'nlm-color': Non-Local Means dla obrazów kolorowych
            - 'median': Filtr medianowy
            - 'bilateral': Filtr bilateralny (zachowuje krawędzie)
            - 'gaussian': Rozmycie Gaussowskie
        h (int, opcjonalnie): Siła filtrowania dla NLM (jasność). Domyślnie 10.
        hColor (int, opcjonalnie): Siła filtrowania dla NLM (kolor). Domyślnie 10.
        ksize (int, opcjonalnie): Rozmiar jądra dla median/gaussian. Domyślnie 3.
    
    Zwraca:
        Image.Image: Odszumiony obraz w formacie PIL.
    
    Wyjątki:
        ValueError: Gdy podano nieznaną metodę odszumiania.
    
    Przykład:
        >>> denoised = denoise_pil(img, method='bilateral')
        >>> denoised = denoise_pil(img, method='median', ksize=5)
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
        img = convert_from_path(image_path, dpi=300)[0]
    else:
        img = Image.open(image_path)

    return img.convert(mode)


def load_and_optionally_denoise(image_path: str, args, mode: str = "L") -> Image.Image:
    """
    Ładuje obraz i opcjonalnie stosuje odszumianie na podstawie argumentów CLI.
    
    Funkcja łączy ładowanie obrazu z opcjonalnym preprocessingiem (odszumianiem).
    Parametry odszumiania są pobierane z obiektu args (argparse Namespace).
    
    Argumenty:
        image_path (str): Ścieżka do pliku obrazu (PNG, JPG, PDF).
        args: Obiekt argparse.Namespace z parametrami. Oczekiwane atrybuty:
            - denoise (bool): Czy stosować odszumianie
            - denoise_method (str): Metoda odszumiania (jeśli denoise=True)
            - h (int, opcjonalnie): Parametr siły dla NLM
            - hColor (int, opcjonalnie): Parametr koloru dla NLM
            - ksize (int, opcjonalnie): Rozmiar jądra dla median/gaussian
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
    folder = os.path.join(".", date.today().strftime("%Y-%m-%d"))
    os.makedirs(folder, exist_ok=True)
    return folder

def save_image_to_today_folder(image_path: str) -> str:
    """
    Kopiuje obraz do folderu z dzisiejszą datą z automatyczną numeracją.
    
    Obrazy są zapisywane jako pliki PNG z kolejnymi numerami (1.png, 2.png, ...).
    Funkcja automatycznie wykrywa istniejące pliki i nadaje następny numer.
    
    Argumenty:
        image_path (str): Ścieżka do obrazu źródłowego do skopiowania.
    
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
