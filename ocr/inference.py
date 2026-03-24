"""
Moduł rozpoznawania tekstu (inference).

Odpowiedzialności:
  - wczytywanie wytrenowanego modelu z dysku
  - predykcja pojedynczej litery
  - segmentacja i rozpoznawanie wyrazu (jedna linia)
  - segmentacja i rozpoznawanie wielu linii tekstu
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from PIL import Image
from .config import CHARS, MODEL_PATH, NUM_CLASSES
from .model import SimpleCNN
from .utils import get_transform, load_and_optionally_denoise, preprocess_letter, save_image_to_temp_folder
from .display import visualize_prediction

# ── Ładowanie modelu ───────────────────────────────────────────────────────────

print(matplotlib.get_backend())

def load_model(model_path: str = MODEL_PATH, device: torch.device = None) -> nn.Module:
    """
    Wczytuje wytrenowany model SimpleCNN z pliku.
    
    Funkcja obsługuje dwa formaty zapisu:
      - Czysty state_dict (stary format)
      - Checkpoint z metadanymi (nowy format)
    
    Automatycznie wykrywa liczbę klas z zapisanego modelu.
    
    Argumenty:
        model_path (str, opcjonalnie): Ścieżka do pliku modelu (.pth).
                                    Domyślnie MODEL_PATH z config.
        device (torch.device, opcjonalnie): Urządzenie do załadowania modelu.
                                         Domyślnie auto-wykrywane (CUDA/CPU).
    
    Zwraca:
        nn.Module: Załadowany model SimpleCNN w trybie ewaluacji (eval mode).
    
    Efekty uboczne:
        - Wyświetla komunikaty o ładowaniu na konsolę
        - Ostrzeżenie jeśli model nie istnieje
    
    Przykład:
        >>> model = load_model("./model_ocr.pth")
        Wczytywanie modelu z ./model_ocr.pth...
        Wykryto 26 klas w zapisanym modelu
        Model wczytany!
    
    Uwaga:
        Jeśli plik modelu nie istnieje, zwraca niezainicjowany model
        z losowymi wagami (wyniki będą losowe).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(model_path):
        print(f"Wczytywanie modelu z {model_path}...")

        checkpoint = torch.load(model_path, map_location=device)

        # Pobierz state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            format_info = "checkpoint (nowy format)"
        else:
            state_dict = checkpoint
            format_info = "state_dict (stary format)"

        # Automatycznie wykryj liczbę klas z zapisanego modelu
        num_classes_from_model = state_dict["classifier.4.weight"].shape[0]
        print(f"Wykryto {num_classes_from_model} klas w zapisanym modelu")

        model = SimpleCNN(num_classes=num_classes_from_model)
        model.load_state_dict(state_dict)
        print(f"Wczytano {format_info}")
        print("Model wczytany!")
    else:
        print(f"UWAGA: Nie znaleziono modelu {model_path}")
        print("Model nie jest wytrenowany - wyniki będą losowe!")
        print("Najpierw uruchom: python run_local.py --train")
        model = SimpleCNN(num_classes=NUM_CLASSES)

    model.to(device)
    model.eval()
    return model


# ── Segmentacja pomocnicza ─────────────────────────────────────────────────────

def _find_bounds(projection: np.ndarray) -> list[tuple[int, int]]:
    """
    Wykrywa granice niepustych segmentów w projekcji 1-D.
    
    Funkcja analizuje tablicę projekcji (suma pikseli wzdłuż osi)
    i znajduje zakresy gdzie wartości są niezerowe.
    
    Argumenty:
        projection (np.ndarray): Jednowymiarowa tablica z wartościami projekcji.
                                 Typowo suma pikseli w wierszach lub kolumnach.
    
    Zwraca:
        list[tuple[int, int]]: Lista krotek (start, end) określających
                               granice segmentów. Indeksy są inkluzywne dla start
                               i ekskluzywne dla end.
    
    Przykład:
        >>> proj = np.array([0, 0, 5, 8, 6, 0, 0, 3, 4, 0])
        >>> _find_bounds(proj)
        [(2, 5), (7, 9)]
    """
    bounds = []
    in_seg = False
    for i, val in enumerate(projection):
        if val > 0 and not in_seg:
            start = i
            in_seg = True
        elif val == 0 and in_seg:
            bounds.append((start, i))
            in_seg = False
    if in_seg:
        bounds.append((start, len(projection)))
    return bounds


def _classify_letter(letter_gray: np.ndarray, model: nn.Module, device: torch.device, more) -> str:
    """
    Klasyfikuje pojedynczy wycięty fragment obrazu jako znak.
    
    Funkcja przetwarza obraz litery, przepuszcza przez model CNN
    i zwraca rozpoznany znak.
    
    Argumenty:
        letter_gray (np.ndarray): Obraz litery w skali szarości.
        model (nn.Module): Wytrenowany model CNN.
        device (torch.device): Urządzenie (CPU/CUDA) do obliczeń.
        more: Flaga określająca format zwracanych danych:
              - Jeśli True (lub 1): zwraca (znak, pewność%, tensor prawdopodobieństw)
              - Jeśli False (lub 0): zwraca tylko znak
    
    Zwraca:
        str | tuple: Rozpoznany znak lub krotka (znak, pewność, probs) 
                     w zależności od parametru 'more'.
    
    Przykład:
        >>> char = _classify_letter(letter_img, model, device, False)
        >>> char, conf, probs = _classify_letter(letter_img, model, device, True)
    """
    pil = Image.fromarray(letter_gray).resize((28, 28)).convert("L")
    tensor = get_transform()(pil).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1)
    confidence, predicted = torch.max(probs, 1)
    if(more):
        return CHARS[predicted.item()], confidence.item() * 100, probs[0]
    else:
        return CHARS[predicted.item()]


def _mean_per_class(class_conf_samples: dict[str, list[float]]) -> dict[str, float]:
    """Liczy średni poziom pewności dla każdej klasy (litery)."""
    class_avg: dict[str, float] = {}
    for cls, samples in class_conf_samples.items():
        if samples:
            class_avg[cls] = float(np.mean(samples))
    return class_avg


# ── Predykcja pojedynczej litery ───────────────────────────────────────────────

def predict_image(
    image_path: str,
    model: nn.Module,
    device: torch.device,
    args,
) -> tuple[str, float, torch.Tensor]:
    """
    Rozpoznaje pojedynczy znak na zdjęciu.
    
    Funkcja ładuje obraz, przycina do bounding boxa znaku,
    przetwarza i klasyfikuje przy użyciu modelu CNN.
    
    Argumenty:
        image_path (str): Ścieżka do obrazu ze znakiem.
        model (nn.Module): Wytrenowany model CNN.
        device (torch.device): Urządzenie (CPU/CUDA) do obliczeń.
        args: Obiekt argparse.Namespace z parametrami odszumiania.
    
    Zwraca:
        tuple[str, float, torch.Tensor]: Krotka zawierająca:
            - predicted_char (str): Rozpoznany znak (np. "A")
            - confidence (float): Pewność predykcji w procentach (0-100)
            - probs (torch.Tensor): Tensor prawdopodobieństw dla wszystkich klas
    
    Przykład:
        >>> char, conf, probs = predict_image("letter.png", model, device, args)
        >>> print(f"Rozpoznano: {char} z pewnością {conf:.1f}%")
    """
    image = load_and_optionally_denoise(image_path, args, mode="L")
    img_array = np.array(image)
    
    model.eval()
    with torch.no_grad():

        rows = np.where(np.sum(img_array < 128, axis=1) > 0)[0]
        img_array = img_array[rows[0]:rows[-1] + 1, :]

        collums = np.where(np.sum(img_array < 128, axis=1) > 0)[0]
        img_array = img_array[collums[0]:collums[-1] + 1, :]
        
        img_array = preprocess_letter(img_array)

        letter = _classify_letter(img_array, model, device, 1)

    return letter



# ── Predykcja wyrazu (jedna linia) ────────────────────────────────────────────

def predict_word(
    image_path: str,
    model: nn.Module,
    device: torch.device,
    args,
) -> tuple[str, float, dict[str, float]]:
    """
    Segmentuje i rozpoznaje litery w jednej linii tekstu.
    
    Funkcja używa projekcji pionowej do wykrycia granic liter,
    następnie klasyfikuje każdą literę osobno i łączy wyniki w wyraz.
    
    Argumenty:
        image_path (str): Ścieżka do obrazu z wyrazem (jedna linia).
        model (nn.Module): Wytrenowany model CNN.
        device (torch.device): Urządzenie (CPU/CUDA) do obliczeń.
        args: Obiekt argparse.Namespace z parametrami odszumiania.
    
    Zwraca:
        str: Rozpoznany wyraz (ciąg znaków bez spacji).
    
    Przykład:
        >>> word = predict_word("hello.png", model, device, args)
        >>> print(word)
        'HELLO'
    
    Uwaga:
        Funkcja oczekuje obrazu z pojedynczą linią tekstu.
        Dla obrazów wieloliniowych użyj predict_segments().
    """
    image = load_and_optionally_denoise(image_path, args, mode="L")
    img_array = np.array(image)
    binary = img_array < 128

    vertical_sum = np.sum(binary, axis=0)
    letters_bounds = _find_bounds(vertical_sum)

    word = ""
    letter_confidences: list[float] = []
    class_conf_samples: dict[str, list[float]] = {}
    model.eval()
    with torch.no_grad():
        for start, end in letters_bounds:
            letter_img = img_array[:, start:end]

            rows = np.where(np.sum(letter_img < 128, axis=1) > 0)[0]
            if len(rows) == 0:
                continue
            
            letter_img = letter_img[rows[0]:rows[-1] + 1, :]

            letter_img = preprocess_letter(letter_img)

            predicted_char, confidence, _ = _classify_letter(letter_img, model, device, 1)
            word += predicted_char
            letter_confidences.append(confidence)
            class_conf_samples.setdefault(predicted_char, []).append(confidence)

    avg_word_confidence = float(np.mean(letter_confidences)) if letter_confidences else 0.0
    class_confidence = _mean_per_class(class_conf_samples)
    return word, avg_word_confidence, class_confidence


# ── Predykcja wielu linii tekstu ───────────────────────────────────────────────

def predict_segments(
    image_path: str,
    model: nn.Module,
    device: torch.device,
    args,
) -> tuple[str, list[tuple[str, float]], dict[str, float]]:
    """
    Segmentuje i rozpoznaje tekst wieloliniowy.
    
    Algorytm:
      1. Projekcja pozioma wykrywa linie tekstu
      2. Dla każdej linii projekcja pionowa wykrywa litery
      3. Heurystyka spacji: przerwa > 1.5× średniej szerokości litery
      4. Każda litera jest klasyfikowana przez model CNN
    
    Argumenty:
        image_path (str): Ścieżka do obrazu z tekstem wieloliniowym.
        model (nn.Module): Wytrenowany model CNN.
        device (torch.device): Urządzenie (CPU/CUDA) do obliczeń.
        args: Obiekt argparse.Namespace z parametrami odszumiania.
    
    Zwraca:
        str: Rozpoznany tekst wieloliniowy (linie oddzielone '\\n').
    
    Przykład:
        >>> text = predict_segments("document.png", model, device, args)
        >>> print(text)
        'HELLO WORLD
         THIS IS TEXT'
    
    Uwaga:
        Funkcja automatycznie wykrywa spacje między wyrazami
        na podstawie odległości między literami.
    """
    image = load_and_optionally_denoise(image_path, args, mode="L")
    img_array = np.array(image)
    binary = img_array < 128

    rows_bounds = _find_bounds(np.sum(binary, axis=1))

    text = ""
    words_with_confidence: list[tuple[str, float]] = []
    class_conf_samples: dict[str, list[float]] = {}

    def _flush_word(word_chars: list[str], word_confs: list[float]) -> None:
        if not word_chars or not word_confs:
            return
        words_with_confidence.append(("".join(word_chars), float(np.mean(word_confs))))

    model.eval()
    with torch.no_grad():
        for row_start, row_end in rows_bounds:
            line_img = img_array[row_start:row_end, :]
            vertical_sum = np.sum(line_img < 128, axis=0)
            letters_bounds = _find_bounds(vertical_sum)

            # Heurystyka spacji: przerwa > 1.5× średniej szerokości litery
            widths = [e - s for s, e in letters_bounds]
            avg_width = float(np.mean(widths)) if widths else 0.0

            line_text = ""
            current_word_chars: list[str] = []
            current_word_confs: list[float] = []
            for idx, (start, end) in enumerate(letters_bounds):
                letter_img = line_img[:, start:end]

                rows = np.where(np.sum(letter_img < 128, axis=1) > 0)[0]
                if len(rows) == 0:
                    continue

                letter_img = letter_img[rows[0]:rows[-1] + 1, :]
                letter_img = preprocess_letter(letter_img)

                predicted_char, confidence, _ = _classify_letter(letter_img, model, device, 1)
                line_text += predicted_char
                current_word_chars.append(predicted_char)
                current_word_confs.append(confidence)
                class_conf_samples.setdefault(predicted_char, []).append(confidence)

                if idx < len(letters_bounds) - 1 and avg_width > 0:
                    gap = letters_bounds[idx + 1][0] - end
                    if gap > 1.5 * avg_width:
                        _flush_word(current_word_chars, current_word_confs)
                        current_word_chars = []
                        current_word_confs = []
                        line_text += " "

            _flush_word(current_word_chars, current_word_confs)
            text += line_text + "\n"

    class_confidence = _mean_per_class(class_conf_samples)
    return text, words_with_confidence, class_confidence
