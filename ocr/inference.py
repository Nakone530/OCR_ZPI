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
from .config import CHARS, MODEL_PATH
from .model import SimpleCNN
from .utils import get_transform, load_and_optionally_denoise, preprocess_letter, save_image_to_temp_folder

# ── Ładowanie modelu ───────────────────────────────────────────────────────────

print(matplotlib.get_backend())

def load_model(model_path: str = MODEL_PATH, device: torch.device = None) -> nn.Module:
    """
    Wczytuje wytrenowany model SimpleCNN.
    Jeśli plik nie istnieje, model działa na losowych wagach.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN(num_classes=46)

    if os.path.exists(model_path):
        print(f"Wczytywanie modelu z {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model wczytany!")
    else:
        print(f"UWAGA: Nie znaleziono modelu {model_path}")
        print("Model nie jest wytrenowany - wyniki będą losowe!")
        print("Najpierw uruchom: python run_local.py --train")

    model.to(device)
    model.eval()
    return model


# ── Predykcja pojedynczej litery ───────────────────────────────────────────────

def predict_image(
    image_path: str,
    model: nn.Module,
    device: torch.device,
    args,
) -> tuple[str, float, torch.Tensor]:
    """
    Rozpoznaje znak na zdjęciu.

    Returns:
        (predicted_char, confidence_percent, all_probs_tensor)
    """
    image = load_and_optionally_denoise(image_path, args, mode="L")

    preprocess_letter(image)

    transform = get_transform()
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        confidence, predicted = torch.max(probs, 1)

    return CHARS[predicted.item()], confidence.item() * 100, probs[0]


# ── Segmentacja pomocnicza ─────────────────────────────────────────────────────

def _find_bounds(projection: np.ndarray) -> list[tuple[int, int]]:
    """Wykrywa granice (start, end) niepustych segmentów w projekcji 1-D."""
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


def _classify_letter(letter_gray: np.ndarray, model: nn.Module, device: torch.device) -> str:
    """Klasyfikuje wycięty fragment (tablica grayscale) jako znak."""
    letter_img = preprocess_letter(letter_gray)
    pil = Image.fromarray(letter_gray).resize((28, 28)).convert("L")
    tensor = get_transform()(pil).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1)
    _, predicted = torch.max(probs, 1)
    return CHARS[predicted.item()]


# ── Predykcja wyrazu (jedna linia) ────────────────────────────────────────────

def predict_word(
    image_path: str,
    model: nn.Module,
    device: torch.device,
    args,
) -> str:
    """
    Segmentuje litery w jednej linii metodą projekcji pionowej
    i rozpoznaje każdą z nich.
    """
    image = load_and_optionally_denoise(image_path, args, mode="L")
    img_array = np.array(image)
    binary = img_array < 128

    vertical_sum = np.sum(binary, axis=0)
    letters_bounds = _find_bounds(vertical_sum)

    word = ""
    model.eval()
    with torch.no_grad():
        for start, end in letters_bounds:
            letter_img = img_array[:, start:end]

            rows = np.where(np.sum(letter_img < 128, axis=1) > 0)[0]
            if len(rows) == 0:
                continue


            preprocess_letter(letter_img)

            word += _classify_letter(letter_img, model, device)

    return word


# ── Predykcja wielu linii tekstu ───────────────────────────────────────────────

def predict_segments(
    image_path: str,
    model: nn.Module,
    device: torch.device,
    args,
) -> str:
    """
    Segmentuje linie (projekcja pozioma), a wewnątrz każdej linii litery
    (projekcja pionowa). Wykrywa spacje między wyrazami na podstawie przerw.
    """
    image = load_and_optionally_denoise(image_path, args, mode="L")

    img_array = np.array(image)
    binary = img_array < 128

    rows_bounds = _find_bounds(np.sum(binary, axis=1))

    text = ""
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
            for idx, (start, end) in enumerate(letters_bounds):
                letter_img = line_img[:, start:end]

                rows = np.where(np.sum(letter_img < 128, axis=1) > 0)[0]
                if len(rows) == 0:
                    continue
                letter_img = letter_img[rows[0]:rows[-1] + 1, :]

                #save_image_to_temp_folder(letter_img, "pre")
                letter_img = preprocess_letter(letter_img)


                #plt.imshow(letter_img, cmap='gray')
                #plt.axis('off')
                #plt.show()

                #save_image_to_temp_folder(letter_img, "post")
                line_text += _classify_letter(letter_img, model, device)

                if idx < len(letters_bounds) - 1 and avg_width > 0:
                    gap = letters_bounds[idx + 1][0] - end
                    if gap > 1.5 * avg_width:
                        line_text += " "

            text += line_text + "\n"

    return text
