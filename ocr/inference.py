"""
Moduł rozpoznawania tekstu (inference).

Odpowiedzialności:
  - wczytywanie wytrenowanego modelu z dysku
  - predykcja pojedynczej litery
  - segmentacja i rozpoznawanie wyrazu (jedna linia)
  - segmentacja i rozpoznawanie wielu linii tekstu
"""

import os
import re
import cv2
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

ACTIVE_CHARS = list(CHARS)


def _map_chars74k_sample_to_char(sample_name: str) -> str:
    """Mapuje nazwę SampleXXX z Chars74K na znak (A-Z, a-z, 0-9) gdy to możliwe."""
    match = re.fullmatch(r"Sample(\d+)", sample_name)
    if not match:
        return sample_name

    idx = int(match.group(1))
    if 1 <= idx <= 10:
        return str(idx - 1)
    if 11 <= idx <= 36:
        return chr(ord("A") + (idx - 11))
    if 37 <= idx <= 62:
        return chr(ord("a") + (idx - 37))
    return sample_name


def get_active_chars() -> list[str]:
    """Zwraca aktualne mapowanie indeks->znak używane przez model."""
    return list(ACTIVE_CHARS)


def _set_active_chars(checkpoint: object | None = None) -> None:
    """Ustawia mapowanie indeks->znak na podstawie checkpointa lub domyślnej konfiguracji."""
    global ACTIVE_CHARS
    if isinstance(checkpoint, dict) and "class_names" in checkpoint:
        class_names = checkpoint.get("class_names")
        if isinstance(class_names, list) and class_names:
            ACTIVE_CHARS = [_map_chars74k_sample_to_char(str(name)) for name in class_names]
            print(f"Wczytano mapowanie klas z checkpointa ({len(ACTIVE_CHARS)} klas)")
            return

    ACTIVE_CHARS = list(CHARS)


def _label_for_idx(idx: int) -> str:
    if 0 <= idx < len(ACTIVE_CHARS):
        return ACTIVE_CHARS[idx]
    return f"<UNK:{idx}>"

print(matplotlib.get_backend())

def load_model(model_path: str = MODEL_PATH, device: torch.device = None) -> nn.Module:
    """
    Wczytuje wytrenowany model SimpleCNN.
    Obsługuje:
    - czysty state_dict (stary format)
    - checkpoint (nowy format)
    Automatycznie wykrywa liczbę klas z checkpointa.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(model_path):
        print(f"Wczytywanie modelu z {model_path}...")

        checkpoint = torch.load(model_path, map_location=device)
        _set_active_chars(checkpoint)

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
        if not any(len(ch) == 1 and ch.islower() for ch in ACTIVE_CHARS):
            print("[UWAGA] Aktualny model nie zawiera małych liter jako klas.")
            print("        Aby wykrywać a-z, wytrenuj model na datasecie z 52 klasami.")
        print("Model wczytany!")
    else:
        print(f"UWAGA: Nie znaleziono modelu {model_path}")
        print("Model nie jest wytrenowany - wyniki będą losowe!")
        print("Najpierw uruchom: python run_local.py --train")
        _set_active_chars(None)
        model = SimpleCNN(num_classes=NUM_CLASSES)

    model.to(device)
    model.eval()
    return model


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


def _tight_crop(gray: np.ndarray) -> np.ndarray:
    """Przycina obraz do obszaru zawierającego piksele znaku."""
    rows = np.where(np.sum(gray < 128, axis=1) > 0)[0]
    cols = np.where(np.sum(gray < 128, axis=0) > 0)[0]
    if len(rows) == 0 or len(cols) == 0:
        return gray
    return gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _watershed_split_component(
    gray_roi: np.ndarray,
    component_mask: np.ndarray,
    fg_ratio: float,
    min_box_w: int,
    min_box_h: int,
    min_box_area: int,
) -> list[tuple[int, int, int, int]]:
    """Dzieli pojedynczy zlepiony komponent na litery metodą watershed."""
    dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []

    _, sure_fg = cv2.threshold(dist, fg_ratio * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    # Za mało ziaren - najpewniej pojedyncza litera.
    num_labels, markers = cv2.connectedComponents(sure_fg)
    if num_labels <= 2:
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sure_bg = cv2.dilate(component_mask, kernel, iterations=1)
    unknown = cv2.subtract(sure_bg, sure_fg)

    markers = markers + 1
    markers[unknown == 255] = 0

    color = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)

    boxes: list[tuple[int, int, int, int]] = []
    for label in np.unique(markers):
        if label <= 1:
            continue

        ys, xs = np.where(markers == label)
        if xs.size == 0:
            continue

        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        w, h = x2 - x1, y2 - y1

        if w < min_box_w or h < min_box_h or (w * h) < min_box_area:
            continue

        boxes.append((x1, y1, x2, y2))

    boxes.sort(key=lambda b: b[0])
    return boxes


def _segment_letters(gray: np.ndarray, args=None) -> list[tuple[int, int, int, int]]:
    """
    Segmentuje litery bez opierania się na pustych przerwach pionowych.
    Najpierw CC, a szerokie komponenty próbuje dzielić watershed.
    """
    ws_fg_ratio = float(getattr(args, "ws_fg_ratio", 0.45))
    ws_split_aspect = float(getattr(args, "ws_split_aspect", 1.15))
    ws_min_comp_area = int(getattr(args, "ws_min_comp_area", 30))
    ws_split_min_area = int(getattr(args, "ws_split_min_area", 250))
    ws_min_box_w = int(getattr(args, "ws_min_box_w", 3))
    ws_min_box_h = int(getattr(args, "ws_min_box_h", 5))
    ws_min_box_area = int(getattr(args, "ws_min_box_area", 20))

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        if area < ws_min_comp_area or w < ws_min_box_w or h < ws_min_box_h:
            continue

        local_labels = labels[y:y + h, x:x + w]
        component_mask = np.uint8(local_labels == label) * 255

        # Szeroki komponent traktujemy jako potencjalnie sklejone litery.
        should_try_split = w > int(ws_split_aspect * h) and area > ws_split_min_area
        if should_try_split:
            gray_roi = gray[y:y + h, x:x + w]
            split_boxes = _watershed_split_component(
                gray_roi,
                component_mask,
                fg_ratio=ws_fg_ratio,
                min_box_w=ws_min_box_w,
                min_box_h=ws_min_box_h,
                min_box_area=ws_min_box_area,
            )
            if split_boxes:
                for sx1, sy1, sx2, sy2 in split_boxes:
                    boxes.append((x + sx1, y + sy1, x + sx2, y + sy2))
                continue

        boxes.append((x, y, x + w, y + h))

    boxes.sort(key=lambda b: b[0])
    return boxes


def _classify_letter(letter_gray: np.ndarray, model: nn.Module, device: torch.device, more) -> str:
    """Klasyfikuje wycięty fragment (tablica grayscale) jako znak."""
##    letter_img = preprocess_letter(letter_gray)
    pil = Image.fromarray(letter_gray).resize((28, 28)).convert("L")
    tensor = get_transform()(pil).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1)
    confidence, predicted = torch.max(probs, 1)
    if(more):
        return _label_for_idx(predicted.item()), confidence.item() * 100, probs[0]
    else:
        return _label_for_idx(predicted.item())


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
    Rozpoznaje znak na zdjęciu.

    Returns:
        (predicted_char, confidence_percent, all_probs_tensor)
    """
    image = load_and_optionally_denoise(image_path, args, mode="L")
    img_array = np.array(image)
    
    model.eval()
    with torch.no_grad():
        img_array = _tight_crop(img_array)
        
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
    Segmentuje litery w jednej linii z użyciem CC + watershed
    i rozpoznaje każdą z nich.
    """
    image = load_and_optionally_denoise(image_path, args, mode="L")
    img_array = np.array(image)
    letter_boxes = _segment_letters(img_array, args=args)

    word = ""
    letter_confidences: list[float] = []
    class_conf_samples: dict[str, list[float]] = {}
    model.eval()
    with torch.no_grad():
        for x1, y1, x2, y2 in letter_boxes:
            letter_img = img_array[y1:y2, x1:x2]
            letter_img = _tight_crop(letter_img)
            if letter_img.size == 0:
                continue

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
    Segmentuje linie (projekcja pozioma), a litery w linii przez CC + watershed.
    Spacje między wyrazami wykrywa heurystyką większej przerwy między literami.
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
            letter_boxes = _segment_letters(line_img, args=args)

            # Heurystyka spacji: przerwa > 1.5× średniej szerokości litery
            widths = [x2 - x1 for x1, _, x2, _ in letter_boxes]
            avg_width = float(np.mean(widths)) if widths else 0.0

            line_text = ""
            current_word_chars: list[str] = []
            current_word_confs: list[float] = []
            for idx, (x1, y1, x2, y2) in enumerate(letter_boxes):
                letter_img = line_img[y1:y2, x1:x2]
                letter_img = _tight_crop(letter_img)
                if letter_img.size == 0:
                    continue

                letter_img = preprocess_letter(letter_img)

                predicted_char, confidence, _ = _classify_letter(letter_img, model, device, 1)
                line_text += predicted_char
                current_word_chars.append(predicted_char)
                current_word_confs.append(confidence)
                class_conf_samples.setdefault(predicted_char, []).append(confidence)

                if idx < len(letter_boxes) - 1 and avg_width > 0:
                    next_x1 = letter_boxes[idx + 1][0]
                    gap = next_x1 - x2
                    if gap > 1.5 * avg_width:
                        _flush_word(current_word_chars, current_word_confs)
                        current_word_chars = []
                        current_word_confs = []
                        line_text += " "

            _flush_word(current_word_chars, current_word_confs)
            text += line_text + "\n"

    class_confidence = _mean_per_class(class_conf_samples)
    return text, words_with_confidence, class_confidence
