"""
Moduł rozpoznawania tekstu (inference).

Odpowiedzialności:
  - wczytywanie wytrenowanego modelu z dysku
  - predykcja pojedynczej litery
  - segmentacja i rozpoznawanie wyrazu (jedna linia)
  - segmentacja i rozpoznawanie wielu linii tekstu
"""

import difflib
import os
import re
import json
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



from PIL import Image

from .config import CHARS, MODEL_PATH, NUM_CLASSES, char2idx, idx2char
from .utils import (
    get_inf_transform,
    load_and_optionally_denoise,
    preprocess_letter,
    save_image_to_temp_folder,
    save_image_to_today_folder,
    list_models,
    select_models,
    load_models,
    aggregate,
    get_active_chars,
    _set_active_chars,
    _label_for_idx,
    load_model,
    generate_model_ensembles,
    load_cache,
    resolve_cache_path,
    version_str,
    to_serializable,
    aux_transform,
    select_version,
)
from .display import visualize_prediction, show_image
from . import info




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


def _tight_crop(gray: np.ndarray) -> np.ndarray:
    """Przycina obraz do obszaru zawierającego piksele znaku."""
    rows = np.where(np.sum(gray < 128, axis=1) > 0)[0]
    cols = np.where(np.sum(gray < 128, axis=0) > 0)[0]
    if len(rows) == 0 or len(cols) == 0:
        return gray
    return gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _merge_fragmented_boxes(
    boxes: list[tuple[int, int, int, int]],
    max_gap: int = 4,
    max_height_ratio: float = 1.8,
    max_vertical_distance: int = 4,
) -> list[tuple[int, int, int, int]]:
    """
    Scala fragmenty litery, które są obok siebie.
    
    Poprzez analizę boksów uważa za fragmenty tego samego znaku boxy które:
    - Są poziomo sąsiadujące (przerwa <= max_gap)
    - Są pionowo wyrównane (zakresy Y się nachodzą lub są blisko)
    - Mają podobne wysokości (ratio <= max_height_ratio)
    
    Argumenty:
        boxes: Lista boksów (x1, y1, x2, y2)
        max_gap: Maksymalna przerwa pozioma między fragmentami (px)
        max_height_ratio: Maksymalny stosunek wysokości między fragmentami
        max_vertical_distance: Maksymalna odległość pionowa do scalenia (px)
    """
    if len(boxes) <= 1:
        return boxes
    
    merged = []
    current_group = [boxes[0]]
    
    for i in range(1, len(boxes)):
        prev_x1, prev_y1, prev_x2, prev_y2 = current_group[-1]
        curr_x1, curr_y1, curr_x2, curr_y2 = boxes[i]
        
        prev_h = prev_y2 - prev_y1
        curr_h = curr_y2 - curr_y1
        
        # Sprawdź czy boxy są sąsiadujące poziomo
        horizontal_gap = curr_x1 - prev_x2
        
        # Sprawdź czy boxy są wyrównane pionowo
        y_overlap = max(0, min(prev_y2, curr_y2) - max(prev_y1, curr_y1))
        vertical_distance = max(0, max(prev_y1, curr_y1) - min(prev_y2, curr_y2))
        
        # Sprawdzenie czy boxy mają podobną wysokość
        height_ratio = max(prev_h, curr_h) / min(prev_h, curr_h) if min(prev_h, curr_h) > 0 else 1.0
        
        # Warunki scalenia - bardziej liberalne
        # Scalaj jeśli: przerwa jest mała LUB boxy się nachodzą pionowo
        should_merge = (
            horizontal_gap <= max_gap and
            (y_overlap > 0 or vertical_distance <= max_vertical_distance) and
            height_ratio <= max_height_ratio
        )
        
        if should_merge:
            current_group.append(boxes[i])
        else:
            # Scal grupę i dodaj do wyników
            if current_group:
                x1 = min(b[0] for b in current_group)
                y1 = min(b[1] for b in current_group)
                x2 = max(b[2] for b in current_group)
                y2 = max(b[3] for b in current_group)
                merged.append((x1, y1, x2, y2))
            current_group = [boxes[i]]
    
    # Dodaj ostatnią grupę
    if current_group:
        x1 = min(b[0] for b in current_group)
        y1 = min(b[1] for b in current_group)
        x2 = max(b[2] for b in current_group)
        y2 = max(b[3] for b in current_group)
        merged.append((x1, y1, x2, y2))
    
    return merged


def _watershed_split_component(
    gray_roi: np.ndarray,
    component_mask: np.ndarray,
    fg_ratio: float,
    min_box_w: int,
    min_box_h: int,
    min_box_area: int,
    args=None,
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
    
    # Scal fragmenty litery które znajdują się obok siebie
    ws_merge_gap = int(getattr(args, "ws_merge_gap", 4)) if args else 4
    ws_merge_height_ratio = float(getattr(args, "ws_merge_height_ratio", 1.8)) if args else 1.8
    ws_merge_vert_dist = int(getattr(args, "ws_merge_vert_dist", 4)) if args else 4
    boxes = _merge_fragmented_boxes(
        boxes,
        max_gap=ws_merge_gap,
        max_height_ratio=ws_merge_height_ratio,
        max_vertical_distance=ws_merge_vert_dist
    )
    
    return boxes


def _segment_letters(gray: np.ndarray, args=None) -> list[tuple[int, int, int, int]]:
    """
    Segmentuje litery bez opierania się na pustych przerwach pionowych.
    Najpierw CC, a szerokie komponenty próbuje dzielić watershed.
    """

    # Parametry używane do wykrywania i dzielenia komponentów.
    # Pozwalają na nadpisanie przez `args` (np. z CLI) lub użycie rozsądnych domyślnych.
    ws_min_comp_area = int(getattr(args, "ws_min_comp_area", 20))
    ws_min_box_w = int(getattr(args, "ws_min_box_w", 4))
    ws_min_box_h = int(getattr(args, "ws_min_box_h", 6))
    ws_split_aspect = float(getattr(args, "ws_split_aspect", 3.0))
    ws_split_min_area = int(getattr(args, "ws_split_min_area", 150))
    ws_fg_ratio = float(getattr(args, "ws_fg_ratio", 0.4))
    ws_min_box_area = int(getattr(args, "ws_min_box_area", 50))

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
                args=args,
            )
            if split_boxes:
                for sx1, sy1, sx2, sy2 in split_boxes:
                    boxes.append((x + sx1, y + sy1, x + sx2, y + sy2))
                continue

        boxes.append((x, y, x + w, y + h))

    boxes.sort(key=lambda b: b[0])
    return boxes


def _classify_letter(letter_gray: np.ndarray, model: nn.Module, device: torch.device, more, args) -> str:
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
    tensor = aux_transform()(pil).unsqueeze(0).to(device)
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


def _is_debug_enabled(args) -> bool:
    """Sprawdza, czy aktywny jest tryb debug (z fallbackiem do legacy --quiet)."""
    debug = bool(getattr(args, "debug", False))
    quiet = bool(getattr(args, "quiet", False))
    return debug and not quiet


def _format_topk_probs(probs: torch.Tensor, k: int = 3) -> str:
    """Formatuje Top-K predykcji jako krótki tekst do logów debug."""
    top_probs, top_indices = torch.topk(probs, k)
    parts = []
    for idx, prob in zip(top_indices, top_probs):
        label = _label_for_idx(int(idx.item()))
        parts.append(f"'{label}': {prob.item() * 100:.1f}%")
    return ", ".join(parts)


def _finalize_debug_crops(
    debug_crops: list[tuple[np.ndarray, str]],
    args,
    source_image_path: str,
    mode_tag: str,
) -> None:
    """Pokazuje i/lub zapisuje wycinki 28x28 podawane do modelu."""
    if not debug_crops:
        return

    show_crops = bool(getattr(args, "debug_show_crops", False))
    save_crops_dir = getattr(args, "debug_save_crops", None)
    if not show_crops and save_crops_dir is None:
        return

    source_stem = Path(source_image_path).stem

    if save_crops_dir is not None:
        if save_crops_dir == "auto":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path("outputs") / "debug_crops" / f"{source_stem}_{mode_tag}_{timestamp}"
        else:
            save_dir = Path(save_crops_dir)

        save_dir.mkdir(parents=True, exist_ok=True)
        for idx, (crop_img, caption) in enumerate(debug_crops, start=1):
            safe_caption = re.sub(r"[^A-Za-z0-9._-]", "_", caption)[:40]
            file_name = f"{mode_tag}_{idx:03d}_{safe_caption}.png"
            out_path = save_dir / file_name
            cv2.imwrite(str(out_path), crop_img)
        info(f"[DEBUG] Zapisano {len(debug_crops)} wycinków do: {save_dir}")

    if show_crops:
        n = len(debug_crops)
        cols = min(8, max(1, n))
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows))
        axes = np.array(axes).reshape(-1)

        for ax in axes:
            ax.axis("off")

        for idx, (crop_img, caption) in enumerate(debug_crops):
            axes[idx].imshow(crop_img, cmap="gray")
            axes[idx].set_title(caption, fontsize=8)
            axes[idx].axis("off")

        fig.suptitle(f"Debug crops: {source_stem} [{mode_tag}]", fontsize=12)
        fig.tight_layout()
        plt.show()


def _load_image_array(image_path: str, args, mode: str = "L") -> np.ndarray:
    """Wczytuje obraz z odszumianiem i zwraca jako numpy array.

    Centralizuje powtarzane wywołanie `load_and_optionally_denoise(...); np.array(...)`.
    Zwraca obraz w postaci 2D (skala szarości) jeśli `mode=='L'`, w przeciwnym wypadku
    zwraca surową tablicę obrazu.
    """
    image = load_and_optionally_denoise(image_path, args, mode=mode)
    return np.array(image)


# ── Przekazanie zdjęć folderu do predykcji ───────────────────────────────────────────────


def process_folder(folder_path, args, models_dir, device, info):

    if not os.path.isdir(folder_path):
        raise ValueError(f"To nie jest katalog: {folder_path}")
    
    models_dir = os.path.dirname(models_dir)
    version = select_version()
    models = list_models(models_dir, version)
    default_model, selected_models = select_models(models)

    loaded_models = load_models(models_dir, selected_models, device, info)

    bbox_data = _load_bbox_data(folder_path)
    bbox_index = 0

    results = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".png"):
            continue

        file_path = os.path.join(folder_path, filename)

        try:
            info(f"\nRozpoznawanie: {file_path}")

            per_model = predict_word_crnn_multi(
                file_path,
                loaded_models,
                device,
                args,
                info
            )

            final_text = aggregate(per_model, default_model)

            best_conf = max(
                r["confidence"] for r in per_model.values()
            )


            bbox = None
            if bbox_data and bbox_index < len(bbox_data):
                bbox = bbox_data[bbox_index].get("bbox")
            bbox_index += 1

            results.append({
                "file": file_path,
                "text": final_text,
                "confidence": best_conf,
                "bbox": bbox,
                "per_model": per_model
            })

        except Exception as e:
            results.append({
                "file": file_path,
                "error": str(e)
            })

    return results


def _load_bbox_data(folder_path):
    """Ładuje dane bounding box z pliku boxes.jsonl."""
    jsonl_path = os.path.join(folder_path, "boxes.jsonl")
    if not os.path.exists(jsonl_path):
        return None
    
    bbox_data = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                bbox_xyxy = row.get("bbox_xyxy", [])
                if len(bbox_xyxy) == 4:
                    bbox_data.append({
                        "bbox": bbox_xyxy,
                    })
    except:
        return None
    
    return bbox_data if bbox_data else None

def predict_word_crnn_multi(image_path, models, device, args, info):
    results = {}

    for name, model in models.items():
        res = predict_word_crnn(image_path, model, device, args, info)
        results[name] = res

    return results


#--- Testowanie wielu kombinacji modeli
def run_ensemble_generation(models_dir):
    

    models_dir = os.path.dirname(models_dir)
    models = list_models(models_dir)
    min_mn = int(input("Podaj min mn: ").strip())
    max_mn = int(input("Podaj max mn: ").strip())

    if min_mn > max_mn:
        raise ValueError("min_mn nie może być większe niż max_mn")

    mode = int(input("Wybierz rodzaj selekcji (1-kombinacje, 2-permutacje)"))
    if mode == 1 :
        n = sum(math.comb(len(models), r) for r in range(min_mn, max_mn + 1))
    elif mode == 2 :
        n = sum(math.perm(len(models), r) for r in range(min_mn, max_mn + 1))
    else :
        info("zły wybór, wybranie default -- kombinacje")

    return generate_model_ensembles(models, min_mn, max_mn, mode), n

def test_models(folder_path, args, models_dir, device, info, ensembles, mn):

    if not os.path.isdir(folder_path):
        raise ValueError(f"To nie jest katalog: {folder_path}")
    
    models_dir = os.path.dirname(models_dir)
    models = list_models(models_dir)

    loaded_models = load_models(models_dir, models, device, info)


    bbox_data = _load_bbox_data(folder_path)
    bbox_index = 0

    results = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".png"):
            continue
    
        file_path = os.path.join(folder_path, filename)
    
        try:
            info(f"\nRozpoznawanie: {file_path}")

            ensemble_results = run_ensembles_inference(
                file_path,
                ensembles,
                loaded_models,
                device,
                args,
                info,
                mn
            )

            bbox = None
            if bbox_data and bbox_index < len(bbox_data):
                bbox = bbox_data[bbox_index].get("bbox")
            bbox_index += 1

            results.append({
                "file": file_path,
                "ensembles": ensemble_results,
                "bbox": bbox
            })
        except Exception as e:
            results.append({
                "file": file_path,
                "error": str(e)
            })
    return results



def test_cache_models(folder_path, args, models_dir, device, info, ensembles, mn, cache_path="./cache"):

    if not os.path.isdir(folder_path):
        raise ValueError(f"To nie jest katalog: {folder_path}")
    
    models_dir = os.path.dirname(models_dir)
    models = list_models(models_dir)
    loaded_models = load_models(models_dir, models, device, info)


    bbox_data = _load_bbox_data(folder_path)
    bbox_index = 0

    results = []
    filename = "1"
    file_path = os.path.join(folder_path, filename)
    file_num = 0
    try:
        info(f"\nRozpoznawanie: {file_path}")
        if cache_path == "./cache" :
            cache = build_cache(
            args.ensemble,
            loaded_models,
            device,
            args,
            info
        )
        else :
            cache = load_cache(cache_path)
                
        ensemble_results = run_ensembles_cache(cache, ensembles, mn)

        bbox = None
        if bbox_data and bbox_index < len(bbox_data):
            bbox = bbox_data[bbox_index].get("bbox")
        bbox_index += 1
        for e in ensemble_results:
            results.append({
                "file": e["file"],
                "ensembles": e["ensembles"],
                "bbox": bbox
            })

    except Exception as e:
        results.append({
            "file": file_path,
            "error": str(e)
        })
    
    file_num = file_num +1
    return results


def run_ensembles_inference(file_path, ensembles, loaded_models, device, args, info, mn):
    ensemble_outputs = []

    for ensemble in ensembles:
        info(f"Ensemble: {ensemble}")

        subset = {m: loaded_models[m] for m in ensemble}

        per_model = predict_word_crnn_multi(
            file_path,
            subset,
            device,
            args,
            info
        )

        # używamy pierwszego modelu jako default (albo możesz zmienić logikę)
        default_model = ensemble[0]

        final_text = aggregate(per_model, default_model)

        best_conf = max(r["confidence"] for r in per_model.values())

        ensemble_outputs.append({
            "ensemble": ensemble,
            "text": final_text,
            "confidence": best_conf,
            "per_model": per_model
        })

    return ensemble_outputs


def run_ensembles_cache(cache, ensembles, mn):
    results = {name: [] for name in cache.keys()}
    n = list_models(os.path.dirname(MODEL_PATH))
    total_files = len(cache)
    total_steps = total_files * mn
    step = 0

    for ensemble in ensembles: 
        for name, full_per_model in cache.items():
            step_comp = math.ceil((step/total_steps) * 10000)/100
            info(f"Progress: {step_comp:.2f}% Ensemble: {ensemble},")
            
            subset = {m: full_per_model[m] for m in ensemble}

            default_model = max(
                subset.items(),
                key=lambda x: x[1]["confidence"]
            )[0]

            final_text = aggregate(subset, default_model)
            best_conf = max(r["confidence"] for r in subset.values())

            results[name].append({
                "ensemble": ensemble,
                "text": final_text,
                "confidence": best_conf
            })

            step += 1
    return [
        {"file": name, "ensembles": ens}
        for name, ens in results.items()
    ]


#--- budowanie cache---
def build_cache(folder_path, loaded_models, device, args, info):
    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = resolve_cache_path()
    cache = {}

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".png"):
            continue

        file_path = os.path.join(folder_path, filename)
        name, _ = os.path.splitext(filename)

        info(f"CACHE: {file_path}")

        per_model = predict_word_crnn_multi(
            file_path,
            loaded_models,
            device,
            args,
            info
        )

        cache[name] = to_serializable(per_model)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    return cache
# ── Predykcja wyrazu modelem CTC ───────────────────────────────────────────────
def predict_word_crnn(
    image_path: str,
    model: nn.Module,
    device: torch.device,
    args,
    info=None,
) -> tuple[str, float, torch.Tensor]:


    """
        funkcja rozpoznaje wyraz słowa
    """
    img_array = _load_image_array(image_path, args, mode="L")
    if info is None:
        info = globals().get("info")
    debug = _is_debug_enabled(args)
    #if(debug):
        #show_image(image, "przed crop")
    
    debug_crops: list[tuple[np.ndarray, str]] = []

    model.eval()
    with torch.no_grad():
        original_shape = img_array.shape
        cropped_shape = img_array.shape
        pil = Image.fromarray(img_array).convert("L")
        
        if(debug):
            #show_image(pil, "po crop", True)
            img_before = pil
            
        tensor = get_inf_transform(args)(pil).unsqueeze(0).to(device)
        
        if(debug):
            show_before_after(img_before, tensor)
            
        preprocessed_shape = tensor.shape

        outputs = model(tensor)  # (T, B, C)
        log_probs = outputs.log_softmax(2)
        probs = log_probs.exp()

        preds = log_probs.argmax(2)[:, 0].cpu().numpy()

        chars = []
        confidences = []

        prev = 0  # blank

        for t in range(len(preds)):
            p = preds[t]

            if p != prev and p != 0:
                chars.append(idx2char[p])

                confidences.append(probs[t, 0, p].item())

            prev = p

        text = "".join(chars)
        confidence = float(np.mean(confidences) * 100) if confidences else 0.0

        # dla kompatybilności: zwracamy probs z pierwszego kroku
        probs_out = probs[0, 0]

    if debug:
        info(
            "[DEBUG][image] kształty obrazu: "
            f"oryginał={original_shape}, po_crop={cropped_shape}, "
            f"po_preprocess={preprocessed_shape}"
        )
        info(
            f"[DEBUG][image] predykcja: '{text}' ({confidence:.1f}%)"
        )

        debug_crops.append((img_array.copy(), f"{text}_{confidence:.1f}"))
        _finalize_debug_crops(debug_crops, args, image_path, mode_tag="image")

    return {"text": text, "confidence": confidence, "per_char_confidences": confidences, "probs": probs_out}


# ── Predykcja pojedynczej litery ──────────────────────────────────────────────

def predict_letter(
    image_path: str,
    model: nn.Module,
    device: torch.device,
    args,
    info=None,
) -> tuple[str, float, torch.Tensor]:
    """
    Rozpoznaje pojedynczą literę na zdjęciu.
    
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
        >>> char, conf, probs = predict_letter("letter.png", model, device, args)
        >>> print(f"Rozpoznano: {char} z pewnością {conf:.1f}%")
    """
    img_array = _load_image_array(image_path, args, mode="L")
    if info is None:
        info = globals().get("info")
    debug = _is_debug_enabled(args)
    
    debug_crops: list[tuple[np.ndarray, str]] = []

    model.eval()
    with torch.no_grad():
        original_shape = img_array.shape
        # przytnij do tight crop przed klasyfikacją pojedynczej litery
        gray = img_array if img_array.ndim == 2 else np.array(Image.fromarray(img_array).convert("L"))
        cropped = _tight_crop(gray)
        cropped_shape = cropped.shape
        preprocessed = preprocess_letter(cropped)
        preprocessed_shape = preprocessed.shape

        letter = _classify_letter(preprocessed, model, device, 1, args)

    if debug:
        predicted_char, confidence, probs = letter
        debug_crops.append((img_array.copy(), f"1_{predicted_char}_{confidence:.1f}"))
        info(
            "[DEBUG][image] kształty obrazu: "
            f"oryginał={original_shape}, po_crop={cropped_shape}, "
            f"po_preprocess={preprocessed_shape}"
        )
        info(
            f"[DEBUG][image] klasyfikacja: '{predicted_char}' ({confidence:.1f}%), "
            f"top3: {_format_topk_probs(probs, k=3)}"
        )
        _finalize_debug_crops(debug_crops, args, image_path, mode_tag="image")

    return letter


# Backward compatibility for older imports.
predict_image = predict_letter
predict_letter_multi = predict_word_crnn_multi

# ── Predykcja wyrazu (jedna linia) ────────────────────────────────────────────

def predict_word(
    image_path: str,
    model: nn.Module,
    device: torch.device,
    args,
    info=None,
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
    img_array = _load_image_array(image_path, args, mode="L")
    if info is None:
        info = globals().get("info")
    letter_boxes = _segment_letters(img_array, args=args)
    debug = _is_debug_enabled(args)

    if debug:
        info(f"[DEBUG][word] wykryto {len(letter_boxes)} segmentów liter")
        for idx, (x1, y1, x2, y2) in enumerate(letter_boxes, start=1):
            info(f"[DEBUG][word] segment {idx}: bbox=({x1},{y1})-({x2},{y2})")

    word = ""
    letter_confidences: list[float] = []
    class_conf_samples: dict[str, list[float]] = {}
    debug_crops: list[tuple[np.ndarray, str]] = []
    model.eval()
    with torch.no_grad():
        for idx, (x1, y1, x2, y2) in enumerate(letter_boxes, start=1):
            letter_img = img_array[y1:y2, x1:x2]
            if letter_img.size == 0:
                if debug:
                    info(f"[DEBUG][word] segment {idx}: pominięty (pusty po przycięciu)")
                continue

            letter_img = preprocess_letter(letter_img)

            predicted_char, confidence, probs = _classify_letter(letter_img, model, device, 1, args)
            word += predicted_char
            letter_confidences.append(confidence)
            class_conf_samples.setdefault(predicted_char, []).append(confidence)
            if debug:
                debug_crops.append((letter_img.copy(), f"{idx}_{predicted_char}_{confidence:.1f}"))

            if debug:
                info(
                    f"[DEBUG][word] segment {idx}: '{predicted_char}' "
                    f"({confidence:.1f}%), top3: {_format_topk_probs(probs, k=3)}"
                )

    avg_word_confidence = float(np.mean(letter_confidences)) if letter_confidences else 0.0
    class_confidence = _mean_per_class(class_conf_samples)
    if debug:
        _finalize_debug_crops(debug_crops, args, image_path, mode_tag="word")
    return word, avg_word_confidence, class_confidence


# ── Predykcja wielu linii tekstu ───────────────────────────────────────────────

def predict_segments(
    image_path: str,
    model: nn.Module,
    device: torch.device,
    args,
    info=None,
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
    img_array = _load_image_array(image_path, args, mode="L")
    if info is None:
        info = globals().get("info")
    binary = img_array < 128
    debug = _is_debug_enabled(args)

    rows_bounds = _find_bounds(np.sum(binary, axis=1))
    if debug:
        info(f"[DEBUG][lines] wykryto {len(rows_bounds)} linii tekstu")

    text = ""
    words_with_confidence: list[tuple[str, float]] = []
    class_conf_samples: dict[str, list[float]] = {}
    debug_crops: list[tuple[np.ndarray, str]] = []

    def _flush_word(word_chars: list[str], word_confs: list[float]) -> None:
        if not word_chars or not word_confs:
            return
        words_with_confidence.append(("".join(word_chars), float(np.mean(word_confs))))

    model.eval()
    with torch.no_grad():
        for line_no, (row_start, row_end) in enumerate(rows_bounds, start=1):
            line_img = img_array[row_start:row_end, :]
            letter_boxes = _segment_letters(line_img, args=args)

            if debug:
                info(
                    f"[DEBUG][lines] linia {line_no}: zakres_wierszy=({row_start},{row_end}), "
                    f"segmenty={len(letter_boxes)}"
                )
                for idx, (x1, y1, x2, y2) in enumerate(letter_boxes, start=1):
                    info(
                        f"[DEBUG][lines] linia {line_no}, segment {idx}: "
                        f"bbox=({x1},{y1})-({x2},{y2})"
                    )

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
                    if debug:
                        info(
                            f"[DEBUG][lines] linia {line_no}, segment {idx + 1}: "
                            "pominięty (pusty po przycięciu)"
                        )
                    continue

                letter_img = preprocess_letter(letter_img)
                predicted_char, confidence, probs = _classify_letter(letter_img, model, device, 1, args)
                line_text += predicted_char
                current_word_chars.append(predicted_char)
                current_word_confs.append(confidence)
                class_conf_samples.setdefault(predicted_char, []).append(confidence)
                if debug:
                    debug_crops.append(
                        (letter_img.copy(), f"L{line_no}_{idx + 1}_{predicted_char}_{confidence:.1f}")
                    )

                if debug:
                    info(
                        f"[DEBUG][lines] linia {line_no}, segment {idx + 1}: "
                        f"'{predicted_char}' ({confidence:.1f}%), "
                        f"top3: {_format_topk_probs(probs, k=3)}"
                    )

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
    if debug:
        _finalize_debug_crops(debug_crops, args, image_path, mode_tag="lines")
    return text, words_with_confidence, class_confidence


def compute_accuracy(predicted: str, reference: str) -> float:
    """Oblicza procentowe podobieństwo (0–100) między predykcją a referencją."""
    pred_norm = predicted.lower().strip()
    ref_norm = reference.lower().strip()
    ratio = difflib.SequenceMatcher(None, pred_norm, ref_norm).ratio()
    return ratio * 100

#Debug

def show_before_after(pil_img, tensor_img):
    before = np.array(pil_img)

    after = tensor_img.squeeze(0).detach().cpu()

    after = after * 0.5 + 0.5
    after = after.numpy()

    if after.shape[0] == 1:
        after = after[0]
        cmap = "gray"
    else:
        after = np.transpose(after, (1, 2, 0))
        cmap = None

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Before")
    plt.imshow(before, cmap="gray" if before.ndim == 2 else None)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("After")
    plt.imshow(after, cmap=cmap)
    plt.axis("off")

    plt.show()



