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
    dump_tensor_stats,
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
    select_version,
    auto_select_models,
)
from .display import visualize_prediction, show_image
from . import info




def _tight_crop(gray: np.ndarray) -> np.ndarray:
    """Przycina obraz do obszaru zawierającego piksele znaku."""
    rows = np.where(np.sum(gray < 128, axis=1) > 0)[0]
    cols = np.where(np.sum(gray < 128, axis=0) > 0)[0]
    if len(rows) == 0 or len(cols) == 0:
        return gray
    return gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]




def _is_debug_enabled(args) -> bool:
    """Sprawdza, czy aktywny jest tryb debug (z fallbackiem do legacy --quiet)."""
    debug = bool(getattr(args, "debug", False))
    quiet = bool(getattr(args, "quiet", False))
    return debug and not quiet



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


# ── Przekazanie zdjęć folderu do predykcji ───────────────────────────────────────────────


def process_folder(folder_path, args, models_dir, device, info):

    if not os.path.isdir(folder_path):
        raise ValueError(f"To nie jest katalog: {folder_path}")
    
    models_dir = os.path.dirname(models_dir)
    version = getattr(args, "model_version", None)
    default_model, selected_models = auto_select_models(models_dir, version)

    if not selected_models:
        fallback_model = load_model(MODEL_PATH, device, info)
        default_model = "model_ocr"
        loaded_models = {default_model: fallback_model}
    else:
        info(f"Automatyczny wybór: {len(selected_models)} model(i), default: {default_model}")
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

            per_model = predict_letter_multi(
                file_path,
                loaded_models,
                device,
                args,
                info
            )

            final_text = aggregate(per_model, default_model)

            # confidence modelu który wygrał głosowanie (lub domyślnego)
            text_conf = {
                name: res["confidence"]
                for name, res in per_model.items()
                if res["text"] == final_text
            }
            best_conf = float(np.mean(list(text_conf.values()))) if text_conf else 0.0

            # letter_vectors z modelu który wyprodukował wybrany tekst (lub domyślnego)
            winning_model = next(
                (n for n in text_conf if n == default_model),
                next(iter(text_conf), default_model)
            )
            letter_vectors = per_model.get(winning_model, {}).get("letter_vectors", [])

            bbox = None
            if bbox_data and bbox_index < len(bbox_data):
                bbox = bbox_data[bbox_index].get("bbox")
            bbox_index += 1

            results.append({
                "file": file_path,
                "text": final_text,
                "confidence": best_conf,
                "bbox": bbox,
                "per_model": per_model,
                "letter_vectors": letter_vectors,
            })

        except Exception as e:
            results.append({
                "file": file_path,
                "error": str(e)
            })

    return results

def process_image(image_path, args, models_dir, device, info):
    if not os.path.isfile(image_path):
        raise ValueError(f"To nie jest plik: {image_path}")

    models_dir = os.path.dirname(models_dir)
    version = getattr(args, "model_version", None)
    default_model, selected_models = auto_select_models(models_dir, version)

    if not selected_models:
        fallback_model = load_model(MODEL_PATH, device, info)
        default_model = "model_ocr"
        loaded_models = {default_model: fallback_model}
    else:
        info(
            f"Automatyczny wybór: {len(selected_models)} model(i), "
            f"default: {default_model}"
        )
        loaded_models = load_models(
            models_dir,
            selected_models,
            device,
            info
        )

    try:
        info(f"\nRozpoznawanie: {image_path}")

        per_model = predict_letter_multi(
            image_path,
            loaded_models,
            device,
            args,
            info
        )

        final_text = aggregate(per_model, default_model)

        # confidence modelu który wygrał głosowanie
        text_conf = {
            name: res["confidence"]
            for name, res in per_model.items()
            if res["text"] == final_text
        }

        best_conf = (
            float(np.mean(list(text_conf.values())))
            if text_conf else 0.0
        )

        # letter_vectors z modelu który wyprodukował wybrany tekst
        winning_model = next(
            (n for n in text_conf if n == default_model),
            next(iter(text_conf), default_model)
        )

        letter_vectors = per_model.get(
            winning_model, {}
        ).get("letter_vectors", [])

        return {
            "file": image_path,
            "text": final_text,
            "confidence": best_conf,
            "bbox": None,
            "per_model": per_model,
            "letter_vectors": letter_vectors,
        }

    except Exception as e:
        return {
            "file": image_path,
            "error": str(e)
        }
    
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

def predict_letter_multi(image_path, models, device, args, info):
    results = {}

    for name, model in models.items():
        res = predict_letter(image_path, model, device, args)
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

        per_model = predict_letter_multi(
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

        per_model = predict_letter_multi(
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
# ── Predykcja pojedynczej litery ───────────────────────────────────────────────
def predict_letter(
    image_path: str,
    model: nn.Module,
    device: torch.device,
    args,
) -> tuple[str, float, torch.Tensor]:


    """
        funkcja rozpoznaje wyraz słowa
    """
    image = load_and_optionally_denoise(image_path, args, mode="L")
    
    img_array = np.array(image)
    debug = _is_debug_enabled(args)
    #if(debug):
        #show_image(image, "przed crop")
    
    debug_crops: list[tuple[np.ndarray, str]] = []

    model.eval()
    with torch.no_grad():
        original_shape = img_array.shape
        
        # crop
        #img_array = _tight_crop(img_array)
        cropped_shape = img_array.shape
        
        # UWAGA: zmień preprocess
        pil = Image.fromarray(img_array).convert("L")
        
        if(debug):
            #show_image(pil, "po crop", True)
            img_before = pil
            
        tensor = get_inf_transform(args)(pil).unsqueeze(0).to(device)
        
        if(debug):
            show_before_after(img_before, tensor)
            
        preprocessed_shape = tensor.shape
        
        
        outputs = model(tensor)  # (T, B, C)
        probs = outputs.softmax(2)

        log_probs = outputs.log_softmax(2)
        probs = log_probs.exp()

        pred = outputs.argmax(2)

        
        preds = log_probs.argmax(2)[:, 0].cpu().numpy()

        chars = []
        confidences = []
        letter_vectors = []   # (litera, wektor_C) dla każdej rozpoznanej litery

        prev = 0  # blank

        for t in range(len(preds)):
            p = preds[t]

            if p != prev and p != 0:
                chars.append(idx2char[p])
                confidences.append(probs[t, 0, p].item())
                letter_vectors.append((idx2char[p], probs[t, 0].cpu().numpy()))

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

    return {"text": text, "confidence": confidence, "per_char_confidences": confidences, "probs": probs_out, "letter_vectors": letter_vectors}


# ── Predykcja tekstu modelem CRNN ─────────────────────────────────────────────



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



