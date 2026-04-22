import difflib
import json
import os
import re
import shutil
import unicodedata

import cv2
import numpy as np


try:
    import easyocr
except ImportError:  # pragma: no cover - runtime guard for optional dependency
    easyocr = None


def clamp_box(box, width, height, min_size=4):
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(1, min(x2, width))
    y2 = max(1, min(y2, height))

    if x2 - x1 < min_size:
        if x1 + min_size <= width:
            x2 = x1 + min_size
        else:
            x1 = max(0, x2 - min_size)

    if y2 - y1 < min_size:
        if y1 + min_size <= height:
            y2 = y1 + min_size
        else:
            y1 = max(0, y2 - min_size)

    return [x1, y1, x2, y2]


def normalize_easyocr_bbox(bbox, width, height):
    xs = [int(point[0]) for point in bbox]
    ys = [int(point[1]) for point in bbox]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return clamp_box([x1, y1, x2, y2], width, height)


def preprocess_image_for_ocr(image, use_denoise=True, use_watershed=True):
    if image.ndim == 3:
        working = image.copy()
        if use_denoise:
            working = cv2.fastNlMeansDenoisingColored(working, None, 10, 10, 7, 21)
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        if use_denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    if not use_watershed:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    max_dist = float(dist_transform.max()) if dist_transform.size else 0.0
    if max_dist <= 0.0:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    _, sure_fg = cv2.threshold(dist_transform, 0.35 * max_dist, 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    marker_source = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(marker_source, markers)

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[markers > 1] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    if cv2.countNonZero(mask) < 10:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    enhanced = cv2.bitwise_and(gray, gray, mask=mask)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def sort_boxes_reading_order(boxes):
    if not boxes:
        return boxes

    prepared = []
    heights = []
    for item in boxes:
        x1, y1, x2, y2 = item["box"]
        h = max(1, int(y2 - y1))
        heights.append(h)
        prepared.append(
            {
                "item": item,
                "x1": int(x1),
                "x2": int(x2),
                "y1": int(y1),
                "y2": int(y2),
                "h": h,
                "cy": (float(y1) + float(y2)) / 2.0,
            }
        )

    median_h = float(np.median(np.array(heights, dtype=np.float32))) if heights else 12.0
    base_tol = max(10.0, 0.60 * median_h)

    prepared.sort(key=lambda d: (d["cy"], d["x1"]))
    lines = []

    for entry in prepared:
        best_idx = None
        best_dist = 1e9

        for idx, line in enumerate(lines):
            line_h = max(1.0, line["y2"] - line["y1"])
            tol = max(base_tol, 0.45 * max(entry["h"], line_h))
            dist = abs(entry["cy"] - line["cy"])

            overlap_h = max(0.0, min(entry["y2"], line["y2"]) - max(entry["y1"], line["y1"]))
            min_h = max(1.0, min(float(entry["h"]), line_h))
            overlap_ratio = overlap_h / min_h

            if dist <= tol or overlap_ratio >= 0.20:
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

        if best_idx is None:
            lines.append(
                {
                    "y1": float(entry["y1"]),
                    "y2": float(entry["y2"]),
                    "cy": float(entry["cy"]),
                    "items": [entry],
                }
            )
        else:
            line = lines[best_idx]
            line["items"].append(entry)
            n = float(len(line["items"]))
            line["y1"] = min(line["y1"], float(entry["y1"]))
            line["y2"] = max(line["y2"], float(entry["y2"]))
            line["cy"] = ((line["cy"] * (n - 1.0)) + entry["cy"]) / n

    lines.sort(key=lambda l: l["cy"])
    merged = []
    for line in lines:
        if not merged:
            merged.append(line)
            continue

        prev = merged[-1]
        prev_h = max(1.0, prev["y2"] - prev["y1"])
        line_h = max(1.0, line["y2"] - line["y1"])
        join_tol = max(8.0, 0.40 * max(prev_h, line_h), 0.35 * median_h)
        center_dist = abs(line["cy"] - prev["cy"])

        overlap_h = max(0.0, min(line["y2"], prev["y2"]) - max(line["y1"], prev["y1"]))
        overlap_ratio = overlap_h / max(1.0, min(prev_h, line_h))

        if center_dist <= join_tol or overlap_ratio >= 0.30:
            prev["items"].extend(line["items"])
            n_prev = float(len(prev["items"]))
            n_line = float(len(line["items"]))
            prev["y1"] = min(prev["y1"], line["y1"])
            prev["y2"] = max(prev["y2"], line["y2"])
            prev["cy"] = ((prev["cy"] * (n_prev - n_line)) + (line["cy"] * n_line)) / n_prev
        else:
            merged.append(line)

    ordered = []
    merged.sort(key=lambda l: l["cy"])
    for line in merged:
        line["items"].sort(key=lambda d: (d["x1"], d["cy"]))
        ordered.extend(entry["item"] for entry in line["items"])

    return ordered


def tokenize_text(text):
    return [token for token in re.split(r"\s+", text.strip()) if token]


def normalize_word_for_match(word):
    word = word.lower().strip()
    word = "".join(ch for ch in word if ch.isalpha() or ch in "-'")
    word = unicodedata.normalize("NFD", word)
    word = "".join(ch for ch in word if unicodedata.category(ch) != "Mn")
    return word


def build_vocab_index(words):
    by_norm = {}
    for word in words:
        base = word.strip()
        if not base:
            continue
        norm = normalize_word_for_match(base)
        if not norm:
            continue
        if norm not in by_norm:
            by_norm[norm] = {}
        by_norm[norm][base] = by_norm[norm].get(base, 0) + 1

    best_form = {}
    for norm, variants in by_norm.items():
        best_form[norm] = max(variants.items(), key=lambda x: x[1])[0]

    return best_form


def load_training_vocabulary(base_dir, letter_name):
    if not os.path.exists(base_dir):
        return []

    pattern = re.compile(rf"^{re.escape(letter_name)}_(\d+)$")
    words = []

    for entry in os.listdir(base_dir):
        if not pattern.match(entry):
            continue

        folder = os.path.join(base_dir, entry)
        if not os.path.isdir(folder):
            continue

        jsonl_path = os.path.join(folder, "boxes.jsonl")
        if not os.path.exists(jsonl_path):
            continue

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    text = str(row.get("text", "")).strip()
                    if text:
                        words.extend(tokenize_text(text))
        except Exception:
            continue

    return words


def predict_text_candidate(text, prob, vocab_index):
    raw_tokens = tokenize_text(text)
    if not raw_tokens or not vocab_index:
        return text, 0.0

    norms = list(vocab_index.keys())
    predicted_tokens = []
    scores = []

    for token in raw_tokens:
        norm = normalize_word_for_match(token)
        if not norm:
            predicted_tokens.append(token)
            scores.append(0.0)
            continue

        if norm in vocab_index:
            predicted_tokens.append(vocab_index[norm])
            scores.append(1.0)
            continue

        match = difflib.get_close_matches(norm, norms, n=1, cutoff=0.74)
        if match:
            best_norm = match[0]
            ratio = difflib.SequenceMatcher(None, norm, best_norm).ratio()
            predicted_tokens.append(vocab_index[best_norm])
            scores.append(ratio)
        else:
            predicted_tokens.append(token)
            scores.append(0.0)

    predicted_text = " ".join(predicted_tokens)
    avg_score = (sum(scores) / len(scores)) if scores else 0.0

    if prob >= 0.80 and avg_score < 0.95:
        return text, avg_score

    return predicted_text, avg_score


def attach_word_predictions(boxes, vocab_words):
    vocab_index = build_vocab_index(vocab_words)
    for box_data in boxes:
        text = box_data.get("text", "")
        prob = float(box_data.get("prob", 0.0))
        suggestion, score = predict_text_candidate(text, prob, vocab_index)
        box_data["suggested_text"] = suggestion
        box_data["prediction_score"] = float(score)
    return boxes


def apply_suggested_texts(boxes, min_score=0.82):
    for box_data in boxes:
        if box_data.get("text_override", False):
            continue
        current_text = str(box_data.get("text", "")).strip()
        suggested_text = str(box_data.get("suggested_text", "")).strip()
        score = float(box_data.get("prediction_score", 0.0))
        if suggested_text and suggested_text != current_text and score >= min_score:
            box_data["text"] = suggested_text
            box_data["text_override"] = True
    return boxes


def draw_boxes_preview(image, boxes, selected_idx, temp_box=None):
    preview = image.copy()
    for i, box_data in enumerate(boxes):
        x1, y1, x2, y2 = box_data["box"]
        color = (0, 255, 255) if i == selected_idx else (0, 200, 0)
        thickness = 2 if i == selected_idx else 1
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, thickness)
        label = f"{i:03d}: {box_data['text'][:20]}"
        cv2.putText(preview, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if temp_box is not None:
        tx1, ty1, tx2, ty2 = temp_box
        cv2.rectangle(preview, (tx1, ty1), (tx2, ty2), (255, 200, 0), 2)
        cv2.putText(preview, "NOWY", (tx1, max(18, ty1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    return preview


def point_in_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def detect_handle(x, y, box, margin=8):
    x1, y1, x2, y2 = box
    near_left = abs(x - x1) <= margin
    near_right = abs(x - x2) <= margin
    near_top = abs(y - y1) <= margin
    near_bottom = abs(y - y2) <= margin

    if near_left and near_top:
        return "lt"
    if near_right and near_top:
        return "rt"
    if near_left and near_bottom:
        return "lb"
    if near_right and near_bottom:
        return "rb"
    if near_left:
        return "l"
    if near_right:
        return "r"
    if near_top:
        return "t"
    if near_bottom:
        return "b"
    return "move"


def get_box_index_at_point(boxes, x, y):
    for i in range(len(boxes) - 1, -1, -1):
        if point_in_box(x, y, boxes[i]["box"]):
            return i
    return None


def apply_resize_from_handle(box, handle, dx, dy):
    x1, y1, x2, y2 = box
    if "l" in handle:
        x1 += dx
    if "r" in handle:
        x2 += dx
    if "t" in handle:
        y1 += dy
    if "b" in handle:
        y2 += dy
    return [x1, y1, x2, y2]


def get_preview_scale(image, max_width=1400, max_height=900):
    height, width = image.shape[:2]
    scale = min(max_width / float(width), max_height / float(height), 1.0)
    return scale


def scale_box(box, scale):
    x1, y1, x2, y2 = box
    return [
        int(round(x1 * scale)),
        int(round(y1 * scale)),
        int(round(x2 * scale)),
        int(round(y2 * scale)),
    ]


def unscale_point(x, y, scale):
    if scale == 0:
        return x, y
    return int(round(x / scale)), int(round(y / scale))


def clamp_value(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def mouse_wheel_delta(flags):
    delta = (flags >> 16) & 0xFFFF
    if delta >= 0x8000:
        delta -= 0x10000
    return delta


def get_next_output_dir(base_dir, letter_name):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    pattern = re.compile(rf"^{re.escape(letter_name)}_(\d+)$")
    max_idx = 0

    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if not os.path.isdir(full_path):
            continue
        match = pattern.match(entry)
        if match:
            max_idx = max(max_idx, int(match.group(1)))

    next_idx = max_idx + 1
    folder_name = f"{letter_name}_{next_idx}"
    return os.path.join(base_dir, folder_name)


def get_latest_output_dir(base_dir, letter_name):
    if not os.path.exists(base_dir):
        return None

    pattern = re.compile(rf"^{re.escape(letter_name)}_(\d+)$")
    latest_idx = -1
    latest_dir = None

    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if not os.path.isdir(full_path):
            continue
        match = pattern.match(entry)
        if not match:
            continue

        idx = int(match.group(1))
        if idx > latest_idx:
            latest_idx = idx
            latest_dir = full_path

    return latest_dir


def load_boxes_from_annotations(folder_path, img_w, img_h):
    if not folder_path:
        return None

    jsonl_path = os.path.join(folder_path, "boxes.jsonl")
    if not os.path.exists(jsonl_path):
        return None

    loaded_boxes = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            bbox = row.get("bbox_xyxy")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            loaded_boxes.append(
                {
                    "box": clamp_box([x1, y1, x2, y2], img_w, img_h),
                    "text": row.get("text", "[manual]"),
                    "prob": float(row.get("prob", 1.0)),
                    "text_override": bool(row.get("text_override", False)),
                }
            )

    return loaded_boxes if loaded_boxes else None


def recognize_text_in_box(reader, image, box):
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return "", 0.0

    crop_for_ocr = preprocess_image_for_ocr(crop)

    result = reader.readtext(crop_for_ocr, detail=1, paragraph=False)
    if not result:
        return "", 0.0

    texts = []
    probs = []
    for _, text, prob in result:
        if text.strip():
            texts.append(text.strip())
            probs.append(float(prob))

    if not texts:
        return "", 0.0

    merged_text = " ".join(texts)
    avg_prob = sum(probs) / len(probs)
    return merged_text, avg_prob


def refresh_box_texts_with_ocr(reader, image, boxes):
    refreshed = []
    for box_data in boxes:
        if box_data.get("text_override", False):
            refreshed.append(
                {
                    "box": box_data["box"],
                    "text": box_data.get("text", ""),
                    "prob": float(box_data.get("prob", 1.0)),
                    "text_override": True,
                }
            )
            continue

        text, prob = recognize_text_in_box(reader, image, box_data["box"])
        if text:
            refreshed.append(
                {
                    "box": box_data["box"],
                    "text": text,
                    "prob": prob,
                    "text_override": False,
                }
            )
        else:
            fallback_text = box_data.get("text", "")
            fallback_prob = float(box_data.get("prob", 0.0))
            refreshed.append(
                {
                    "box": box_data["box"],
                    "text": fallback_text,
                    "prob": fallback_prob,
                    "text_override": bool(box_data.get("text_override", False)),
                }
            )
    return refreshed


def optionally_edit_box_texts(boxes, image):
    answer = input("Czy chcesz ręcznie poprawić rozpoznane wyrazy? [t/N]: ").strip().lower()
    if answer not in ("t", "tak", "y", "yes"):
        return boxes

    print("Wpisz nowy tekst i ENTER. Pusty ENTER = bez zmian.")
    window_name = "Podglad wycinka"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for i, box_data in enumerate(boxes):
        x1, y1, x2, y2 = box_data["box"]
        cx1, cx2 = sorted((int(x1), int(x2)))
        cy1, cy2 = sorted((int(y1), int(y2)))
        crop = image[max(0, cy1):max(0, cy2), max(0, cx1):max(0, cx2)]

        try:
            if crop.size == 0:
                preview = np.full((220, 440, 3), 35, dtype=np.uint8)
                cv2.putText(preview, f"[{i:03d}] Pusty wycinek", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
                cv2.putText(preview, "Sprawdz bbox", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
            else:
                crop_h, crop_w = crop.shape[:2]
                max_w, max_h = 900, 360
                scale = min(max_w / float(crop_w), max_h / float(crop_h))
                new_w = max(1, int(round(crop_w * scale)))
                new_h = max(1, int(round(crop_h * scale)))
                interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
                resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

                preview = np.full((max_h, max_w, 3), 30, dtype=np.uint8)
                off_x = (max_w - new_w) // 2
                off_y = (max_h - new_h) // 2
                preview[off_y:off_y + new_h, off_x:off_x + new_w] = resized
                cv2.putText(preview, f"[{i:03d}] {crop_w}x{crop_h}", (10, max_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow(window_name, preview)
            for _ in range(4):
                cv2.waitKey(25)
        except Exception as e:
            print(f"[{i:03d}] Błąd wyświetlania: {e}")

        current_text = box_data.get("text", "")
        suggested_text = box_data.get("suggested_text", current_text)
        print(f"[{i:03d}] aktualnie: {current_text}")
        if suggested_text and suggested_text != current_text:
            print(f"      sugestia: {suggested_text}")
        new_text = input("Nowy tekst (ENTER=bez zmian, /=uzyj sugestii): ").strip()
        if new_text == "/":
            new_text = suggested_text
        if new_text:
            box_data["text"] = new_text
            box_data["prob"] = float(box_data.get("prob", 0.0))
            box_data["text_override"] = True

    cv2.destroyWindow(window_name)

    return boxes


def save_box_annotations(output_dir, image_path, records):
    tsv_path = os.path.join(output_dir, "boxes.tsv")
    jsonl_path = os.path.join(output_dir, "boxes.jsonl")

    with open(tsv_path, "w", encoding="utf-8") as tsv_file:
        tsv_file.write("id\tcrop_file\txmin\tymin\txmax\tymax\ttext\tsuggested_text\tprediction_score\tprob\tsource_image\n")
        for rec in records:
            safe_text = rec["text"].replace("\t", " ").replace("\n", " ").strip()
            safe_suggested = rec.get("suggested_text", "").replace("\t", " ").replace("\n", " ").strip()
            tsv_file.write(
                f"{rec['id']}\t{rec['crop_file']}\t{rec['xmin']}\t{rec['ymin']}\t{rec['xmax']}\t{rec['ymax']}\t{safe_text}\t{safe_suggested}\t{rec.get('prediction_score', 0.0):.6f}\t{rec['prob']:.6f}\t{image_path}\n"
            )

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for rec in records:
            row = {
                "id": rec["id"],
                "crop_file": rec["crop_file"],
                "bbox_xyxy": [rec["xmin"], rec["ymin"], rec["xmax"], rec["ymax"]],
                "text": rec["text"],
                "suggested_text": rec.get("suggested_text", ""),
                "prediction_score": rec.get("prediction_score", 0.0),
                "text_override": bool(rec.get("text_override", False)),
                "prob": rec["prob"],
                "source_image": image_path,
            }
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return tsv_path, jsonl_path


def edit_boxes_interactive(image, boxes):
    height, width = image.shape[:2]
    selected_idx = 0 if boxes else -1
    step = 2
    min_new_box_size = 8
    preview_scale = get_preview_scale(image)
    zoom = 1.0

    state = {
        "selected_idx": selected_idx,
        "drag_mode": None,
        "drag_anchor": None,
        "start_box": None,
        "temp_box": None,
        "zoom": zoom,
        "view_offset_x": 0,
        "view_offset_y": 0,
        "pan_mode": False,
        "pan_anchor": None,
        "pan_start_offset": None,
    }

    print("\nTryb poprawy bboxów:")
    print("Myszka: kliknij box, przeciągaj wnętrze (move), przeciągaj krawędzie/rogi (resize)")
    print("Myszka: przeciągnij poza boxami, aby narysować nowy box")
    print("n/p - następny/poprzedni box")
    print("strzałki lub w/a/s/d - przesuwanie całego boxa")
    print("+/- - zmiana kroku ruchu")
    print("kółko myszy lub z/c - przybliżanie/oddalanie")
    print("PPM + przeciąganie - przesuwanie widoku (pan)")
    print("x lub Delete - usuń aktualny box")
    print("ENTER - zatwierdź, q - anuluj edycję")

    window_name = "Korekta bounding boxow"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    def get_display_image_and_scale():
        display_scale = preview_scale * state["zoom"]
        if display_scale == 1.0:
            return image, display_scale
        display_image = cv2.resize(image, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA if display_scale < 1.0 else cv2.INTER_CUBIC)
        return display_image, display_scale

    def get_view_geometry(display_image):
        disp_h, disp_w = display_image.shape[:2]
        max_view_w = min(1400, disp_w)
        max_view_h = min(900, disp_h)

        max_off_x = max(0, disp_w - max_view_w)
        max_off_y = max(0, disp_h - max_view_h)
        state["view_offset_x"] = clamp_value(state["view_offset_x"], 0, max_off_x)
        state["view_offset_y"] = clamp_value(state["view_offset_y"], 0, max_off_y)

        return max_view_w, max_view_h, state["view_offset_x"], state["view_offset_y"]

    def on_mouse(event, x, y, flags, param):
        display_image, display_scale = get_display_image_and_scale()
        view_w, view_h, view_x, view_y = get_view_geometry(display_image)
        display_x = clamp_value(x + view_x, 0, display_image.shape[1] - 1)
        display_y = clamp_value(y + view_y, 0, display_image.shape[0] - 1)
        ix, iy = unscale_point(display_x, display_y, display_scale)
        ix = clamp_value(ix, 0, width - 1)
        iy = clamp_value(iy, 0, height - 1)

        if event == cv2.EVENT_LBUTTONDOWN:
            hit_idx = get_box_index_at_point(boxes, ix, iy)
            if hit_idx is not None:
                state["selected_idx"] = hit_idx
                box = boxes[hit_idx]["box"]
                handle = detect_handle(ix, iy, box)
                state["drag_mode"] = handle
                state["drag_anchor"] = (ix, iy)
                state["start_box"] = box.copy()
            else:
                state["drag_mode"] = "draw"
                state["drag_anchor"] = (ix, iy)
                state["temp_box"] = [ix, iy, ix, iy]

        elif event == cv2.EVENT_RBUTTONDOWN:
            state["pan_mode"] = True
            state["pan_anchor"] = (x, y)
            state["pan_start_offset"] = (view_x, view_y)

        elif event == cv2.EVENT_RBUTTONUP:
            state["pan_mode"] = False
            state["pan_anchor"] = None
            state["pan_start_offset"] = None

        elif event == cv2.EVENT_MOUSEWHEEL:
            wheel_delta = mouse_wheel_delta(flags)
            if wheel_delta > 0:
                state["zoom"] = clamp_value(state["zoom"] * 1.15, 0.5, 6.0)
            elif wheel_delta < 0:
                state["zoom"] = clamp_value(state["zoom"] / 1.15, 0.5, 6.0)

        elif event == cv2.EVENT_MOUSEMOVE and state["pan_mode"] and state["pan_anchor"] and state["pan_start_offset"]:
            ax, ay = state["pan_anchor"]
            sx, sy = state["pan_start_offset"]
            dx = x - ax
            dy = y - ay
            max_off_x = max(0, display_image.shape[1] - view_w)
            max_off_y = max(0, display_image.shape[0] - view_h)
            state["view_offset_x"] = clamp_value(sx - dx, 0, max_off_x)
            state["view_offset_y"] = clamp_value(sy - dy, 0, max_off_y)

        elif event == cv2.EVENT_MOUSEMOVE and state["drag_mode"]:
            ax, ay = state["drag_anchor"]
            dx = ix - ax
            dy = iy - ay

            if state["drag_mode"] == "draw":
                x1, x2 = sorted([ax, ix])
                y1, y2 = sorted([ay, iy])
                state["temp_box"] = clamp_box([x1, y1, x2, y2], width, height)
            elif state["selected_idx"] >= 0 and state["selected_idx"] < len(boxes):
                start_box = state["start_box"]
                if state["drag_mode"] == "move":
                    x1, y1, x2, y2 = start_box
                    moved_box = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
                    boxes[state["selected_idx"]]["box"] = clamp_box(moved_box, width, height)
                else:
                    resized = apply_resize_from_handle(start_box, state["drag_mode"], dx, dy)
                    boxes[state["selected_idx"]]["box"] = clamp_box(resized, width, height)

        elif event == cv2.EVENT_LBUTTONUP:
            if state["drag_mode"] == "draw" and state["temp_box"] is not None:
                x1, y1, x2, y2 = state["temp_box"]
                if (x2 - x1) >= min_new_box_size and (y2 - y1) >= min_new_box_size:
                    boxes.append({"box": [x1, y1, x2, y2], "text": "[manual]", "prob": 1.0})
                    state["selected_idx"] = len(boxes) - 1

            state["drag_mode"] = None
            state["drag_anchor"] = None
            state["start_box"] = None
            state["temp_box"] = None

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        selected_idx = state["selected_idx"]
        if boxes and selected_idx < 0:
            selected_idx = 0
            state["selected_idx"] = 0
        if selected_idx >= len(boxes):
            selected_idx = len(boxes) - 1
            state["selected_idx"] = selected_idx

        display_image, display_scale = get_display_image_and_scale()
        view_w, view_h, view_x, view_y = get_view_geometry(display_image)
        preview_boxes = [dict(item, box=scale_box(item["box"], display_scale)) for item in boxes]
        temp_box = scale_box(state["temp_box"], display_scale) if state["temp_box"] is not None else None
        preview = draw_boxes_preview(display_image, preview_boxes, selected_idx, temp_box)
        view = preview[view_y:view_y + view_h, view_x:view_x + view_w]
        cv2.imshow(window_name, view)
        key = cv2.waitKeyEx(30)

        if key == -1:
            continue

        if key in (13, 10):
            cv2.destroyWindow(window_name)
            return boxes
        if key == ord("q"):
            cv2.destroyWindow(window_name)
            return None

        if key in (ord("n"), 9) and boxes:
            selected_idx = (selected_idx + 1) % len(boxes)
            state["selected_idx"] = selected_idx
            continue
        if key == ord("p") and boxes:
            selected_idx = (selected_idx - 1) % len(boxes)
            state["selected_idx"] = selected_idx
            continue

        if key == ord("b"):
            if boxes:
                x1, y1, x2, y2 = boxes[selected_idx]["box"]
                w = max(20, x2 - x1)
                h = max(20, y2 - y1)
                nx1 = min(width - 1, x1 + 10)
                ny1 = min(height - 1, y1 + 10)
                nx2 = min(width, nx1 + w)
                ny2 = min(height, ny1 + h)
            else:
                w = max(20, width // 8)
                h = max(20, height // 8)
                nx1 = max(0, (width - w) // 2)
                ny1 = max(0, (height - h) // 2)
                nx2 = nx1 + w
                ny2 = ny1 + h

            new_box = clamp_box([nx1, ny1, nx2, ny2], width, height)
            boxes.append({"box": new_box, "text": "[manual]", "prob": 1.0})
            selected_idx = len(boxes) - 1
            state["selected_idx"] = selected_idx
            continue

        if key in (ord("+"), ord("=")):
            step = min(50, step + 1)
            print(f"Krok: {step}")
            continue
        if key in (ord("-"), ord("_")):
            step = max(1, step - 1)
            print(f"Krok: {step}")
            continue

        if key == ord("z"):
            state["zoom"] = clamp_value(state["zoom"] * 1.15, 0.5, 6.0)
            continue
        if key == ord("x"):
            if boxes:
                boxes.pop(selected_idx)
                if not boxes:
                    state["selected_idx"] = -1
                    continue
                selected_idx = min(selected_idx, len(boxes) - 1)
                state["selected_idx"] = selected_idx
            continue
        if key == ord("c"):
            state["zoom"] = clamp_value(state["zoom"] / 1.15, 0.5, 6.0)
            continue

        if key in (3014656,) and boxes:
            boxes.pop(selected_idx)
            if not boxes:
                state["selected_idx"] = -1
                continue
            selected_idx = min(selected_idx, len(boxes) - 1)
            state["selected_idx"] = selected_idx
            continue

        if not boxes:
            continue

        x1, y1, x2, y2 = boxes[selected_idx]["box"]

        if key in (ord("w"), 2490368):
            y1 -= step
            y2 -= step
        elif key in (ord("s"), 2621440):
            y1 += step
            y2 += step
        elif key in (ord("a"), 2424832):
            x1 -= step
            x2 -= step
        elif key in (ord("d"), 2555904):
            x1 += step
            x2 += step
        elif key == ord("i"):
            y1 -= step
        elif key == ord("k"):
            y1 += step
        elif key == ord("j"):
            x1 -= step
        elif key == ord("l"):
            x1 += step
        elif key == ord("t"):
            y2 -= step
        elif key == ord("g"):
            y2 += step
        elif key == ord("f"):
            x2 -= step
        elif key == ord("h"):
            x2 += step

        boxes[selected_idx]["box"] = clamp_box([x1, y1, x2, y2], width, height)


def process_letter(image_path, base_dir="ttdata", enable_box_edit=True, non_interactive=False):
    if easyocr is None:
        print("Błąd: brak pakietu easyocr. Zainstaluj: pip install easyocr")
        return None

    if not os.path.exists(image_path):
        print(f"Błąd: Plik '{image_path}' nie istnieje.")
        return None

    letter_name = os.path.splitext(os.path.basename(image_path))[0]

    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        print(f"Błąd: nie udało się wczytać obrazu '{image_path}'.")
        return None

    img_h, img_w = img_cv2.shape[:2]

    reader = easyocr.Reader(["pl"], gpu=False)

    previous_dir = get_latest_output_dir(base_dir, letter_name)
    boxes = load_boxes_from_annotations(previous_dir, img_w, img_h)

    if boxes:
        print(f"Wczytano {len(boxes)} poprzednich boxów z: {previous_dir}")
    else:
        print("Brak poprzednich adnotacji - uruchamiam OCR.")
        print(f"Analizuję: {letter_name}...")
        ocr_image = preprocess_image_for_ocr(img_cv2)
        results = reader.readtext(ocr_image, paragraph=False)

        boxes = []
        for (bbox, text, prob) in results:
            boxes.append(
                {
                    "box": normalize_easyocr_bbox(bbox, img_w, img_h),
                    "text": text,
                    "prob": prob,
                    "text_override": False,
                }
            )

        boxes = sort_boxes_reading_order(boxes)

    edited_boxes = boxes
    if enable_box_edit:
        edited_boxes = edit_boxes_interactive(img_cv2, boxes)
        if edited_boxes is None:
            print("Anulowano edycję bboxów. Nic nie zapisano.")
            return None

    print("Odświeżam rozpoznanie tekstu dla finalnych boxów...")
    edited_boxes = refresh_box_texts_with_ocr(reader, img_cv2, edited_boxes)

    vocab_words = load_training_vocabulary(base_dir, letter_name)
    edited_boxes = attach_word_predictions(edited_boxes, vocab_words)
    if non_interactive:
        edited_boxes = apply_suggested_texts(edited_boxes)
    if not non_interactive:
        edited_boxes = optionally_edit_box_texts(edited_boxes, img_cv2)

    latest_dir_for_save = get_latest_output_dir(base_dir, letter_name)

    if latest_dir_for_save:
        print(f"Wykryto poprzedni folder: {latest_dir_for_save}")
        if non_interactive:
            save_choice = "n"
        else:
            save_choice = input("Tryb zapisu: [N]owy folder / [O]nadpisz poprzedni / [A]nuluj: ").strip().lower()

        if save_choice in ("a", "anuluj", "cancel", "q"):
            print("Anulowano zapis.")
            return None

        if save_choice in ("o", "overwrite", "nadpisz"):
            output_dir = latest_dir_for_save
            try:
                shutil.rmtree(output_dir)
            except Exception as e:
                print(f"Błąd usuwania folderu do nadpisania: {e}")
                return None
            os.makedirs(output_dir, exist_ok=True)
            print(f"Nadpisuję poprzedni folder: {output_dir}")
        else:
            output_dir = get_next_output_dir(base_dir, letter_name)
            os.makedirs(output_dir, exist_ok=True)
            print(f"Utworzono nowy podfolder dla listu: {output_dir}")
    else:
        output_dir = get_next_output_dir(base_dir, letter_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Utworzono podfolder dla listu: {output_dir}")

    counter = 0
    annotation_records = []

    source_image_copy = os.path.join(output_dir, "source_image.jpg")
    cv2.imwrite(source_image_copy, img_cv2, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Zapisano oryginalne zdjęcie: {source_image_copy}")

    for box_data in edited_boxes:
        xmin, ymin, xmax, ymax = box_data["box"]
        text = box_data["text"]
        prob = float(box_data.get("prob", 1.0))
        crop_img = img_cv2[ymin:ymax, xmin:xmax]

        if crop_img.size == 0:
            continue

        file_name = f"word_{counter:03d}.png"
        save_path = os.path.join(output_dir, file_name)
        cv2.imwrite(save_path, crop_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        annotation_records.append(
            {
                "id": counter,
                "crop_file": file_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "text": text,
                "suggested_text": box_data.get("suggested_text", ""),
                "prediction_score": float(box_data.get("prediction_score", 0.0)),
                "text_override": bool(box_data.get("text_override", False)),
                "prob": prob,
            }
        )

        suggested = box_data.get("suggested_text", "")
        if suggested and suggested != text:
            print(f"{file_name} | {text} | sugestia: {suggested}")
        else:
            print(f"{file_name} | {text}")
        counter += 1

    tsv_path, jsonl_path = save_box_annotations(output_dir, image_path, annotation_records)
    print(f"Zapisano adnotacje bboxów: {tsv_path}")
    print(f"Zapisano adnotacje bboxów (JSONL): {jsonl_path}")

    print(f"\nGotowe! Wszystkie wycinki z '{letter_name}' znajdziesz w: {output_dir}")
    return output_dir
