import json
import os
import re
import shutil

import cv2
import numpy as np


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


def get_bbox_dimensions(box):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    return width, height


def analyze_bbox_lengths(boxes):
    if not boxes:
        return None
    
    lengths = []
    for box_data in boxes:
        box = box_data["box"]
        w, h = get_bbox_dimensions(box)
        lengths.append(w)
    
    lengths = np.array(lengths, dtype=np.float32)
    
    stats = {
        "count": len(lengths),
        "min": float(np.min(lengths)),
        "max": float(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "p25": float(np.percentile(lengths, 25)),
        "p50": float(np.percentile(lengths, 50)),
        "p75": float(np.percentile(lengths, 75)),
        "p90": float(np.percentile(lengths, 90)),
    }
    
    return stats


def categorize_bbox_by_length(box, thresholds=None):
    w, h = get_bbox_dimensions(box)
    
    if thresholds is None:
        thresholds = {"short": 50, "medium": 120}
    
    if w < thresholds["short"]:
        return "short"
    elif w < thresholds["medium"]:
        return "medium"
    else:
        return "long"


def add_length_to_boxes(boxes):
    for box_data in boxes:
        box = box_data["box"]
        w, h = get_bbox_dimensions(box)
        box_data["width"] = w
        box_data["height"] = h
        box_data["length_category"] = categorize_bbox_by_length(box)
    return boxes


def split_lines_from_roi(roi, x, y, median_height, min_peak_distance=10):
    print("SPLIT CALLED")
    projection = np.sum(roi > 0, axis=1).astype(np.float32)
    

    if np.max(projection) == 0:
        return []

    # normalizacja
    projection = projection / (np.max(projection) + 1e-6)

    # 1. znajdź piki (linie tekstu)
    peaks = []
    for i in range(1, len(projection) - 1):
        if projection[i] > projection[i - 1] and projection[i] > projection[i + 1]:
            if projection[i] > 0.4:  # minimalna aktywność linii
                peaks.append(i)

    if len(peaks) < 2:
        return [(x, y, roi.shape[1], roi.shape[0])]

    # 2. znajdź doliny między pikami
    cuts = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]

        valley_region = projection[start:end]
        if len(valley_region) == 0:
            continue

        cut_y = start + np.argmin(valley_region)
        cuts.append(cut_y)

    # 3. budowa segmentów
    segments = []
    prev = 0

    for cut in cuts:
        if cut - prev > min_peak_distance:
            segments.append((prev, cut))
        prev = cut

    # ostatni segment
    if roi.shape[0] - prev > min_peak_distance:
        segments.append((prev, roi.shape[0]))

    # 5. konwersja do bboxów
    return [
        (x, y + s, roi.shape[1], e - s)
        for s, e in segments
    ]


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
    base_tol = max(4.0, 0.35 * median_h)

    prepared.sort(key=lambda d: (d["cy"], d["x1"]))
    lines = []

    for entry in prepared:
        best_idx = None
        best_dist = 1e9

        for idx, line in enumerate(lines):
            line_h = max(1.0, line["y2"] - line["y1"])
            tol = max(base_tol, 0.30 * max(entry["h"], line_h))
            dist = abs(entry["cy"] - line["cy"])

            overlap_h = max(0.0, min(entry["y2"], line["y2"]) - max(entry["y1"], line["y1"]))
            min_h = max(1.0, min(float(entry["h"]), line_h))
            overlap_ratio = overlap_h / min_h

            if dist <= tol or overlap_ratio >= 0.65:
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
        join_tol = max(4.0, 0.20 * max(prev_h, line_h), 0.20 * median_h)
        center_dist = abs(line["cy"] - prev["cy"])

        overlap_h = max(0.0, min(line["y2"], prev["y2"]) - max(line["y1"], prev["y1"]))
        overlap_ratio = overlap_h / max(1.0, min(prev_h, line_h))

        if center_dist <= join_tol or overlap_ratio >= 0.75:
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


def preprocess_for_detection(image, clip_limit=2.0, tile_grid_size=(6, 6)):
    """
    Przygotowuje obraz do lepszej detekcji - tylko kontrast (CLAHE),
    bez odszumiania które powoduje rozmycie.
    Uwaga: detect_word_boxes_auto ma w\u0142asny preprocessing wewn\u0105trz,
    wi\u0119c ta funkcja jest u\u017cywana g\u0142ównie dla detect_word_boxes_auto
    przy ponownej detekcji (klawisz 'a' w edytorze).
    """
    if image is None or image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def build_detection_preview(image):
    if image is None or image.size == 0:
        return image

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    tile_size = max(4, min(16, min(height, width) // 80))
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return blurred


def tighten_box_to_foreground(image, box, pad=2):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, width, height)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return [x1, y1, x2, y2]

    if crop.ndim == 3:
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        crop_gray = crop.copy()

    blurred = cv2.GaussianBlur(crop_gray, (3, 3), 0)

    def score_mask(mask):
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            return None

        bx1 = int(xs.min())
        by1 = int(ys.min())
        bx2 = int(xs.max()) + 1
        by2 = int(ys.max()) + 1
        area = max(1, (bx2 - bx1) * (by2 - by1))
        fg_ratio = float(np.count_nonzero(mask)) / float(mask.size)

        if fg_ratio < 0.002 or fg_ratio > 0.80:
            return None

        # Preferuj maski dajace zwarte, relatywnie male ramki.
        sparsity = area / float(mask.size)
        return {
            "box": [bx1, by1, bx2, by2],
            "score": (sparsity * 10.0) + abs(fg_ratio - 0.12),
        }

    masks = []

    otsu_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    masks.append(otsu_inv)

    otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    masks.append(otsu)

    block_size = max(21, (min(crop_gray.shape[:2]) // 8) | 1)
    adaptive_inv = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        5,
    )
    masks.append(adaptive_inv)

    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        5,
    )
    masks.append(adaptive)

    inv_best = None
    for mask in masks:
        candidate = score_mask(mask)
        if candidate is None:
            continue
        if inv_best is None or candidate["score"] < inv_best["score"]:
            inv_best = candidate

    if inv_best is None:
        return [x1, y1, x2, y2]

    bx1, by1, bx2, by2 = inv_best["box"]
    nx1 = x1 + bx1 - pad
    ny1 = y1 + by1 - pad
    nx2 = x1 + bx2 + pad
    ny2 = y1 + by2 + pad

    return clamp_box([nx1, ny1, nx2, ny2], width, height)


def split_wide_box_by_projection(image, box):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, width, height)
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return []

    # Nie dziel małych lub umiarkowanie szerokich boxów - to zwykle kursywa albo liczby.
    if w < max(110, 4 * h):
        return [[x1, y1, x2, y2]]

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return [[x1, y1, x2, y2]]

    if crop.ndim == 3:
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        crop_gray = crop.copy()

    blurred = cv2.GaussianBlur(crop_gray, (3, 3), 0)
    mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        max(21, (min(crop_gray.shape[:2]) // 8) | 1),
        5,
    )

    proj = np.sum(mask > 0, axis=0).astype(np.float32)
    if proj.size < 8 or np.max(proj) <= 0:
        return [[x1, y1, x2, y2]]

    foreground_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if foreground_ratio > 0.25:
        return [[x1, y1, x2, y2]]

    p20 = float(np.percentile(proj, 20))
    p85 = float(np.percentile(proj, 85))
    valley_thr = max(0.0, p20 + 0.48 * max(0.0, p85 - p20))

    empty = proj <= valley_thr
    min_gap = max(14, int(round(0.22 * w)))
    split_points = []
    run_start = None

    for idx, is_empty in enumerate(empty):
        if is_empty and run_start is None:
            run_start = idx
        elif (not is_empty) and run_start is not None:
            run_len = idx - run_start
            if run_len >= min_gap:
                split_points.append(run_start + (run_len // 2))
            run_start = None

    if run_start is not None:
        run_len = len(empty) - run_start
        if run_len >= min_gap:
            split_points.append(run_start + (run_len // 2))

    if not split_points:
        return [[x1, y1, x2, y2]]

    pieces = []
    left = 0
    for sp in split_points:
        if sp - left >= max(16, int(round(0.14 * w))):
            pieces.append((left, sp))
        left = sp
    if w - left >= max(16, int(round(0.14 * w))):
        pieces.append((left, w))

    if len(pieces) <= 1:
        return [[x1, y1, x2, y2]]

    split_boxes = []
    for sx1, sx2 in pieces:
        seg = mask[:, sx1:sx2]
        ys, xs = np.where(seg > 0)
        if xs.size == 0 or ys.size == 0:
            continue
        bx1 = x1 + sx1 + int(xs.min())
        bx2 = x1 + sx1 + int(xs.max()) + 1
        by1 = y1 + int(ys.min())
        by2 = y1 + int(ys.max()) + 1
        split_boxes.append(clamp_box([bx1, by1, bx2, by2], width, height))

    return split_boxes if split_boxes else [[x1, y1, x2, y2]]


def merge_boxes_into_words(boxes):
    if not boxes:
        return boxes

    prepared = []
    heights = []
    widths = []
    for box in boxes:
        x1, y1, x2, y2 = box
        w = max(1, int(x2 - x1))
        h = max(1, int(y2 - y1))
        widths.append(w)
        heights.append(h)
        prepared.append(
            {
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "x1": int(x1),
                "x2": int(x2),
                "y1": int(y1),
                "y2": int(y2),
                "w": w,
                "h": h,
                "cy": (float(y1) + float(y2)) / 2.0,
            }
        )

    median_h = float(np.median(np.array(heights, dtype=np.float32))) if heights else 12.0
    median_w = float(np.median(np.array(widths, dtype=np.float32))) if widths else 12.0
    line_tol = max(4.0, 0.40 * median_h)

    prepared.sort(key=lambda d: (d["cy"], d["x1"]))
    lines = []

    for entry in prepared:
        best_idx = None
        best_dist = 1e9

        for idx, line in enumerate(lines):
            line_h = max(1.0, line["y2"] - line["y1"])
            dist = abs(entry["cy"] - line["cy"])
            overlap_h = max(0.0, min(entry["y2"], line["y2"]) - max(entry["y1"], line["y1"]))
            overlap_ratio = overlap_h / max(1.0, min(float(entry["h"]), line_h))

            if dist <= line_tol or overlap_ratio >= 0.60:
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

        if best_idx is None:
            lines.append(
                {
                    "items": [entry],
                    "cy": float(entry["cy"]),
                    "y1": float(entry["y1"]),
                    "y2": float(entry["y2"]),
                }
            )
        else:
            line = lines[best_idx]
            line["items"].append(entry)
            n = float(len(line["items"]))
            line["cy"] = ((line["cy"] * (n - 1.0)) + entry["cy"]) / n
            line["y1"] = min(line["y1"], float(entry["y1"]))
            line["y2"] = max(line["y2"], float(entry["y2"]))

    merged = []
    lines.sort(key=lambda l: l["cy"])
    for line in lines:
        items = sorted(line["items"], key=lambda d: d["x1"])
        if not items:
            continue

        line_h = max(1.0, line["y2"] - line["y1"])
        merge_gap_thr = max(2.0, min(4.0, 0.18 * median_w, 0.15 * median_h))
        current = items[0]["box"]

        for entry in items[1:]:
            box = entry["box"]
            cx1, cy1, cx2, cy2 = current
            bx1, by1, bx2, by2 = box

            gap = float(bx1 - cx2)
            inter_h = max(0.0, min(cy2, by2) - max(cy1, by1))
            overlap_ratio = inter_h / max(1.0, min(float(cy2 - cy1), float(by2 - by1)))

            if gap <= merge_gap_thr and overlap_ratio >= 0.65:
                current = clamp_box(
                    [min(cx1, bx1), min(cy1, by1), max(cx2, bx2), max(cy2, by2)],
                    10 ** 9,
                    10 ** 9,
                )
            else:
                merged.append(current)
                current = box

        merged.append(current)

    return merged


def detect_word_boxes_auto(image):
    if image is None or image.size == 0:
        return []

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    heights = []
    widths = []
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh)
    for i in range(1, num_labels):
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 20:
            heights.append(h)
            widths.append(w)

    if not heights:
        return []

    median_height = int(np.median(heights))
    median_width = int(np.median(widths)) if widths else 1

    too_large = []
    words = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if 5 + 0.2 * median_height <= h <= 3.0 * median_height:
            if w > 5:
                words.append((x, y, w, h))
        else:
            too_large.append((x, y, w, h))
            roi = thresh[y:y + h, x:x + w]
            if roi.size == 0:
                continue
            split = split_lines_from_roi(roi, x, y, median_height)

            most_safe = [
                (bx, by, bw, bh)
                for (bx, by, bw, bh) in split
                if not bw >= 5 or bh >= 5 + 0.3 * median_height
            ]
            words.extend(most_safe)

    if not words:
        return []

    prepared = [
        {
            "word": {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "cx": float(x + w / 2),
                "cy": float(y + h / 2),
            },
            "x1": float(x),
            "y1": float(y),
            "x2": float(x + w),
            "y2": float(y + h),
            "h": float(max(1, h)),
            "cy": float(y + h / 2),
        }
        for (x, y, w, h) in words
    ]

    base_tol = max(4.0, 0.45 * float(max(1, median_height)))
    prepared.sort(key=lambda d: (d["cy"], d["x1"]))

    lines = []
    for entry in prepared:
        best_idx = None
        best_score = 1e9

        for i, line in enumerate(lines):
            line_h = max(1.0, line["y2"] - line["y1"])
            tol = max(base_tol, 0.30 * max(entry["h"], line_h))
            dist = abs(entry["cy"] - line["cy"])

            overlap_h = max(0.0, min(entry["y2"], line["y2"]) - max(entry["y1"], line["y1"]))
            overlap_ratio = overlap_h / max(1.0, min(entry["h"], line_h))

            if dist <= tol or overlap_ratio >= 0.45:
                score = dist - (overlap_ratio * base_tol)
                if score < best_score:
                    best_score = score
                    best_idx = i

        if best_idx is None:
            lines.append(
                {
                    "items": [entry],
                    "y1": entry["y1"],
                    "y2": entry["y2"],
                    "cy": entry["cy"],
                }
            )
        else:
            line = lines[best_idx]
            line["items"].append(entry)
            n = float(len(line["items"]))
            line["y1"] = min(line["y1"], entry["y1"])
            line["y2"] = max(line["y2"], entry["y2"])
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
        merge_tol = max(2.0, 0.18 * max(prev_h, line_h), 0.20 * float(max(1, median_height)))
        center_dist = abs(line["cy"] - prev["cy"])

        overlap_h = max(0.0, min(line["y2"], prev["y2"]) - max(line["y1"], prev["y1"]))
        overlap_ratio = overlap_h / max(1.0, min(prev_h, line_h))

        if center_dist <= merge_tol and overlap_ratio >= 0.35:
            prev["items"].extend(line["items"])
            prev["y1"] = min(prev["y1"], line["y1"])
            prev["y2"] = max(prev["y2"], line["y2"])
            prev["cy"] = float(np.mean([e["cy"] for e in prev["items"]]))
        else:
            merged.append(line)

    ordered = []
    merged.sort(key=lambda l: l["cy"])
    for line in merged:
        line["items"].sort(key=lambda e: e["x1"])
        ordered.extend(e["word"] for e in line["items"])

    boxes = []
    for word in ordered:
        x1 = int(word["x"])
        y1 = int(word["y"])
        x2 = int(word["x"] + word["w"])
        y2 = int(word["y"] + word["h"])
        boxes.append({"box": clamp_box([x1, y1, x2, y2], width, height)})

    return boxes




def draw_boxes_preview(image, boxes, selected_idx, temp_box=None):
    preview = image.copy()
    for i, box_data in enumerate(boxes):
        x1, y1, x2, y2 = box_data["box"]
        color = (0, 255, 255) if i == selected_idx else (0, 200, 0)
        thickness = 2 if i == selected_idx else 1
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, thickness)
        label = f"{i:03d}"
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
            box_data = {
                "box": clamp_box([x1, y1, x2, y2], img_w, img_h),
            }
            
            # Wczytaj dodatkowe informacje o wymiarach jeśli istnieją
            if "width" in row and "height" in row:
                box_data["width"] = row["width"]
                box_data["height"] = row["height"]
            else:
                w, h = get_bbox_dimensions(box_data["box"])
                box_data["width"] = w
                box_data["height"] = h
            
            if "length_category" in row:
                box_data["length_category"] = row["length_category"]
            else:
                box_data["length_category"] = categorize_bbox_by_length(box_data["box"])
            
            loaded_boxes.append(box_data)

    return loaded_boxes if loaded_boxes else None




def edit_boxes_interactive(image, boxes):
    height, width = image.shape[:2]
    selected_idx = 0 if boxes else -1
    step = 2
    min_new_box_size = 8
    preview_scale = get_preview_scale(image)
    zoom = 1.0

    preview_original = image
    preview_preprocessed = build_detection_preview(image)
    if preview_preprocessed is not None and preview_preprocessed.ndim == 2:
        preview_preprocessed = cv2.cvtColor(preview_preprocessed, cv2.COLOR_GRAY2BGR)

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
        "view_mode": "original",
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
    print("a - automatycznie wykryj boxy ponownie")
    print("v - przełącz widok: oryginał / preprocessing")
    print("ENTER - zatwierdź, q - anuluj edycję")

    window_name = "Korekta bounding boxow"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    def get_display_image_and_scale():
        display_scale = preview_scale * state["zoom"]
        base = preview_original if state["view_mode"] == "original" else preview_preprocessed
        if base is None:
            base = preview_original
        if display_scale == 1.0:
            return base, display_scale
        display_image = cv2.resize(base, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA if display_scale < 1.0 else cv2.INTER_CUBIC)
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
                    boxes.append({"box": [x1, y1, x2, y2]})
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

        if key == ord("a"):
            print("\nPonowna automatyczna detekcja boxow...")
            auto_boxes = detect_word_boxes_auto(image)
            if auto_boxes:
                boxes.clear()
                boxes.extend(auto_boxes)
                selected_idx = 0
                state["selected_idx"] = 0
                print(f"Wykryto {len(boxes)} boxow automatycznie.")
            else:
                print("Nie wykryto boxow - pozostawiono obecne.")
            continue

        if key == ord("v"):
            state["view_mode"] = "preprocessed" if state["view_mode"] == "original" else "original"
            print(f"Widok: {state['view_mode']}")
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
            boxes.append({"box": new_box})
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


def process_letter(image_path, base_dir="inference", enable_box_edit=True, non_interactive=False):
    if not os.path.exists(image_path):
        print(f"Błąd: Plik '{image_path}' nie istnieje.")
        return None

    letter_name = os.path.splitext(os.path.basename(image_path))[0]

    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        print(f"Błąd: nie udało się wczytać obrazu '{image_path}'.")
        return None

    img_h, img_w = img_cv2.shape[:2]

    previous_dir = get_latest_output_dir(base_dir, letter_name)
    boxes = load_boxes_from_annotations(previous_dir, img_w, img_h)

    if boxes:
        print(f"Wczytano {len(boxes)} poprzednich boxów z: {previous_dir}")
    else:
        print("Brak poprzednich adnotacji - automatycznie wykrywam wyrazy (bez OCR).")
        boxes = detect_word_boxes_auto(img_cv2)
        print(f"Wykryto automatycznie {len(boxes)} bboxów.")
        if not boxes and not enable_box_edit:
            boxes = [{"box": [0, 0, img_w, img_h]}]
            print("Nie wykryto bboxów. Tryb bez edycji: utworzono 1 box obejmujący cały obraz.")

    edited_boxes = boxes
    if enable_box_edit:
        edited_boxes = edit_boxes_interactive(img_cv2, boxes)
        if edited_boxes is None:
            print("Anulowano edycję bboxów. Nic nie zapisano.")
            return None
        edited_boxes = sort_boxes_reading_order(edited_boxes)
        print("Ponownie posortowano bboxy po ręcznej edycji (kolejność czytania).")

    if not edited_boxes:
        print("Brak boxów do zapisania. Dodaj bboxy i spróbuj ponownie.")
        return None

    edited_boxes = add_length_to_boxes(edited_boxes)
    
    # Analiza długości bboxów
    length_stats = analyze_bbox_lengths(edited_boxes)
    if length_stats:
        print(f"\nStatystyki długości bboxów:")
        print(f"  Liczba: {length_stats['count']}")
        print(f"  Min: {length_stats['min']:.1f}, Max: {length_stats['max']:.1f}")
        print(f"  Średnia: {length_stats['mean']:.1f}, Mediana: {length_stats['median']:.1f}")
        print(f"  P25: {length_stats['p25']:.1f}, P50: {length_stats['p50']:.1f}, P75: {length_stats['p75']:.1f}")

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

    source_image_copy = os.path.join(output_dir, "source_image.jpg")
    cv2.imwrite(source_image_copy, img_cv2, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Zapisano oryginalne zdjęcie: {source_image_copy}")

    # Otwórz plik boxes.jsonl do zapisu
    jsonl_path = os.path.join(output_dir, "boxes.jsonl")
    jsonl_file = open(jsonl_path, "w", encoding="utf-8")

    counter = 0

    for box_data in edited_boxes:
        xmin, ymin, xmax, ymax = box_data["box"]
        crop_img = img_cv2[ymin:ymax, xmin:xmax]

        if crop_img.size == 0:
            continue

        if crop_img.ndim == 3:
            crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        else:
            crop_gray = crop_img.copy()

        file_name = f"word_{counter:03d}.png"
        save_path = os.path.join(output_dir, file_name)
        cv2.imwrite(save_path, crop_gray, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        width = box_data.get("width", xmax - xmin)
        height = box_data.get("height", ymax - ymin)
        length_cat = box_data.get("length_category", "unknown")
        
        print(f"{file_name} | bbox=({xmin},{ymin},{xmax},{ymax}) | w={width}, h={height}, cat={length_cat}")
        
        # Zapisz bbox do jsonl
        bbox_entry = {
            "file_name": file_name,
            "bbox_xyxy": [xmin, ymin, xmax, ymax],
            "width": int(width),
            "height": int(height),
            "length_category": length_cat
        }
        jsonl_file.write(json.dumps(bbox_entry, ensure_ascii=False) + "\n")
        
        counter += 1

    jsonl_file.close()
    print(f"Zapisano bounding boxy do: {jsonl_path}")

    print(f"\nGotowe! Wszystkie wycinki z '{letter_name}' znajdziesz w: {output_dir}")
    return output_dir
