import os
import glob
import argparse
import json
import cv2
import numpy as np


def collect_images(path):
    if os.path.isfile(path):
        return [path]

    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.webp"]

    files = []

    for ext in exts:
        files.extend(
            glob.glob(
                os.path.join(path, "**", ext),
                recursive=True
            )
        )

    return sorted(files)

def resize_keep_ratio(
    img,
    max_h=900,
    max_w=1600,
    min_h=300
):
    h, w = img.shape[:2]

    # skala ograniczona wysokością
    scale_h = max_h / h

    # skala ograniczona szerokością
    scale_w = max_w / w

    # wybierz bardziej restrykcyjną
    scale = min(scale_h, scale_w)

    # nie zmniejszaj za bardzo małych obrazów
    if h * scale < min_h:
        scale = min_h / h

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h))



def visualize_connected_components(thresh):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    # kolorowy obraz wyjściowy
    output = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)

    rng = np.random.default_rng(42)
    colors = rng.integers(0, 255, size=(num_labels, 3))

    # tło = czarne
    colors[0] = [0, 0, 0]

    h, w = thresh.shape

    for y in range(h):
        for x in range(w):
            label = labels[y, x]
            output[y, x] = colors[label]

    return output, num_labels, stats

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

def detect_color_blocks(img, min_area=600, chroma_thresh=6):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    h, w = l.shape
    y0 = int(h * 0.15)
    y1 = int(h * 0.85)
    x0 = int(w * 0.15)
    x1 = int(w * 0.85)

    a_bg = float(np.median(a[y0:y1, x0:x1]))
    b_bg = float(np.median(b[y0:y1, x0:x1]))

    da = a.astype(np.float32) - a_bg
    db = b.astype(np.float32) - b_bg
    chroma = np.sqrt(da * da + db * db)

    color_mask = (chroma > chroma_thresh) & (l > 40) & (l < 245)
    color_mask = color_mask.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(color_mask)

    blocks = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        blocks.append((x, y, w, h))

    return color_mask, blocks

def normalize_illumination(gray, ksize=51):
    if ksize % 2 == 0:
        ksize += 1

    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    blur = np.clip(blur, 1, 255).astype(np.uint8)
    norm = cv2.divide(gray, blur, scale=255)

    return norm

def threshold_by_color_blocks(img, gray, base_thresh, min_area=600, chroma_thresh=6):
    color_mask, blocks = detect_color_blocks(
        img,
        min_area=min_area,
        chroma_thresh=chroma_thresh
    )

    refined = base_thresh.copy()
    per_block_thresholds = []

    for (x, y, w, h) in blocks:
        roi_gray = gray[y:y+h, x:x+w]
        roi_mask = color_mask[y:y+h, x:x+w]

        if np.count_nonzero(roi_mask) < 20:
            continue

        # Otsu w obrębie bloku kolorowego
        t, _ = cv2.threshold(
            roi_gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        per_block_thresholds.append(((x, y, w, h), float(t)))

        _, roi_thresh = cv2.threshold(
            roi_gray,
            t,
            255,
            cv2.THRESH_BINARY_INV
        )

        refined[y:y+h, x:x+w] = roi_thresh

    return refined, color_mask, blocks, per_block_thresholds

def detect_text_lines(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Nie można wczytać obrazu: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Korekcja nierownego oswietlenia
    illum_ksize = 51
    gray_norm = normalize_illumination(gray, ksize=illum_ksize)

    # Binaryzacja bazowa (globalna)
    _, base_thresh = cv2.threshold(
        gray_norm,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Binaryzacja z lokalnym progiem w blokach kolorowych
    color_min_area = 600
    color_chroma_thresh = 6
    thresh, color_mask, color_blocks, per_block_thresholds = \
        threshold_by_color_blocks(
            img,
            gray_norm,
            base_thresh,
            min_area=color_min_area,
            chroma_thresh=color_chroma_thresh
        )
    

    # Łączenie znaków w poziome linie
    kernel_size = (10, 2)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        kernel_size
    )

    dilation_iterations = 1
    dilated = cv2.dilate(thresh, kernel, iterations=dilation_iterations)

    # Szukanie konturów
    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    heights = []
    widths = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    cc_min_area = 20
    for i in range(1, num_labels):
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        area = stats[i, cv2.CC_STAT_AREA]

        if area > cc_min_area:
            heights.append(h)
            widths.append(w)

    median_height = int(np.median(heights))
    median_width = int(np.median(widths))
    too_large = []
    lines = []
    min_w = 5
    min_h_base = 5
    min_h_ratio = 0.2
    max_h_ratio = 3.0
    for cnt in contours:
        
        x, y, w, h = cv2.boundingRect(cnt)

        # Odrzucenie małych elementów
        if min_h_base + min_h_ratio * median_height <= h <= max_h_ratio * median_height:
            print(" ")
            if w > min_w:
                lines.append((x, y, w, h))
        else:
            too_large.append((x, y, w, h))

            roi = thresh[y:y+h, x:x+w]
            print("TOO LARGE:", (x, y, w, h))
            split = split_lines_from_roi(roi, x, y, median_height)
            
            too_small = [
                (bx, by, bw, bh) 
                for (bx, by, bw, bh) in split
                    if bw < min_w or bh < min_h_base + 0.3 * median_height
            ]

            most_safe = [
                (bx, by, bw, bh)
                for (bx, by, bw, bh) in split
                    if not bw >= min_w or bh >= min_h_base + 0.3 * median_height
            ]
            center_y = y + h / 2
            center_x = x + w / 2
            
            
            #lines.extend(too_small)
            lines.extend(most_safe)
        
    # Sortowanie od góry do dołuedian_heig
    lines = sorted(lines, key=lambda b: b[1])

    # Kopia do rysowania
    output = img.copy()

    for i, (x, y, w, h) in enumerate(lines):
        # Zielony prostokąt
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Numer linii
        cv2.putText(
            output,
            f"{i}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # Oznaczenie bloków kolorowych + próg
    for (x, y, w, h) in color_blocks:
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    for (x, y, w, h), t in per_block_thresholds:
        cv2.putText(
            output,
            f"T={int(round(t))}",
            (x, max(10, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    report = {
        "image_path": image_path,
        "image_shape": [int(img.shape[0]), int(img.shape[1])],
        "grayscale": {"method": "cv2.COLOR_BGR2GRAY"},
        "illumination_correction": {
            "method": "divide_by_gaussian",
            "ksize": int(illum_ksize)
        },
        "binarization_global": {
            "method": "otsu",
            "type": "THRESH_BINARY_INV + THRESH_OTSU"
        },
        "color_blocks": {
            "space": "LAB",
            "min_area": int(color_min_area),
            "chroma_thresh": float(color_chroma_thresh),
            "morph_kernel": [5, 5],
            "morph_open_iter": 1,
            "morph_close_iter": 2,
            "count": int(len(color_blocks))
        },
        "binarization_color_blocks": {
            "method": "otsu_per_block",
            "thresholds": [
                {
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "threshold": float(t)
                }
                for (x, y, w, h), t in per_block_thresholds
            ]
        },
        "morphology": {
            "dilate_kernel": [int(kernel_size[0]), int(kernel_size[1])],
            "dilate_iterations": int(dilation_iterations)
        },
        "connected_components": {
            "min_area": int(cc_min_area),
            "median_height": int(median_height),
            "median_width": int(median_width)
        },
        "line_filtering": {
            "min_w": int(min_w),
            "min_h_base": int(min_h_base),
            "min_h_ratio": float(min_h_ratio),
            "max_h_ratio": float(max_h_ratio)
        }
    }

    return output, thresh, dilated, lines, color_mask, report



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Wykrywanie linii tekstu"
    )

    parser.add_argument(
        "input",
        type=str,
        help="Obraz lub folder"
    )

    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Tryb debug"
    )

    args = parser.parse_args()

    image_paths = collect_images(args.input)

    if not image_paths:
        print("Nie znaleziono obrazów")
        exit(1)

    for image_path in image_paths:

        print(f"\n=== {image_path} ===")

        output, thresh, dilated, lines, color_mask, report = \
            detect_text_lines(image_path)

        print(f"Znaleziono {len(lines)} linii")

        print("Parametry przetwarzania:")
        print(json.dumps(report, ensure_ascii=True, indent=2))

        report_path = os.path.splitext(image_path)[0] + "_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=True, indent=2)
        print(f"Zapisano raport: {report_path}")

        for i, (x, y, w, h) in enumerate(lines):
            print(f"{i}: x={x}, y={y}, w={w}, h={h}")

        if not args.debug:
            continue

        original = cv2.imread(image_path)

        cc_vis, _, _ = visualize_connected_components(thresh)

        color_mask_bgr = cv2.cvtColor(
            color_mask,
            cv2.COLOR_GRAY2BGR
        )

        cc_vis_d, _, _ = \
            visualize_connected_components(dilated)

        thresh_bgr = cv2.cvtColor(
            thresh,
            cv2.COLOR_GRAY2BGR
        )

        dilated_bgr = cv2.cvtColor(
            dilated,
            cv2.COLOR_GRAY2BGR
        )

        debug_views = [
            ("Output", output),
            ("Original", original),
            ("Color Blocks", color_mask_bgr),
            ("Binaryzacja", thresh_bgr),
            ("Dylatacja", dilated_bgr),
            ("CC", cc_vis),
            ("CC Dilated", cc_vis_d),
        ]

        debug_views = [
            (name, resize_keep_ratio(img))
            for name, img in debug_views
        ]

        idx = 0

        while True:

            name, img = debug_views[idx]

            display = img.copy()

            cv2.putText(
                display,
                f"{idx+1}/{len(debug_views)} : {name}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Debug", display)

            key = cv2.waitKey(0)

            # ESC
            if key == 27 or key == ord('q'):
                cv2.destroyAllWindows()
                exit(0)

            # D / strzałka w prawo
            elif key in [ord('d'), 83]:
                idx = (idx + 1) % len(debug_views)

            # A / strzałka w lewo
            elif key in [ord('a'), 81]:
                idx = (idx - 1) % len(debug_views)

            # ENTER -> następny obraz
            elif key == 13:
                break

    cv2.destroyAllWindows()
