import os
import glob
import argparse
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

def detect_text_lines(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Nie można wczytać obrazu: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binaryzacja
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    

    # Łączenie znaków w poziome linie
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (10, 2)
    )

    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Szukanie konturów
    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    heights = []
    widths = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    for i in range(1, num_labels):
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        area = stats[i, cv2.CC_STAT_AREA]

        if area > 20:
            heights.append(h)
            widths.append(w)

    median_height = int(np.median(heights))
    median_width = int(np.median(widths))
    most_safe_bb = []
    too_large = []
    lines = []
    for cnt in contours:
        
        x, y, w, h = cv2.boundingRect(cnt)

        # Odrzucenie małych elementów
        if 5 + 0.2 * median_height <= h <= 3.0 * median_height:
            print(" ")
            if w > 5:
                lines.append((x, y, w, h))
        else:
            too_large.append((x, y, w, h))

            roi = thresh[y:y+h, x:x+w]
            print("TOO LARGE:", (x, y, w, h))
            split = split_lines_from_roi(roi, x, y, median_height)
##            split = [
##                (bx, by, bw, bh)
##                for (bx, by, bw, bh) in split
##                if bw > 5 and bh > 5 + 0.3 * median_height
##            ]
##            lines.extend(split)
            too_small = [
                (bx, by, bw, bh) 
                for (bx, by, bw, bh) in split
                    if bw < 5 or bh < 5 + 0.3 * median_height
            ]

            most_safe = [
                (bx, by, bw, bh)
                for (bx, by, bw, bh) in split
                    if not bw >= 5 or bh >= 5 + 0.3 * median_height
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

    return output, thresh, dilated, lines



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

        output, thresh, dilated, lines = \
            detect_text_lines(image_path)

        print(f"Znaleziono {len(lines)} linii")

        for i, (x, y, w, h) in enumerate(lines):
            print(f"{i}: x={x}, y={y}, w={w}, h={h}")

        if not args.debug:
            continue

        original = cv2.imread(image_path)

        cc_vis, _, _ = visualize_connected_components(thresh)

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
