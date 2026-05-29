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

def detect_text_words(img):
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
    too_large = []
    words = []
    for cnt in contours:
        
        x, y, w, h = cv2.boundingRect(cnt)

        # Odrzucenie małych elementów
        if 5 + 0.2 * median_height <= h <= 3.0 * median_height:
            print(" ")
            if w > 5:
                words.append((x, y, w, h))
        else:
            too_large.append((x, y, w, h))

            roi = thresh[y:y+h, x:x+w]
            print("TOO LARGE:", (x, y, w, h))
            split = split_lines_from_roi(roi, x, y, median_height)
            
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
            
            
            #words.extend(too_small)
            words.extend(most_safe)
        
    # Konwersja do obiektów Word
    words = [
        Word(0, box)
        for box in words
    ]

    # Grupowanie w linie: dopasowanie po cy + pionowym overlapie (bardziej stabilne)
    if words:
        prepared = [
            {
                "word": w,
                "x1": float(w.x),
                "y1": float(w.y),
                "x2": float(w.x + w.w),
                "y2": float(w.y + w.h),
                "h": float(max(1, w.h)),
                "cy": float(w.cy),
            }
            for w in words
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

        # Scal linie, które są praktycznie tą samą linią (po podziale na fragmenty)
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

        # Sortowanie finalne: linie góra->dół, słowa w linii lewo->prawo
        ordered = []
        merged.sort(key=lambda l: l["cy"])
        for line in merged:
            line["items"].sort(key=lambda e: e["x1"])
            ordered.extend(e["word"] for e in line["items"])

        for idx, w in enumerate(ordered, start=1):
            w.idx = idx

        # Utrzymaj listę words również w kolejności czytania (ważne dla dalszego pipeline)
        words = ordered

    # Kopia do rysowania
    output = img.copy()

    for i, word in enumerate(words):
        
        #Zielony prostokąt
        cv2.rectangle(
            output,
            (word.x, word.y),
            (word.x + word.w, word.y + word.h),
            (0, 255, 0),
            2
        )
        #Numer lini
        cv2.putText(
            output,
            f"{word.idx}",
            (word.x, word.y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    return output, thresh, dilated, words, output.copy()

class Word:
    def __init__(self, idx, box):
        self.idx = idx

        self.x, self.y, self.w, self.h = box

        self.cx = self.x + self.w / 2
        self.cy = self.y + self.h / 2

        self.used = False

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.w

    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.h

    @property
    def center(self):
        return (int(self.cx), int(self.cy))
    
    @property
    def box(self):
        return (self.x, self.y, self.w, self.h)
    
    def __iter__(self):
        return iter((self.x, self.y, self.w, self.h))

    def __repr__(self):
        return (
            f"Word(idx={self.idx}, "
            f"x={self.x}, y={self.y}, "
            f"w={self.w}, h={self.h})"
        )

def candidate_neighbors(word, words, median_h):

    candidates = []

    for other in words:

        if other.idx == word.idx:
            continue

        dx = other.cx - word.cx
        dy = other.cy - word.cy

        if dx <= 0:
            continue

        if dx > median_h * 15:
            continue

        if abs(dy) > median_h * 1.5:
            continue

        candidates.append((other, dx, dy))

    return candidates

def neighbor_score(word, other):

    dx = other.cx - word.cx
    dy = other.cy - word.cy

    slope = abs(dy / max(dx, 1))

    height_ratio = abs(other.h - word.h) / max(word.h, 1)

    score = (
        abs(dy) * 2.0 +
        slope * 30.0 +
        dx * 0.1 +
        height_ratio * 25.0
    )

    return score

def best_neighbor(word, words, median_h):

    candidates = candidate_neighbors(
        word,
        words,
        median_h
    )

    if not candidates:
        return None

    scored = []

    for other, dx, dy in candidates:

        score = neighbor_score(word, other)

        scored.append((score, other))

    scored.sort(key=lambda x: x[0])

    return scored[0][1]

def grow_line(start_word, words, median_h):

    line = [start_word]

    current = start_word

    current.used = True

    while True:

        nxt = best_neighbor(
            current,
            words,
            median_h
        )

        if nxt is None:
            break

        if nxt.used:
            break

        nxt.used = True

        line.append(nxt)

        current = nxt

    return line


def detect_text_lines(img, output, words):
    lines = []
    true_l  = img.copy()
    false_l = img.copy()
    heights = [w.h for w in words]
    median_h = np.median(heights)
    for word in words:

        if word.used:
            continue

        line = grow_line(
            word,
            words,
            median_h
        )

        if len(line) > 0:
            lines.append(line)

    for line in lines:

        for i in range(len(line) - 1):

            w1 = line[i]
            w2 = line[i + 1]
            
            pc1 = (int(w1.cx), int(w1.cy))
            pc2 = (int(w2.cx), int(w2.cy))
            
            y1 = int(w1.cy)
            y2 = int(w2.cy)


            p1 = (int(w1.x), y1)
            p2 = (int(w1.x + w1.w), y1)

            cv2.line(output, p1, p2, (0, 0, 255), 2)
            cv2.line(img, p1, p2, (0, 0, 255), 2)
            cv2.line(false_l, p1, p2, (0, 0, 255), 2)
            
            p3 = (int(w1.x + w1.w), y1)
            p4 = (int(w2.x), y2)

            cv2.line(output, p3, p4, (0, 0, 255), 2)
            cv2.line(img, p3, p4, (0, 0, 255), 2)
            cv2.line(false_l, p3, p4, (0, 0, 255), 2)

            p5 = (int(w2.x), y2)
            p6 = (int(w2.x + w2.w), y2)

            cv2.line(output, p5, p6, (255, 0, 0), 2)
            cv2.line(img, p5, p6, (255, 0, 0), 2)
            cv2.line(false_l, p5, p6, (0, 0, 255), 2)

            cv2.line(output, pc1, pc2, (255, 0, 0), 2)
            cv2.line(img, pc1, pc2, (255, 0, 0), 2)
            cv2.line(true_l, pc1, pc2, (255, 0, 0), 2)
            
    return img, output, true_l, false_l, lines


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
        original = cv2.imread(image_path)

        if original is None:
            raise ValueError(f"Nie można wczytać obrazu: {image_path}")
        
        output, thresh, dilated, words, w_output = detect_text_words(original.copy())

        l_output, output, true_l_output, false_l_output, lines  = detect_text_lines(original.copy(), output, words)        
        print(f"Znaleziono {len(words)} linii")

        for i, w in enumerate(words):
            print(f"{i}: idx={w.idx}, x={w.x}, y={w.y}, w={w.w}, h={w.h}")

        # Dodatkowo: wydrukuj słowa posortowane według idx (kolejność czytania)
        print("--- Reading order ---")
        for w in sorted(words, key=lambda x: x.idx):
            print(f"idx={w.idx}: x={w.x}, y={w.y}, w={w.w}, h={w.h}")

        if not args.debug:
            continue


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
            ("Wyrazy", w_output),
            ("Linie", l_output),
            ("Prawdziwe Linie", true_l_output),
            ("Fałszywe Linie", false_l_output),
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
