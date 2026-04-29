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


def detect_word_boxes_auto(image):
    if image is None or image.size == 0:
        return []

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Odsiew drobnego szumu przed wykrywaniem komponentow (znakow).
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, open_kernel, iterations=1)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)

    char_candidates = []
    min_char_area = max(8, (width * height) // 50000)
    max_char_area = max(1200, (width * height) // 220)
    min_char_h = max(6, height // 260)
    max_char_h = int(height * 0.30)
    min_char_w = 2

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_char_area or area > max_char_area:
            continue
        if h < min_char_h or h > max_char_h:
            continue
        if w < min_char_w:
            continue

        char_candidates.append(
            {
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "w": w,
                "h": h,
                "cy": y + (h / 2.0),
            }
        )

    if not char_candidates:
        return []

    char_heights = [c["h"] for c in char_candidates]
    char_widths = [c["w"] for c in char_candidates]
    median_h = float(np.median(np.array(char_heights, dtype=np.float32)))
    median_w = float(np.median(np.array(char_widths, dtype=np.float32)))

    def vertical_overlap_ratio(a, b):
        overlap = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
        denom = max(1.0, min(float(a["h"]), float(b["h"])))
        return overlap / denom

    # Grupowanie znakow w linie.
    line_tol = max(8.0, 0.65 * median_h)
    char_candidates.sort(key=lambda c: (c["cy"], c["x1"]))
    lines = []

    for char in char_candidates:
        best_line_idx = None
        best_dist = 1e9
        for idx, line in enumerate(lines):
            ref = line["ref"]
            dist = abs(char["cy"] - line["cy"])
            if dist <= line_tol or vertical_overlap_ratio(char, ref) >= 0.18:
                if dist < best_dist:
                    best_dist = dist
                    best_line_idx = idx

        if best_line_idx is None:
            lines.append(
                {
                    "items": [char],
                    "cy": char["cy"],
                    "y1": float(char["y1"]),
                    "y2": float(char["y2"]),
                    "ref": char,
                }
            )
        else:
            line = lines[best_line_idx]
            line["items"].append(char)
            n = float(len(line["items"]))
            line["cy"] = ((line["cy"] * (n - 1.0)) + char["cy"]) / n
            line["y1"] = min(line["y1"], float(char["y1"]))
            line["y2"] = max(line["y2"], float(char["y2"]))
            line["ref"] = {
                "y1": int(line["y1"]),
                "y2": int(line["y2"]),
                "h": max(1, int(round(line["y2"] - line["y1"]))),
            }

    # Grupowanie znakow w wyrazy po odstepach poziomych.
    word_boxes = []
    lines.sort(key=lambda l: l["cy"])
    pad_x = max(1, int(round(0.2 * median_w)))
    pad_y = max(1, int(round(0.2 * median_h)))
    min_word_w = max(10, int(round(1.2 * median_w)))
    min_word_h = max(8, int(round(0.7 * median_h)))

    for line in lines:
        items = sorted(line["items"], key=lambda c: c["x1"])
        line_h = max(1.0, line["y2"] - line["y1"])

        positive_gaps = []
        for i in range(1, len(items)):
            raw_gap = float(items[i]["x1"] - items[i - 1]["x2"])
            if raw_gap > 0:
                positive_gaps.append(raw_gap)

        if positive_gaps:
            typical_gap = float(np.percentile(np.array(positive_gaps, dtype=np.float32), 60))
            gap_threshold = max(6.0, 1.6 * typical_gap, 0.55 * median_w, 0.18 * line_h)
        else:
            gap_threshold = max(6.0, 0.55 * median_w, 0.18 * line_h)

        current_group = [items[0]]
        for char in items[1:]:
            prev = current_group[-1]
            gap = float(char["x1"] - prev["x2"])
            overlap_ok = vertical_overlap_ratio(char, prev) >= 0.12

            # Scalaj tylko gdy znaki sa blisko i w tej samej strefie pionowej.
            if gap <= gap_threshold and overlap_ok:
                current_group.append(char)
            else:
                gx1 = min(c["x1"] for c in current_group)
                gy1 = min(c["y1"] for c in current_group)
                gx2 = max(c["x2"] for c in current_group)
                gy2 = max(c["y2"] for c in current_group)
                if (gx2 - gx1) >= min_word_w and (gy2 - gy1) >= min_word_h:
                    word_boxes.append(clamp_box([gx1 - pad_x, gy1 - pad_y, gx2 + pad_x, gy2 + pad_y], width, height))
                current_group = [char]

        if current_group:
            gx1 = min(c["x1"] for c in current_group)
            gy1 = min(c["y1"] for c in current_group)
            gx2 = max(c["x2"] for c in current_group)
            gy2 = max(c["y2"] for c in current_group)
            if (gx2 - gx1) >= min_word_w and (gy2 - gy1) >= min_word_h:
                word_boxes.append(clamp_box([gx1 - pad_x, gy1 - pad_y, gx2 + pad_x, gy2 + pad_y], width, height))

    boxes = [{"box": b} for b in word_boxes]
    return sort_boxes_reading_order(boxes)




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
            loaded_boxes.append(
                {
                    "box": clamp_box([x1, y1, x2, y2], img_w, img_h),
                }
            )

    return loaded_boxes if loaded_boxes else None




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

        file_name = f"word_{counter:03d}.png"
        save_path = os.path.join(output_dir, file_name)
        cv2.imwrite(save_path, crop_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        print(f"{file_name} | bbox=({xmin},{ymin},{xmax},{ymax})")
        
        # Zapisz bbox do jsonl
        bbox_entry = {
            "file_name": file_name,
            "bbox_xyxy": [xmin, ymin, xmax, ymax]
        }
        jsonl_file.write(json.dumps(bbox_entry, ensure_ascii=False) + "\n")
        
        counter += 1

    jsonl_file.close()
    print(f"Zapisano bounding boxy do: {jsonl_path}")

    print(f"\nGotowe! Wszystkie wycinki z '{letter_name}' znajdziesz w: {output_dir}")
    return output_dir
