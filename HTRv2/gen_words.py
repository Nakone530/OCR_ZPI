import os
import sys
import random
import re
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import albumentations as A

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STRIP_RE = re.compile(r"^\W+|\W+$", re.UNICODE)

WIDTH, HEIGHT = 512, 64
FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")
OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "output_words")
os.makedirs(OUTPUT_BASE, exist_ok=True)

FONT_PATHS = sorted([
    os.path.join(FONTS_DIR, f) for f in os.listdir(FONTS_DIR) if f.endswith(".ttf")
])

def get_augmentation_pipeline():
    return A.Compose([
        A.Affine(
            translate_percent={"x": (-0.03, 0.03), "y": (-0.02, 0.02)},
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            rotate=(-3, 3),
            fill=255,
            p=0.8,
        ),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.4),
        A.ElasticTransform(alpha=1, sigma=50, fill=255, p=0.2),
    ])

def render_text(text, font_path):
    img = Image.new("L", (WIDTH * 2, HEIGHT * 2), 255)
    draw = ImageDraw.Draw(img)
    font_size = 45
    while font_size >= 8:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font_size -= 2
            continue
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= WIDTH - 10:
            break
        font_size -= 2
    x = (WIDTH - tw) // 2
    y = max(0, (HEIGHT - th) // 2 - 8)
    draw.text((x, y), text, font=font, fill=0)
    img = np.array(img)
    coords = cv2.findNonZero(255 - img)
    if coords is not None:
        x_, y_, w_, h_ = cv2.boundingRect(coords)
        img = img[y_:y_+h_, x_:x_+w_]
    h, w = img.shape
    scale = HEIGHT / h
    new_w = int(w * scale)
    if new_w > WIDTH:
        scale = WIDTH / w
        new_h = int(h * scale)
        new_w = WIDTH
    else:
        new_h = HEIGHT
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    x_offset = (WIDTH - img.shape[1]) // 2
    y_offset = (HEIGHT - img.shape[0]) // 2
    canvas[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
    return canvas

def get_next_run(base_dir):
    max_n = 0
    for f in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, f)) and f.isdigit():
            n = int(f)
            if n > max_n:
                max_n = n
    return max_n + 1

def load_words(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    words = []
    for line in lines:
        for w in line.split():
            w = STRIP_RE.sub("", w)
            if w:
                words.append(w)
    return words

def main():
    source_file = sys.argv[1] if len(sys.argv) > 1 else None
    if not source_file:
        print("Usage: python gen_words.py <text_file>")
        return
    words = load_words(source_file)
    random.seed(42)
    random.shuffle(words)
    num_words = 174
    word_list = [words[i % len(words)] for i in range(num_words)]
    num_fonts = len(FONT_PATHS)
    base_per_font = 574 // num_fonts
    extra = 574 - base_per_font * num_fonts
    aug_pipeline = get_augmentation_pipeline()
    run_num = get_next_run(OUTPUT_BASE)
    run_dir = os.path.join(OUTPUT_BASE, str(run_num))
    os.makedirs(run_dir, exist_ok=True)
    total = 0
    for wi, word in enumerate(word_list):
        word_dir = os.path.join(run_dir, str(wi + 1))
        os.makedirs(word_dir, exist_ok=True)
        img_idx = 1
        for fi, font_path in enumerate(FONT_PATHS):
            imgs_per_font = base_per_font + (1 if fi < extra else 0)
            base_img = render_text(word, font_path)
            for _ in range(imgs_per_font):
                aug = aug_pipeline(image=base_img)
                img_aug = aug["image"]
                fname = f"sample_{img_idx}.png"
                Image.fromarray(img_aug).save(os.path.join(word_dir, fname))
                with open(os.path.join(word_dir, "labels.txt"), "a", encoding="utf-8") as f:
                    f.write(f"{fname}\t{word}\n")
                img_idx += 1
                total += 1
        safe_word = word.encode("ascii", errors="replace").decode("ascii")
        print(f"  Word {wi+1}/{num_words}: '{safe_word}' -> {img_idx-1} images")
    print(f"\nDone! Total: {total} images")

if __name__ == "__main__":
    main()
