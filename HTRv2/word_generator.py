import os
import random
import re
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import albumentations as A

WIDTH, HEIGHT = 512, 64
FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_words")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FONT_PATHS = sorted([
    os.path.join(FONTS_DIR, f) for f in os.listdir(FONTS_DIR) if f.endswith(".ttf")
])

POLISH_TEXTS = [
    "Witaj w swiecie generowania danych HTR.",
    "To jest przykladowy tekst do syntetycznego uczenia.",
    "Handwriting Text Recognition wymaga duzo danych.",
    "Generowanie syntetycznych danych to skuteczna metoda.",
    "Kazda epoka trenowania modelu poprawia dokladnosc.",
    "Sieci neuronowe osiagaja coraz lepsze wyniki.",
    "Augmentacja danych pomaga w generalizacji modelu.",
    "Polskie znaki diakrytyczne to ważny element testow.",
    "Systemy OCR sa wykorzystywane w wielu dziedzinach.",
    "Wspolczesne modele HTR ucza sie na ogromnych zbiorach.",
    "Automatyczne rozpoznawanie pisma recznego to wyzwanie.",
    "Jakosc danych treningowych ma kluczowe znaczenie.",
    "Preprocessing obrazu poprawia skutecznosc rozpoznawania.",
    "Modele oparte na transformatorach rewolucjonizuja HTR.",
    "Trenowanie modeli wymaga odpowiedniej mocy obliczeniowej.",
    "Zastosowania HTR obejmuja digitalizacje dokumentow.",
    "Rekurencyjne sieci neuronowe sa czesto uzywane w HTR.",
    "Mechanizm uwagi poprawia dokladnosc transkrypcji.",
    "Dane syntetyczne uzupelniaja rzeczywiste zbiory danych.",
    "Walidacja modelu na rzeczywistych danych jest niezbedna.",
    "Optymalizacja hiperparametrow moze znacząco poprawic wyniki.",
    "Transfer learning przyspiesza trenowanie modeli HTR.",
    "Bledy w transkrypcji moga wynikac z nakladania sie liter.",
    "Segmentacja linii tekstu to pierwszy krok w HTR.",
    "Modele sekwencyjne przetwarzaja tekst znak po znaku.",
    "Syntetyczne obrazy moga zawierac rozne style pisma.",
    "Zmiana czcionki i rozmiaru poprawia roznorodnosc danych.",
    "Szum i znieksztalcenia obrazu symuluja realne warunki.",
    "Krzywizny i pochylenia tekstu sa czeste w dokumentach.",
    "Augmentacja w czasie rzeczywistym oszczedza miejsce na dysku.",
]

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

def get_next_number(output_dir):
    max_n = 0
    for f in os.listdir(output_dir):
        m = re.match(r"sample_(\d+)\.png", f)
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n
    return max_n + 1

def main():
    aug_pipeline = get_augmentation_pipeline()
    next_num = get_next_number(OUTPUT_DIR)
    words_per_font = 174
    random.seed(42)
    words = []
    for line in POLISH_TEXTS:
        words.extend(w.strip(",.!?") for w in line.split())
    random.shuffle(words)
    for font_idx, font_path in enumerate(FONT_PATHS):
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        print(f"Font {font_idx+1}/{len(FONT_PATHS)}: {font_name}")
        for i in range(words_per_font):
            word = words[i % len(words)]
            img = render_text(word, font_path)
            aug = aug_pipeline(image=img)
            img_aug = aug["image"]
            fname = f"sample_{next_num}.png"
            Image.fromarray(img_aug).save(os.path.join(OUTPUT_DIR, fname))
            with open(os.path.join(OUTPUT_DIR, "labels.txt"), "a", encoding="utf-8") as f:
                f.write(f"{fname}\t{word}\n")
            next_num += 1
            if (i + 1) % 30 == 0:
                print(f"  {i+1}/{words_per_font}")

if __name__ == "__main__":
    main()
