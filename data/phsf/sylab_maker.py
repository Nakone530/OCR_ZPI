from PIL import Image
import os
import random
import csv
import math
import re
import numpy as np

BASELINE_Y = 0

LETTER_SCALE = {
    "a": 0.6,
    "b": 1.1,
    "c": 0.6,
    "d": 1.1,
    "e": 0.6,
    "f": 1.1,
    "g": 0.9,
    "h": 1.1,
    "i": 0.6,
    "j": 0.8,
    "k": 1.1,
    "l": 1.1,
    "m": 0.6,
    "n": 0.6,
    "o": 0.6,
    "p": 0.9,
    "q": 0.9,
    "r": 0.6,
    "s": 0.6,
    "t": 1.1,
    "u": 0.6,
    "v": 0.6,
    "w": 0.6,
    "x": 0.6,
    "y": 0.9,
    "z": 0.6,
    "ą": 0.9,
    "ć": 0.8,
    "ę": 0.9,
    "ł": 1.1,
    "ń": 0.8,
    "ó": 0.8,
    "ś": 0.8,
    "ź": 0.8,
    "ż": 0.8,
    "0": 1.1,
    "1": 1.1,
    "2": 1.1,
    "3": 1.1,
    "4": 1.1,
    "5": 1.1,
    "6": 1.1,
    "7": 1.1,
    "8": 1.1,
    "9": 1.1,
}
LETTER_BASELINE = {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 0,
    "7": 0,
    "8": 0,
    "9": 0,
    "a": 0,
    "b": 0,
    "c": 0,
    "d": 0,
    "e": 0,
    "f": 0,
    "g": 12,
    "h": 0,
    "i": 0,
    "j": 8,
    "k": 0,
    "l": 0,
    "m": 0,
    "n": 0,
    "o": 0,
    "p": 12,
    "q": 12,
    "r": 0,
    "s": 0,
    "t": 0,
    "u": 0,
    "v": 0,
    "w": 0,
    "x": 0,
    "y": 12,
    "z": 0,
    "ą": 12,
    "ć": 0,
    "ę": 12,
    "ł": 0,
    "ń": 0,
    "ó": 0,
    "ś": 0,
    "ź": 0,
    "ż": 0,
}
letters_dir = "./znaki/png"
output_dir = "./syllables"
mapping = os.path.join(letters_dir, "numeracja.csv")
sylaby = "syllables.csv"
print(mapping)
folder_to_char = {}

def crop_horizontal_whitespace(img):
    pixels = img.load()
    w, h = img.size

    left = 0
    while left < w:
        if any(pixels[left, y][:3] != (255, 255, 255) for y in range(h)):
            break
        left += 1

    right = w - 1
    while right >= 0:
        if any(pixels[right, y][:3] != (255, 255, 255) for y in range(h)):
            break
        right -= 1

    if left > right:
        return img

    return img.crop((left, 0, right + 1, h))


with open(mapping, encoding="utf-8") as f:
    for row in csv.reader(f, delimiter=";"):
        if len(row) != 2:
            continue

        folder, char = row
        folder_to_char[folder.strip()] = char.strip()

        
os.makedirs(output_dir, exist_ok=True)

# wczytaj ścieżki do wszystkich wariantów liter
letters = {}

for folder_name in os.listdir(letters_dir):
    folder_path = os.path.join(letters_dir, folder_name)

    if not os.path.isdir(folder_path):
        continue

    if folder_name not in folder_to_char:
        continue

    char = folder_to_char[folder_name]

    letters[char] = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


# ile wariantów każdej sylaby wygenerować
samples_per_syllable = 100

syllables = []

with open(sylaby, encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=";")

    for syllable, freq in reader:
        syllables.append((syllable, int(freq)))

margin = 2  # własny odstęp w pikselach
for syllable, freq in syllables:

    total_samples = math.floor(samples_per_syllable * math.sqrt(freq))

    print(f"{syllable}: {total_samples} próbek")

    for sample_idx in range(total_samples):

        images = []

        for char in syllable:

            

            img_path = random.choice(letters[char])
            
            img_o = Image.open(img_path).convert("RGBA")
            scale = LETTER_SCALE.get(char.lower(), 1.0)

            new_h = int(img_o.height * scale)
            new_w = int(img_o.width * scale)

            img = img_o.resize((new_w, new_h), Image.Resampling.LANCZOS)

            images.append((char, img))


        x = 0
        cropped_images = [(char, crop_horizontal_whitespace(img)) for char, img in images]

        width = sum(img.width for char, img in cropped_images)
        width += margin * (len(cropped_images) - 1)

        height = max(img.height + LETTER_BASELINE[char] for char, img in cropped_images)

        result = Image.new("RGB", (width, height), (255, 255, 255))
        baseline = max(LETTER_BASELINE[char] for char, img in cropped_images)
        for char, img in cropped_images:
            y = LETTER_BASELINE[char]
            result.paste(img, (x, (height - img.height + y) - baseline), img)
            x += img.width + margin

        syl_dir = os.path.join(output_dir,syllable)
        os.makedirs(syl_dir, exist_ok=True)
            
        # Dodanie lekkiego szumu (Gaussian noise)
        img_arr = np.array(result, dtype=np.float32)
        noise = np.random.normal(loc=0, scale=10, size=img_arr.shape)
        noisy_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
        result = Image.fromarray(noisy_arr)

        result.save(
            os.path.join(
                syl_dir,
                f"{syllable}_{sample_idx:04d}.png"
            )
        )


def natural_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', s)
    ]

BASE_DIR = "./syllables"
CSV_FILE = os.path.join(BASE_DIR, "numeracja.csv")

folders = [
    name for name in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, name))
]

folders.sort(key=natural_key)


with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    for idx, folder in enumerate(folders):
        writer.writerow([folder, str(idx)])

print(f"Zapisano {len(folders)} wpisów do {CSV_FILE}")


with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)

    for row_num, row in enumerate(reader, start=1):
        if len(row) < 2:
            print(f"Pominięto wiersz {row_num}: za mało kolumn")
            continue

        old_name = row[0].strip()
        new_name = row[1].strip()

        old_path = os.path.join(BASE_DIR, old_name)
        new_path = os.path.join(BASE_DIR, new_name)

        if not os.path.isdir(old_path):
            print(f"Nie istnieje folder: {old_name}")
            continue

        if os.path.exists(new_path):
            print(f"Docelowa nazwa już istnieje: {new_name}")
            continue

        os.rename(old_path, new_path)
        print(f"{old_name} -> {new_name}")

