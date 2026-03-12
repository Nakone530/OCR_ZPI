import os
import numpy as np
from PIL import Image
import argparse

INPUT_IMAGE = "alphabet.jpg"
OUTPUT_DIR = "letters"
#DARKNESS = 128
# alfabet w kolejności na stronie
alphabet = list("ABCDEFGHIJKLMNOPQRSTUWYZabcdefghijklmnopqrstuwyz")

# margines w pikselach (~1mm przy 300dpi ≈ 12px)
MARGIN = 4


def add_margin(img, margin):
    h, w = img.shape
    canvas = np.full((h + 2*margin, w + 2*margin), 255, dtype=np.uint8)
    canvas[margin:margin+h, margin:margin+w] = img
    return canvas

def trim_binary(binary_img):

    rows = np.where(binary_img.sum(axis=1) > 0)[0]
    cols = np.where(binary_img.sum(axis=0) > 0)[0]

    if len(rows) == 0 or len(cols) == 0:
        return binary_img

    return binary_img[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

def segment_letters(image_path, darkness):

    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

    binary = img_array < darkness

    tollerance = 0
    
    horizontal_sum = np.sum(binary, axis=1)

    rows_bounds = []
    in_row = False

    for i, val in enumerate(horizontal_sum):
        if val > tollerance and not in_row:
            start = i
            in_row = True
        elif val == tollerance and in_row:
            rows_bounds.append((start, i))
            in_row = False

    if in_row:
        rows_bounds.append((start, len(horizontal_sum)))

    letters = []

    for (row_start, row_end) in rows_bounds:

        line_img = img_array[row_start:row_end, :]
        binary_line = binary[row_start:row_end, :]

#        binary_line = trim_binary(binary_line)
#        line_img = trim_binary(line_img)
        
        vertical_sum = np.sum(binary_line, axis=0)

        in_letter = False
        letters_bounds = []
        
        for i, val in enumerate(vertical_sum):
            if val > tollerance and not in_letter:
                start = i
                in_letter = True
            elif val == tollerance and in_letter:
                letters_bounds.append((start, i))
                in_letter = False

        if in_letter:
            letters_bounds.append((start, len(vertical_sum)))

        for (col_start, col_end) in letters_bounds:

            letter_region = line_img[:, col_start:col_end]

            # projekcja pozioma wewnątrz litery
            horizontal_sum_letter = np.sum(letter_region < 128, axis=1)

            top = None
            bottom = None

            for i, val in enumerate(horizontal_sum_letter):
                if val > tollerance and top is None:
                    top = i
                if val > tollerance:
                    bottom = i

            if top is not None and bottom is not None:
                letter = letter_region[top:bottom+1, :]
                letters.append(letter)

    return letters


def save_letters(letters):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    counters = {}

    for i, letter_img in enumerate(letters):

        if i >= len(alphabet):
            break

        letter = alphabet[i]

        if letter not in counters:
            counters[letter] = 1

        folder = os.path.join(OUTPUT_DIR, letter)
        os.makedirs(folder, exist_ok=True)

        img = add_margin(letter_img, MARGIN)

        filename = f"{letter}_{counters[letter]:03d}.jpg"

        Image.fromarray(img).save(os.path.join(folder, filename))

        counters[letter] += 1


def main():

    parser = argparse.ArgumentParser(description="OCR - Rozpoznawanie znaków (z opcjonalnym odszumianiem)")
    parser.add_argument("--image", "-i", type=str, help="Ścieżka do zdjęcia do dzielenia")
    parser.add_argument("--darkness", "-d", type=int, help="Jak ciemne są litery")
    args = parser.parse_args()
    

    if args.image:

        letters = segment_letters(args.image, args.darkness)
        
        print("Wykryto liter:", len(letters))

        save_letters(letters)

        print("Zapisano litery.")
    else:
        # Domyślnie: pokaż pomoc
        parser.print_help()
        print("\n" + "=" * 60)
        print("PRZYKŁADY UŻYCIA:")
        print("=" * 60)
        print("1. podziel zdjęcie:     python split.py --image *.png")
        print("=" * 60)

if __name__ == "__main__":
    main()
