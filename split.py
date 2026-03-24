import os
import numpy as np
from PIL import Image
import argparse

# ── Konfiguracja ───────────────────────────────────────────────────────────────
# Domyślna ścieżka obrazu wejściowego
INPUT_IMAGE = "alphabet.jpg"

# Folder wyjściowy dla wyciętych liter
OUTPUT_DIR = "letters"

# Alfabet w kolejności na stronie (do nazewnictwa plików)
alphabet = list("ABCDEFGHIJKLMNOPQRSTUWXYZ")

# Margines w pikselach (~1mm przy 300dpi ≈ 12px)
MARGIN = 4


def add_margin(img: np.ndarray, margin: int) -> np.ndarray:
    """
    Dodaje biały margines wokół obrazu.
    
    Tworzy nowy canvas o większych wymiarach i umieszcza
    oryginalny obraz w jego centrum z białym obramowaniem.
    
    Argumenty:
        img (np.ndarray): Obraz wejściowy w skali szarości.
                          Oczekiwany kształt: (wysokość, szerokość).
        margin (int): Szerokość marginesu w pikselach (dodawana z każdej strony).
    
    Zwraca:
        np.ndarray: Obraz z dodanym białym marginesem.
                    Kształt: (wysokość + 2*margin, szerokość + 2*margin).
    
    Przykład:
        >>> img_with_margin = add_margin(letter_img, 4)
    """
    h, w = img.shape
    canvas = np.full((h + 2*margin, w + 2*margin), 255, dtype=np.uint8)
    canvas[margin:margin+h, margin:margin+w] = img
    return canvas


def trim_binary(binary_img: np.ndarray) -> np.ndarray:
    """
    Przycina obraz binarny usuwając puste wiersze i kolumny.
    
    Znajduje najmniejszy prostokąt zawierający wszystkie
    niezerowe piksele i zwraca ten region.
    
    Argumenty:
        binary_img (np.ndarray): Obraz binarny (0/1 lub 0/255).
    
    Zwraca:
        np.ndarray: Przycięty obraz bez pustych marginesów.
                    Zwraca oryginalny obraz jeśli jest pusty.
    
    Przykład:
        >>> trimmed = trim_binary(binary_letter)
    """
    rows = np.where(binary_img.sum(axis=1) > 0)[0]
    cols = np.where(binary_img.sum(axis=0) > 0)[0]

    if len(rows) == 0 or len(cols) == 0:
        return binary_img

    return binary_img[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]


def segment_letters(image_path: str, darkness: int) -> list[np.ndarray]:
    """
    Segmentuje obraz na pojedyncze litery metodą projekcji.
    
    Algorytm:
      1. Konwertuje obraz do skali szarości i binaryzuje (próg: darkness)
      2. Używa projekcji poziomej do wykrycia wierszy tekstu
      3. Dla każdego wiersza używa projekcji pionowej do wykrycia liter
      4. Przycina każdą literę do jej bounding boxa
    
    Argumenty:
        image_path (str): Ścieżka do obrazu wejściowego.
        darkness (int): Próg binaryzacji (0-255). Piksele ciemniejsze
                        od tej wartości są traktowane jako tekst.
    
    Zwraca:
        list[np.ndarray]: Lista obrazów pojedynczych liter (tablice numpy
                          w skali szarości).
    
    Przykład:
        >>> letters = segment_letters("alphabet.png", darkness=128)
        >>> len(letters)
        26
    
    Uwaga:
        Funkcja oczekuje obrazu z literami ułożonymi w liniach.
        Najlepiej działa na skanach z równomiernym oświetleniem.
    """
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
            horizontal_sum_letter = np.sum(letter_region < darkness, axis=1)

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


def save_letters(letters: list[np.ndarray]) -> None:
    """
    Zapisuje listę obrazów liter do uporządkowanej struktury folderów.
    
    Tworzy folder dla każdej litery (A, B, C, ...) i zapisuje obrazy
    z automatyczną numeracją (A_001.jpg, A_002.jpg, ...).
    
    Argumenty:
        letters (list[np.ndarray]): Lista obrazów liter do zapisania.
    
    Zwraca:
        None
    
    Efekty uboczne:
        - Tworzy folder OUTPUT_DIR jeśli nie istnieje
        - Tworzy podfoldery dla każdej litery
        - Zapisuje pliki JPG z marginesem i numeracją
    
    Przykład:
        >>> letters = segment_letters("alphabet.png", 128)
        >>> save_letters(letters)
        # Tworzy: letters/A/A_001.jpg, letters/B/B_001.jpg, ...
    
    Uwaga:
        Liczba liter jest ograniczona do długości zmiennej 'alphabet'.
        Pliki są numerowane, aby uniknąć nadpisywania istniejących.
    """
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

        counter = counters[letter]

        while True:
            filename = f"{letter}_{counter:03d}.jpg"
            filepath = os.path.join(folder, filename)

            if not os.path.exists(filepath):
                break

            counter += 1

        Image.fromarray(img).save(filepath)

        counters[letter] = counter + 1


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
