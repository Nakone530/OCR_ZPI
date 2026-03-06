"""
Skrypt do uruchamiania lokalnie - przygotowanie danych + rozpoznawanie zdjęć
"""
import numpy as np
import os
import sys
import argparse
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import subprocess
from torchvision import datasets, transforms
from PIL import Image

# ============================================================================
# KONFIGURACJA
# ============================================================================

DATA_URL = "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz"
DATA_DIR = "./data"
ARCHIVE_PATH = os.path.join(DATA_DIR, "EnglishFnt.tgz")
EXTRACTED_DIR = os.path.join(DATA_DIR, "English", "Fnt")
MODEL_PATH = "./model_ocr.pth"

IMAGE_SIZE = 48
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Mapowanie klas na znaki (52 klasy: A-Z, a-z)
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


# ============================================================================
# SIEĆ CNN
# ============================================================================

class SimpleCNN(nn.Module):
    """sieć CNN do rozpoznawania znaków."""

    def __init__(self, num_classes=52):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Blok 1: 3x48x48 -> 32x24x24
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Blok 2: 32x24x24 -> 64x12x12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Blok 3: 64x12x12 -> 128x6x6
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================================
# FUNKCJE POMOCNICZE
# ============================================================================

def download_dataset():
    """Pobiera i rozpakowuje dataset."""
    if not os.path.exists(ARCHIVE_PATH):
        print(f"Pobieranie datasetu z {DATA_URL}...")
        os.makedirs(DATA_DIR, exist_ok=True)

        def progress(block_num, block_size, total_size):
            percent = min(100, block_num * block_size * 100 / total_size)
            print(f"\rPostęp: {percent:.1f}%", end="")

        urllib.request.urlretrieve(DATA_URL, ARCHIVE_PATH, progress)
        print("\nPobrano!")

    if not os.path.exists(EXTRACTED_DIR):
        print("Rozpakowywanie...")
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)
        print("Rozpakowano!")


def get_transform():
    """Zwraca transformację dla obrazów."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Wczytuje wytrenowany model."""
    model = SimpleCNN(num_classes=52)

    if os.path.exists(model_path):
        print(f"Wczytywanie modelu z {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model wczytany!")
    else:
        print(f"UWAGA: Nie znaleziono modelu {model_path}")
        print("Model nie jest wytrenowany - wyniki będą losowe!")
        print("Najpierw uruchom: python run_local.py --train")

    model.to(device)
    model.eval()
    return model


def predict_image(image_path: str, model: nn.Module, device: torch.device) -> tuple:
    """
    Rozpoznaje znak na zdjęciu.

    Returns:
        (predicted_char, confidence, all_probs)
    """
    # Wczytaj obraz
    image = Image.open(image_path).convert('RGB')

    # Przetwórz
    transform = get_transform()
    tensor = transform(image).unsqueeze(0).to(device)

    # Predykcja
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    predicted_char = CHARS[predicted.item()]
    confidence_val = confidence.item() * 100

    return predicted_char, confidence_val, probs[0]


def visualize_prediction(image_path: str, predicted_char: str, confidence: float):
    """Wizualizuje predykcję."""
    image = Image.open(image_path).convert('RGB')

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f"Rozpoznany znak: '{predicted_char}'\nPewność: {confidence:.1f}%",
              fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Zapisano wizualizację do: prediction_result.png")


def train_model(epochs: int = 10, batch_size: int = 32):
    """Trenuje model na datasecie Chars74K."""
    print("\n" + "=" * 60)
    print("TRENOWANIE MODELU")
    print("=" * 60)

    # Pobierz dataset jeśli brak
    download_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    # Transformacje
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    # Dataset
    dataset = datasets.ImageFolder(root=EXTRACTED_DIR, transform=train_transform)

    # Podział na train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Zbiór treningowy: {len(train_dataset)} obrazów")
    print(f"Zbiór walidacyjny: {len(val_dataset)} obrazów")
    print(f"Liczba klas: {len(dataset.classes)}")

    # Model
    model = SimpleCNN(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Trenowanie
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Walidacja
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Zapisano najlepszy model (acc: {best_acc:.2f}%)")

    print(f"\nTrenowanie zakończone! Najlepsza dokładność: {best_acc:.2f}%")
    print(f"Model zapisany do: {MODEL_PATH}")

# ============================================================================
# Rozpoznanie wyrazu
# ============================================================================
def predict_word(image_path, model, device):
    
    # Wczytanie obrazu i konwersja do grayscale
    image = Image.open(image_path).convert("L")  # grayscale
    img_array = np.array(image)
    
    # Binaryzacja
    binary = img_array < 128  # zakładamy, że ciemne litery <128
    
    # Projekcja pionowa do segmentacji liter
    vertical_sum = np.sum(binary, axis=0)
    
    # Wykrycie granic liter
    letters_bounds = []
    in_letter = False
    for i, val in enumerate(vertical_sum):
        if val > 0 and not in_letter:
            start = i
            in_letter = True
        elif val == 0 and in_letter:
            end = i
            letters_bounds.append((start, end))
            in_letter = False
    # jeśli ostatnia litera sięga końca obrazu
    if in_letter:
        letters_bounds.append((start, len(vertical_sum)))
    
    word = ""
    
    # Rozpoznanie każdej litery
    model.eval()  # tryb ewaluacji
    with torch.no_grad():
        for (start, end) in letters_bounds:
            letter_img = img_array[:, start:end]
            
            # usuń marginesy w pionie
            rows = np.where(np.sum(letter_img < 128, axis=1) > 0)[0]
            if len(rows) == 0:
                continue
            top, bottom = rows[0], rows[-1]
            letter_img = letter_img[top:bottom, :]
            plt.imshow(letter_img, cmap="gray")
            plt.show()
            # konwersja do PIL
            letter_pil = Image.fromarray(letter_img)
            letter_pil = letter_pil.resize((48, 48))
            letter_pil = letter_pil.convert("RGB")
            
            # transformacja
            transform = get_transform()
            tensor = transform(letter_pil).unsqueeze(0).to(device)
            
            # predykcja
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            predicted_char = CHARS[predicted.item()]
            
            word += predicted_char


    return word

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="OCR - Rozpoznawanie znaków")
    parser.add_argument("--image", "-i", type=str, help="Ścieżka do zdjęcia do rozpoznania")
    parser.add_argument("--word", "-w", type=str, help="Ścieżka do zdjęcia do rozpoznania wyrazów")
    parser.add_argument("--train", "-t", action="store_true", help="Trenuj model")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Liczba epok (domyślnie 10)")
    parser.add_argument("--prepare", "-p", action="store_true", help="Tylko pobierz i przygotuj dane")
    parser.add_argument("--multi", "-m", type=str, nargs="+", help="Ścieżka do zdjęcia do rozpoznania z wieloma argumentami")
    args = parser.parse_args()

    # Sprawdź CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    if args.prepare:
        # Tylko przygotowanie danych
        download_dataset()
        print("\nDane przygotowane!")

    elif args.train:
        # Trenowanie modelu
        train_model(epochs=args.epochs)

    elif args.image:
        # Rozpoznawanie zdjęcia
        if not os.path.exists(args.image):
            print(f"Błąd: Nie znaleziono pliku {args.image}")
            sys.exit(1)

        print(f"\nRozpoznawanie zdjęcia: {args.image}")

        # Wczytaj model
        model = load_model(MODEL_PATH, device)

        # Predykcja
        predicted_char, confidence, probs = predict_image(args.image, model, device)

        print("\n" + "=" * 40)
        print(f"WYNIK: '{predicted_char}' (pewność: {confidence:.1f}%)")
        print("=" * 40)

        # Top 5 predykcji
        top5_probs, top5_indices = torch.topk(probs, 5)
        print("\nTop 5 predykcji:")
        for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
            char = CHARS[idx.item()]
            print(f"  {i+1}. '{char}' - {prob.item()*100:.1f}%")

        # Wizualizacja
        visualize_prediction(args.image, predicted_char, confidence)
        
    elif args.word:

       # Rozpoznawanie zdjęcia
        if not os.path.exists(args.word):
            print(f"Błąd: Nie znaleziono pliku {args.image}")
            sys.exit(1)

        print(f"\nRozpoznawanie zdjęcia: {args.image}")

        # Wczytaj model
        model = load_model(MODEL_PATH, device)
        word = predict_word(args.word, model, device);

        print(f"wyraz : '{word}'")

    elif args.multi:

        for n in args.multi:
            subprocess.run([sys.executable, sys.argv[0], "-i", str(n)])
        sys.exit()


        
    else:
        # Domyślnie: pokaż pomoc
        parser.print_help()
        print("\n" + "=" * 60)
        print("PRZYKŁADY UŻYCIA:")
        print("=" * 60)
        print("1. Pobierz dane:        python run_local.py --prepare")
        print("2. Trenuj model:        python run_local.py --train --epochs 15")
        print("3. Rozpoznaj zdjęcie:   python run_local.py --image moje_zdjecie.png")
        print("=" * 60)


if __name__ == "__main__":
    main()
