"""
Przygotowanie danych do trenowania sieci CNN dla zadania OCR
Dataset: Chars74K (EnglishFnt)
Środowisko: Google Colab
"""

import os
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from PIL import Image

# ============================================================================
# KONFIGURACJA
# ============================================================================

DATA_URL = "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz"
DATA_DIR = "./data"
ARCHIVE_NAME = "EnglishFnt.tgz"
ARCHIVE_PATH = os.path.join(DATA_DIR, ARCHIVE_NAME)
EXTRACTED_DIR = os.path.join(DATA_DIR, "English", "Fnt")

IMAGE_SIZE = 48
MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
STD = [0.229, 0.224, 0.225]   # ImageNet std

# ============================================================================
# FUNKCJE POMOCNICZE
# ============================================================================

def download_dataset(url: str, save_path: str) -> None:
    """Pobiera dataset z podanego URL, jeśli nie istnieje lokalnie."""
    if os.path.exists(save_path):
        print(f"Plik {save_path} już istnieje. Pomijam pobieranie.")
        return

    print(f"Pobieranie datasetu z {url}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Pobieranie z paskiem postępu
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        print(f"\rPostęp: {percent:.1f}%", end="")

    urllib.request.urlretrieve(url, save_path, show_progress)
    print("\nPobieranie zakończone!")


def extract_archive(archive_path: str, extract_to: str) -> None:
    """Rozpakowuje archiwum .tgz do wskazanego katalogu."""
    # Sprawdź czy już rozpakowano
    if os.path.exists(os.path.join(extract_to, "English", "Fnt")):
        print("Archiwum już rozpakowane. Pomijam ekstrakcję.")
        return

    print(f"Rozpakowywanie {archive_path}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print("Rozpakowywanie zakończone!")


def get_transforms() -> dict:
    """Zwraca transformacje dla zbiorów treningowego i walidacyjnego."""

    # Transformacje dla zbioru treningowego (z data augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    # Transformacje dla zbioru walidacyjnego (bez augmentacji)
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    return {
        "train": train_transform,
        "val": val_transform
    }


def show_sample_image(dataset_path: str) -> Image.Image:
    """Wyświetla przykładowy obraz przed przetwarzaniem."""
    # Znajdź pierwszy dostępny obraz
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(root, file)
                image = Image.open(image_path).convert('RGB')

                # Wyświetl obraz
                plt.figure(figsize=(6, 6))
                plt.imshow(image)
                plt.title(f"Przykładowy obraz przed przetwarzaniem\n"
                         f"Rozmiar: {image.size}\nŚcieżka: {os.path.basename(root)}/{file}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()

                return image

    raise FileNotFoundError("Nie znaleziono żadnych obrazów w datasecie!")


def process_image_to_tensor(image: Image.Image, transform: transforms.Compose) -> torch.Tensor:
    """Przetwarza obraz PIL do tensora przy użyciu podanych transformacji."""
    tensor = transform(image)
    return tensor


def add_batch_dimension(tensor: torch.Tensor) -> torch.Tensor:
    """Dodaje wymiar batcha do tensora (unsqueeze na pozycji 0)."""
    return tensor.unsqueeze(0)


def visualize_tensor(tensor: torch.Tensor, title: str = "Przetworzony obraz") -> None:
    """Wizualizuje tensor jako obraz (po denormalizacji)."""
    # Usuń wymiar batcha jeśli istnieje
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Denormalizacja
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    tensor_denorm = tensor * std + mean
    tensor_denorm = torch.clamp(tensor_denorm, 0, 1)

    # Konwersja do formatu HWC dla matplotlib
    image_np = tensor_denorm.permute(1, 2, 0).numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(image_np)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ============================================================================
# GŁÓWNY KOD
# ============================================================================

def main():
    print("=" * 60)
    print("PRZYGOTOWANIE DANYCH DLA OCR - CHARS74K")
    print("=" * 60)

    # 1. Pobierz dataset
    print("\n[1/7] Pobieranie datasetu...")
    download_dataset(DATA_URL, ARCHIVE_PATH)

    # 2. Rozpakuj archiwum
    print("\n[2/7] Rozpakowywanie archiwum...")
    extract_archive(ARCHIVE_PATH, DATA_DIR)

    # 3. Wyświetl przykładowy obraz przed przetwarzaniem
    print("\n[3/7] Wyświetlanie przykładowego obrazu przed przetwarzaniem...")
    sample_image = show_sample_image(EXTRACTED_DIR)
    print(f"Rozmiar oryginalnego obrazu: {sample_image.size}")

    # 4. Przygotuj transformacje
    print("\n[4/7] Przygotowywanie transformacji...")
    transforms_dict = get_transforms()
    print("Transformacje treningowe:")
    print(transforms_dict["train"])

    # 5. Wczytaj dataset przy użyciu ImageFolder
    print("\n[5/7] Wczytywanie datasetu przy użyciu ImageFolder...")
    train_dataset = datasets.ImageFolder(
        root=EXTRACTED_DIR,
        transform=transforms_dict["train"]
    )

    val_dataset = datasets.ImageFolder(
        root=EXTRACTED_DIR,
        transform=transforms_dict["val"]
    )

    print(f"Liczba obrazów w datasecie: {len(train_dataset)}")
    print(f"Liczba klas: {len(train_dataset.classes)}")
    print(f"Przykładowe klasy: {train_dataset.classes[:10]}...")

    # 6. Przetwórz przykładowy obraz do tensora
    print("\n[6/7] Przetwarzanie obrazu do tensora...")
    sample_tensor = process_image_to_tensor(sample_image, transforms_dict["train"])
    print(f"Kształt tensora po przetworzeniu: {sample_tensor.shape}")
    print(f"Typ danych tensora: {sample_tensor.dtype}")
    print(f"Zakres wartości: [{sample_tensor.min():.4f}, {sample_tensor.max():.4f}]")

    # 7. Dodaj wymiar batcha
    print("\n[7/7] Dodawanie wymiaru batcha...")
    batch_tensor = add_batch_dimension(sample_tensor)
    print(f"Kształt końcowego tensora: {batch_tensor.shape}")
    print(f"  - Batch size: {batch_tensor.shape[0]}")
    print(f"  - Kanały (RGB): {batch_tensor.shape[1]}")
    print(f"  - Wysokość: {batch_tensor.shape[2]}")
    print(f"  - Szerokość: {batch_tensor.shape[3]}")

    # Wizualizacja przetworzonego obrazu
    print("\nWizualizacja przetworzonego obrazu...")
    visualize_tensor(batch_tensor, f"Obraz po przetworzeniu ({IMAGE_SIZE}x{IMAGE_SIZE})")

    # Podsumowanie
    print("\n" + "=" * 60)
    print("PODSUMOWANIE")
    print("=" * 60)
    print(f"Dataset: Chars74K (EnglishFnt)")
    print(f"Liczba obrazów: {len(train_dataset)}")
    print(f"Liczba klas: {len(train_dataset.classes)}")
    print(f"Rozmiar obrazu: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Kształt tensora wejściowego: {batch_tensor.shape}")
    print(f"Format: (batch_size, channels, height, width)")
    print("=" * 60)

    # Przykład użycia DataLoader
    print("\nPrzykład tworzenia DataLoader:")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    # Pobranie przykładowego batcha
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"Kształt batcha: {sample_batch.shape}")
    print(f"Kształt etykiet: {sample_labels.shape}")

    return train_dataset, val_dataset, batch_tensor


if __name__ == "__main__":
    train_dataset, val_dataset, final_tensor = main()
