"""
Moduł wyświetlania wyników działania programu.

Odpowiedzialności:
  - wizualizacja predykcji (obraz + tytuł) z zapisem do pliku
  - wyświetlanie Top-5 predykcji w konsoli
  - formatowane komunikaty wyników
"""

import torch
import matplotlib.pyplot as plt

from .config import CHARS
from .utils import load_and_optionally_denoise


RESULT_IMAGE_PATH = "prediction_result.png"


# ── Wizualizacja ───────────────────────────────────────────────────────────────

def visualize_prediction(
    image_path: str,
    predicted_char: str,
    confidence: float,
    args,
    save_path: str = RESULT_IMAGE_PATH,
) -> None:
    """Wyświetla obraz z tytułem zawierającym wynik i zapisuje go do pliku PNG."""
    image = load_and_optionally_denoise(image_path, args, mode="L")

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(
        f"Rozpoznany znak: '{predicted_char}'\nPewność: {confidence:.1f}%",
        fontsize=16,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Zapisano wizualizację do: {save_path}")


# ── Wyniki w konsoli ───────────────────────────────────────────────────────────

def print_single_result(predicted_char: str, confidence: float) -> None:
    """Drukuje wynik predykcji pojedynczej litery."""
    print("\n" + "=" * 40)
    print(f"WYNIK: '{predicted_char}' (pewność: {confidence:.1f}%)")
    print("=" * 40)


def print_top5(probs: torch.Tensor) -> None:
    """Drukuje Top-5 predykcji z prawdopodobieństwami."""
    top5_probs, top5_indices = torch.topk(probs, 5)
    print("\nTop 5 predykcji:")
    for rank, (prob, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
        char = CHARS[idx.item()]
        print(f"  {rank}. '{char}' – {prob.item() * 100:.1f}%")


def print_word_result(word: str) -> None:
    """Drukuje rozpoznany wyraz."""
    print(f"\nRozpoznany wyraz: '{word}'")


def print_text_result(text: str) -> None:
    """Drukuje rozpoznany tekst wieloliniowy."""
    print(f"\nRozpoznany tekst:\n{text}")


def print_multi_result(image_path: str, predicted_char: str, confidence: float) -> None:
    """Drukuje wynik dla jednego obrazu w trybie wielu plików."""
    print(f"  {image_path}  →  '{predicted_char}' ({confidence:.1f}%)")
