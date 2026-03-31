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
    info=print,
) -> None:
    """
    Wizualizuje wynik predykcji - wyświetla obraz z tytułem i zapisuje do pliku.
    
    Funkcja tworzy wykres matplotlib z obrazem wejściowym i tytułem
    zawierającym rozpoznany znak oraz poziom pewności predykcji.
    
    Argumenty:
        image_path (str): Ścieżka do obrazu źródłowego do wyświetlenia.
        predicted_char (str): Rozpoznany znak (np. "A").
        confidence (float): Pewność predykcji w procentach (0-100).
        args: Obiekt argparse.Namespace z parametrami odszumiania.
        save_path (str, opcjonalnie): Ścieżka do zapisania wizualizacji.
                                   Domyślnie "prediction_result.png".
    
    Zwraca:
        None
    
    Efekty uboczne:
        - Wyświetla okno matplotlib z obrazem
        - Zapisuje wizualizację do pliku PNG
        - Wypisuje komunikat o zapisie na konsolę
    
    Przykład:
        >>> visualize_prediction("letter.png", "B", 95.5, args)
        Zapisano wizualizację do: prediction_result.png
    """
    image = load_and_optionally_denoise(image_path, args, mode="L")

    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    plt.title(
        f"Rozpoznany znak: '{predicted_char}'\nPewność: {confidence:.1f}%",
        fontsize=16,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    info(f"Zapisano wizualizację do: {save_path}")


# ── Wyniki w konsoli ───────────────────────────────────────────────────────────

def print_single_result(predicted_char: str, confidence: float, info=print) -> None:
    """
    Drukuje sformatowany wynik predykcji pojedynczej litery w konsoli.
    
    Wyświetla rozpoznany znak i poziom pewności w ramce wizualnej.
    
    Argumenty:
        predicted_char (str): Rozpoznany znak (np. "A").
        confidence (float): Pewność predykcji w procentach (0-100).
    
    Zwraca:
        None
    
    Przykład:
        >>> print_single_result("B", 95.5)
        
        ========================================
        WYNIK: 'B' (pewność: 95.5%)
        ========================================
    """
    info("\n" + "=" * 40)
    info(f"WYNIK: '{predicted_char}' (pewność: {confidence:.1f}%)")
    info("=" * 40)


def print_top5(probs: torch.Tensor, info=print) -> None:
    """
    Drukuje 5 najbardziej prawdopodobnych predykcji z prawdopodobieństwami.
    
    Funkcja sortuje prawdopodobieństwa i wyświetla 5 najwyższych
    wraz z odpowiadającymi im znakami.
    
    Argumenty:
        probs (torch.Tensor): Tensor prawdopodobieństw dla każdej klasy.
                              Oczekiwany kształt: (NUM_CLASSES,) lub (1, NUM_CLASSES).
    
    Zwraca:
        None
    
    Przykład:
        >>> print_top5(probs_tensor)
        
        Top 5 predykcji:
          1. 'A' – 85.3%
          2. 'H' – 8.2%
          3. 'R' – 3.1%
          4. 'N' – 2.0%
          5. 'M' – 1.4%
    """
    top5_probs, top5_indices = torch.topk(probs, 5)
    info("\nTop 5 predykcji:")
    for rank, (prob, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
        char = CHARS[idx.item()]
        info(f"  {rank}. '{char}' – {prob.item() * 100:.1f}%")


def print_word_result(
    word: str,
    avg_word_confidence: float,
    class_confidence: dict[str, float],
    info=print
) -> None:
    """
    Drukuje rozpoznany wyraz w konsoli.
    
    Argumenty:
        word (str): Rozpoznany wyraz (ciąg znaków).
    
    Zwraca:
        None
    
    Przykład:
        >>> print_word_result("HELLO")
        
        Rozpoznany wyraz: 'HELLO'
    """
    info(f"\nRozpoznany wyraz: '{word}'")
    info(f"Średnia pewność liter (słowo): {avg_word_confidence:.1f}%")

    if class_confidence:
        info("Średni poziom pewności na klasę:")
        for cls in sorted(class_confidence):
            info(f"  '{cls}': {class_confidence[cls]:.1f}%")


def print_text_result(
    text: str,
    words_with_confidence: list[tuple[str, float]],
    class_confidence: dict[str, float],
    info=print
) -> None:
    """
    Drukuje rozpoznany tekst wieloliniowy w konsoli.
    
    Argumenty:
        text (str): Rozpoznany tekst (może zawierać wiele linii).
    
    Zwraca:
        None
    
    Przykład:
        >>> print_text_result("HELLO\\nWORLD")
        
        Rozpoznany tekst:
        HELLO
        WORLD
    """

    info(f"\nRozpoznany tekst:\n{text}")

    if words_with_confidence:
        info("Średnia pewność liter na słowo:")
        for idx, (word, confidence) in enumerate(words_with_confidence, start=1):
            info(f"  {idx}. '{word}': {confidence:.1f}%")

    if class_confidence:
        info("Średni poziom pewności na klasę:")
        for cls in sorted(class_confidence):
            info(f"  '{cls}': {class_confidence[cls]:.1f}%")


def print_multi_result(image_path: str, predicted_char: str, confidence: float, info=print) -> None:
    """
    Drukuje wynik dla jednego obrazu w trybie przetwarzania wielu plików.
    
    Format wyjścia jest kompaktowy - jedna linia na obraz.
    
    Argumenty:
        image_path (str): Ścieżka do przetworzonego obrazu.
        predicted_char (str): Rozpoznany znak.
        confidence (float): Pewność predykcji w procentach (0-100).
    
    Zwraca:
        None
    
    Przykład:
        >>> print_multi_result("letter1.png", "A", 98.2)
          letter1.png  →  'A' (98.2%)
    """
    info(f"  {image_path}  →  '{predicted_char}' ({confidence:.1f}%)")
