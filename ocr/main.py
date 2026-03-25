"""
Punkt wejścia programu OCR.

Łączy wszystkie moduły:
  ocr.config    – stałe konfiguracyjne
  ocr.model     – architektura sieci CNN
  ocr.trainer   – pobieranie danych i trening
  ocr.inference – rozpoznawanie znaków / wyrazów / tekstu
  ocr.display   – wyświetlanie i wizualizacja wyników
  ocr.utils     – narzędzia pomocnicze

Przykłady użycia:
  python main.py --prepare
  python main.py --train --epochs 15
  python main.py --image litera.png
  python main.py --image litera.png --denoise --denoise-method nlm-color
  python main.py --word wyraz.png
  python main.py --lines tekst.png
  python main.py --multi a.png b.png c.png
"""

import argparse
import os
import sys

import torch

from ocr.json_output import (
    build_image_result_json,
    build_lines_result_json,
    build_multi_result_json,
    build_word_result_json,
    dump_json,
    write_json,
)
from ocr.config import MODEL_PATH
from ocr.display import (
    print_multi_result,
    print_single_result,
    print_text_result,
    print_top5,
    print_word_result,
    visualize_prediction,
)
from ocr.inference import load_model, predict_image, predict_segments, predict_word
from ocr.trainer import download_dataset, train_model
from ocr.utils import save_image_to_today_folder


# ── Parsowanie argumentów ──────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OCR – Rozpoznawanie znaków (z opcjonalnym odszumianiem)"
    )

    # Tryby działania
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", "-p", action="store_true",
                      help="Pobierz i przygotuj dane")
    mode.add_argument("--train", "-t", action="store_true",
                      help="Trenuj model")
    mode.add_argument("--image", "-i", type=str, metavar="PLIK",
                      help="Rozpoznaj pojedynczą literę")
    mode.add_argument("--word", "-w", type=str, metavar="PLIK",
                      help="Rozpoznaj wyraz (jedna linia)")
    mode.add_argument("--lines", "-l", type=str, metavar="PLIK",
                      help="Rozpoznaj tekst wieloliniowy")
    mode.add_argument("--multi", "-m", type=str, nargs="+", metavar="PLIK",
                      help="Rozpoznaj wiele zdjęć pojedynczych liter")

    # Parametry trenowania
    parser.add_argument("--epochs", "-e", type=int, default=10,
                        help="Liczba epok (domyślnie: 10)")

    # Parametry odszumiania
    parser.add_argument("--denoise", action="store_true",
                        help="Włącz odszumianie")
    parser.add_argument("--denoise-method", default="nlm-color",
                        choices=["nlm-color", "median", "bilateral", "gaussian"],
                        help="Metoda odszumiania (domyślnie: nlm-color)")
    parser.add_argument("--h", type=int, default=10,
                        help="Siła NLM – luminancja")
    parser.add_argument("--hColor", type=int, default=10,
                        help="Siła NLM – kolor")
    parser.add_argument("--ksize", type=int, default=3,
                        help="Rozmiar jądra dla median/gaussian (3, 5, 7…)")

    # Wyjście
    parser.add_argument(
        "--json",
        action="store_true",
        help="Wypisz wynik jako JSON (zamiast formatu tekstowego)",
    )
    parser.add_argument(
        "--json-pretty",
        action="store_true",
        help="Sformatuj JSON (wcięcia, czytelniej dla człowieka)",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        metavar="PLIK",
        help="Zapisz wynik JSON do pliku (domyślnie: obok skopiowanego obrazu w folderze z datą)",
    )

    return parser


# ── Pomocnik walidacji pliku ───────────────────────────────────────────────────

def _require_file(path: str) -> None:
    if not os.path.exists(path):
        print(f"Błąd: Nie znaleziono pliku {path}")
        sys.exit(1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    # ── Przygotowanie danych ──
    if args.prepare:
        download_dataset()
        print("\nDane przygotowane!")

    # ── Trening ──
    elif args.train:
        train_model(epochs=args.epochs)

    # ── Pojedyncza litera ──
    elif args.image:
        _require_file(args.image)
        print(f"\nRozpoznawanie: {args.image}")
        saved_copy_path = save_image_to_today_folder(args.image)

        model = load_model(MODEL_PATH, device)
        predicted_char, confidence, probs = predict_image(args.image, model, device, args)

        if args.json:
            payload = build_image_result_json(
                image_path=args.image,
                saved_copy_path=saved_copy_path,
                predicted_char=predicted_char,
                confidence=confidence,
                probs=probs,
                device=str(device),
            )
            print(dump_json(payload, pretty=args.json_pretty))
            out_path = args.json_path
            if out_path is None:
                out_path = os.path.splitext(saved_copy_path)[0] + ".json"
            write_json(out_path, payload, pretty=args.json_pretty)
        else:
            print_single_result(predicted_char, confidence)
            print_top5(probs)
            visualize_prediction(args.image, predicted_char, confidence, args)

    # ── Wyraz ──
    elif args.word:
        _require_file(args.word)
        print(f"\nRozpoznawanie wyrazu: {args.word}")
        saved_copy_path = save_image_to_today_folder(args.word)

        model = load_model(MODEL_PATH, device)
        word = predict_word(args.word, model, device, args)
        if args.json:
            payload = build_word_result_json(
                image_path=args.word,
                saved_copy_path=saved_copy_path,
                word=word,
                device=str(device),
            )
            print(dump_json(payload, pretty=args.json_pretty))
            out_path = args.json_path
            if out_path is None:
                out_path = os.path.splitext(saved_copy_path)[0] + ".json"
            write_json(out_path, payload, pretty=args.json_pretty)
        else:
            print_word_result(word)

    # ── Tekst wieloliniowy ──
    elif args.lines:
        _require_file(args.lines)
        print(f"\nRozpoznawanie tekstu: {args.lines}")
        saved_copy_path = save_image_to_today_folder(args.lines)

        model = load_model(MODEL_PATH, device)
        text = predict_segments(args.lines, model, device, args)
        if args.json:
            payload = build_lines_result_json(
                image_path=args.lines,
                saved_copy_path=saved_copy_path,
                text=text,
                device=str(device),
            )
            print(dump_json(payload, pretty=args.json_pretty))
            out_path = args.json_path
            if out_path is None:
                out_path = os.path.splitext(saved_copy_path)[0] + ".json"
            write_json(out_path, payload, pretty=args.json_pretty)
        else:
            print_text_result(text)

    # ── Wiele zdjęć ──
    elif args.multi:
        model = load_model(MODEL_PATH, device)
        print(f"\nRozpoznawanie {len(args.multi)} pliku(-ów):")
        results = []
        for img_path in args.multi:
            if not os.path.exists(img_path):
                print(f"  Pominięto (nie znaleziono): {img_path}")
                continue
            saved_copy_path = save_image_to_today_folder(img_path)
            predicted_char, confidence, _ = predict_image(img_path, model, device, args)
            results.append(
                {
                    "image_path": img_path,
                    "saved_copy_path": saved_copy_path,
                    "predicted_char": predicted_char,
                    "confidence_percent": round(float(confidence), 4),
                }
            )
            if not args.json:
                print_multi_result(img_path, predicted_char, confidence)

        if args.json:
            payload = build_multi_result_json(
                results=results,
                device=str(device),
            )
            print(dump_json(payload, pretty=args.json_pretty))
            out_path = args.json_path
            if out_path is None:
                # multi: zapisz w bieżącym katalogu jako wynik.json
                out_path = "results.json"
            write_json(out_path, payload, pretty=args.json_pretty)

    # ── Brak argumentów → pomoc ──
    else:
        parser.print_help()
        print("\n" + "=" * 60)
        print("PRZYKŁADY UŻYCIA:")
        print("=" * 60)
        print("  python main.py --prepare")
        print("  python main.py --train --epochs 15")
        print("  python main.py --image litera.png")
        print("  python main.py --image litera.png --denoise --denoise-method nlm-color")
        print("  python main.py --word wyraz.png")
        print("  python main.py --lines tekst.png")
        print("  python main.py --multi a.png b.png c.png")
        print("=" * 60)


if __name__ == "__main__":
    main()
