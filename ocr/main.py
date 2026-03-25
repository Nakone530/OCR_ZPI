"""
Punkt wejścia programu OCR.

Łączy wszystkie moduły:
  ocr.config    – stałe konfiguracyjne
  ocr.model     – architektura sieci CNN
  ocr.trainer   – pobieranie danych i trening
  ocr.inference – rozpoznawanie znaków / wyrazów / tekstu
  ocr.display   – wyświetlanie i wizualizacja wyników
  ocr.utils     – narzędzia pomocnicze

"""

import argparse
import os
import sys

import torch

from ocr.config import MODEL_PATH
from ocr.display import visualize_prediction, print_single_result,print_text_result, print_multi_result
from ocr.inference import load_model, predict_image, predict_segments, predict_word
from ocr.output import OCRResult, create_output_handler
from ocr.trainer import download_dataset, train_model, infinite_train
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
                      help="Trenuj model (określona liczba epok)")
    mode.add_argument("--infinite", action="store_true",
                      help="Nieskończony trening do przerwania (Ctrl+C)")
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
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                        help="Rozmiar batcha (domyślnie: 32)")
    parser.add_argument("--checkpoint-interval", type=int, default=5,
                        help="Co ile epok zapisywać checkpoint w trybie infinite (domyślnie: 5)")
    parser.add_argument("--resume", "-r", type=str, metavar="PLIK",
                        help="Wznów trening z checkpointu")

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

    # Parametry wyjścia
    parser.add_argument("--output", "-o", type=str, metavar="PLIK",
                        help="Zapisz wynik do pliku (txt/json)")
    parser.add_argument("--output-format", "-f", type=str, default="console",
                        choices=["console", "txt", "json"],
                        help="Format wyjścia (domyślnie: console)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Cichy tryb - tylko zapis do pliku, bez wypisywania")

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

    # W trybie quiet, komunikaty informacyjne idą na stderr
    import sys
    def info(msg):
        if args.quiet:
            print(msg, file=sys.stderr)
        else:
            print(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Używane urządzenie: {device}")

    # ── Przygotowanie danych ──
    if args.prepare:
        download_dataset()
        print("\nDane przygotowane!")

    # ── Trening ──
    elif args.train:
        train_model(epochs=args.epochs, batch_size=args.batch_size, model_path=args.resume)

    # ── Nieskończony trening ──
    elif args.infinite:
        info("\nUruchamianie nieskończonego treningu...")
        info("Naciśnij Ctrl+C aby wstrzymać i wyświetlić menu opcji.\n")
        infinite_train(
            batch_size=args.batch_size,
            model_path=args.resume,
            checkpoint_interval=args.checkpoint_interval
        )

    # ── Pojedyncza litera ──
    elif args.image:
        _require_file(args.image)
        info(f"\nRozpoznawanie: {args.image}")
        save_image_to_today_folder(args.image)

        model = load_model(MODEL_PATH, device)
        predicted_char, confidence, probs = predict_image(args.image, model, device, args)

        output_handler = create_output_handler(args, source_image=args.image)
        result = OCRResult(predicted_char, confidence, probs, mode="single")
        output_handler.output(result)

        if not args.quiet:
            visualize_prediction(args.image, predicted_char, confidence, args)

    # ── Wyraz ──
    elif args.word:
        _require_file(args.word)
        info(f"\nRozpoznawanie wyrazu: {args.word}")
        save_image_to_today_folder(args.word)

        model = load_model(MODEL_PATH, device)
        word, avg_word_confidence, class_confidence = predict_word(args.word, model, device, args)
        print_word_result(word, avg_word_confidence, class_confidence)

    # ── Tekst wieloliniowy ──
    elif args.lines:
        _require_file(args.lines)
        info(f"\nRozpoznawanie tekstu: {args.lines}")
        save_image_to_today_folder(args.lines)

        model = load_model(MODEL_PATH, device)
        text, words_with_confidence, class_confidence = predict_segments(args.lines, model, device, args)
        print_text_result(text, words_with_confidence, class_confidence)

    # ── Wiele zdjęć ──
    elif args.multi:
        model = load_model(MODEL_PATH, device)
        info(f"\nRozpoznawanie {len(args.multi)} pliku(-ów):")

        results = []

        for img_path in args.multi:
            if not os.path.exists(img_path):
                info(f"  Pominięto (nie znaleziono): {img_path}")
                continue
            save_image_to_today_folder(img_path)
            predicted_char, confidence, probs = predict_image(img_path, model, device, args)
            results.append({
                "file": img_path,
                "char": predicted_char,
                "confidence": confidence,
                "probs": probs
            })

            if args.output_format == "console":
                info(f"  {img_path}  →  '{predicted_char}' ({confidence:.1f}%)")

        # Dla JSON/TXT zapisz wszystkie wyniki
        if args.output_format in ("json", "txt") and args.output:
            import json
            from pathlib import Path

            if args.output_format == "json":
                data = {
                    "results": [
                        {
                            "file": r["file"],
                            "char": r["char"],
                            "confidence": round(r["confidence"], 2)
                        }
                        for r in results
                    ]
                }
                Path(args.output).write_text(
                    json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
            else:  # txt
                lines = [f"{r['file']}: {r['char']} ({r['confidence']:.1f}%)" for r in results]
                Path(args.output).write_text("\n".join(lines), encoding="utf-8")

            info(f"Zapisano wyniki do: {args.output}")

    # ── Brak argumentów → pomoc ──
    else:
        parser.print_help()
        print("\n" + "=" * 60)
        print("PRZYKŁADY UŻYCIA:")
        print("=" * 60)
        print("  python main.py --prepare")
        print("  python main.py --train --epochs 15")
        print("  python main.py --train --epochs 20 --batch-size 64")
        print("  python main.py --train --resume checkpoint.pth")
        print("")
        print("NIESKOŃCZONY TRENING:")
        print("  python main.py --infinite")
        print("  python main.py --infinite --batch-size 64")
        print("  python main.py --infinite --checkpoint-interval 10")
        print("  python main.py --infinite --resume checkpoint.pth")
        print("")
        print("ROZPOZNAWANIE:")
        print("  python main.py --image litera.png")
        print("  python main.py --image litera.png --denoise --denoise-method nlm-color")
        print("  python main.py --word wyraz.png")
        print("  python main.py --lines tekst.png")
        print("  python main.py --multi a.png b.png c.png")
        print("")
        print("OPCJE WYJŚCIA (wyniki zapisywane w folderze 'wynik/'):")
        print("  python main.py --image litera.png -f txt        # wynik/ + input.png + wynik.txt")
        print("  python main.py --image litera.png -f json       # wynik/ + input.png + wynik.json")
        print("  python main.py --word wyraz.png -o custom.json  # zapis do custom.json")
        print("  python main.py --image litera.png -f json -q    # cichy tryb")
        print("=" * 60)


if __name__ == "__main__":
    main()
