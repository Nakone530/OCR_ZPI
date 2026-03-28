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
  python main.py --image litera.png --denoise
  python main.py --word wyraz.png
  python main.py --lines tekst.png
  python main.py --multi a.png b.png c.png
"""

import argparse
import os
import sys

import torch

from ocr.config import MODEL_PATH
from ocr.display import print_text_result, print_word_result, visualize_prediction
from ocr.inference import get_active_chars, load_model, predict_image, predict_segments, predict_word
from ocr.output import OCRResult, create_output_handler
from ocr.trainer import download_dataset, train_model, infinite_train
from ocr.utils import save_image_to_today_folder
from ocr.json_output import (
    build_image_result_json,
    build_lines_result_json,
    build_multi_result_json,
    build_word_result_json,
    dump_json,
    write_json,
)


# ── Parsowanie argumentów ──────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """
    Buduje parser argumentów CLI dla programu OCR.
    
    Zwraca:
        argparse.ArgumentParser: Skonfigurowany parser argumentów.
    """
    parser = argparse.ArgumentParser(
        description="OCR – Rozpoznawanie znaków z opcjonalnym odszumianiem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady:
  %(prog)s --prepare                            # Pobierz dane treningowe
  %(prog)s --train --epochs 20                  # Trenuj przez 20 epok
  %(prog)s --infinite                           # Nieskończony trening (Ctrl+C = menu)
  %(prog)s --image litera.png                   # Rozpoznaj pojedynczą literę
  %(prog)s --word wyraz.png --denoise           # Rozpoznaj wyraz z odszumianiem
  %(prog)s --lines tekst.png --json --pretty    # Rozpoznaj tekst, ładny JSON
  %(prog)s --multi a.png b.png -o wynik.json    # Wiele obrazów, zapis do pliku
"""
    )

    # ── Tryby działania (wzajemnie wykluczające się) ──
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prepare", "-p",
        action="store_true",
        help="Pobierz i przygotuj dane treningowe (Chars74K)"
    )
    mode.add_argument(
        "--train", "-t",
        action="store_true",
        help="Trenuj model przez określoną liczbę epok"
    )
    mode.add_argument(
        "--infinite",
        action="store_true",
        help="Nieskończony trening do przerwania (Ctrl+C wyświetla menu)"
    )
    mode.add_argument(
        "--image", "-i",
        type=str,
        metavar="PLIK",
        help="Rozpoznaj pojedynczą literę z obrazu"
    )
    mode.add_argument(
        "--word", "-w",
        type=str,
        metavar="PLIK",
        help="Rozpoznaj wyraz (jedna linia tekstu)"
    )
    mode.add_argument(
        "--lines", "-l",
        type=str,
        metavar="PLIK",
        help="Rozpoznaj tekst wieloliniowy"
    )
    mode.add_argument(
        "--multi", "-m",
        type=str,
        nargs="+",
        metavar="PLIK",
        help="Rozpoznaj wiele obrazów pojedynczych liter"
    )

    # ── Parametry trenowania ──
    train_group = parser.add_argument_group("Trening")
    train_group.add_argument(
        "--epochs", "-e",
        type=int,
        default=10,
        metavar="N",
        help="Liczba epok treningu (domyślnie: 10)"
    )
    train_group.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        metavar="N",
        help="Rozmiar batcha (domyślnie: 32)"
    )
    train_group.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        metavar="N",
        help="Co ile epok zapisywać checkpoint w trybie infinite (domyślnie: 5)"
    )
    train_group.add_argument(
        "--resume", "-r",
        type=str,
        metavar="PLIK",
        help="Wznów trening z pliku checkpointu (.pth)"
    )

    # ── Preprocessing obrazu ──
    preproc_group = parser.add_argument_group("Preprocessing")
    preproc_group.add_argument(
        "--denoise",
        action="store_true",
        help="Włącz odszumianie obrazu (filtr bilateralny - zachowuje krawędzie)"
    )

    # ── Wyjście wyników ──
    output_group = parser.add_argument_group("Wyjście")
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Wypisz wynik jako JSON (zamiast formatu tekstowego)"
    )
    output_group.add_argument(
        "--pretty",
        action="store_true",
        help="Formatuj JSON z wcięciami (czytelniejszy)"
    )
    output_group.add_argument(
        "--output", "-o",
        type=str,
        metavar="PLIK",
        help="Zapisz wynik do pliku (format zależny od rozszerzenia: .json/.txt)"
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Tryb cichy - tylko zapis do pliku, bez wypisywania na ekran"
    )

    return parser


# ── Pomocnik walidacji pliku ───────────────────────────────────────────────────

def _require_file(path: str) -> None:
    """Sprawdza czy plik istnieje, kończy program jeśli nie."""
    if not os.path.exists(path):
        print(f"Błąd: Nie znaleziono pliku '{path}'", file=sys.stderr)
        sys.exit(1)


def _info(msg: str, quiet: bool = False) -> None:
    """Wypisuje komunikat informacyjny (na stderr w trybie quiet)."""
    if quiet:
        print(msg, file=sys.stderr)
    else:
        print(msg)


def _get_output_path(args, saved_copy_path: str, default_name: str = "results") -> str:
    """Określa ścieżkę wyjściową dla pliku JSON."""
    if args.output:
        return args.output
    if saved_copy_path:
        return os.path.splitext(saved_copy_path)[0] + ".json"
    return f"{default_name}.json"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Główna funkcja programu OCR."""
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _info(f"Używane urządzenie: {device}", args.quiet)

    # ── Przygotowanie danych ──
    if args.prepare:
        download_dataset()
        print("\nDane przygotowane!")

    # ── Trening (określona liczba epok) ──
    elif args.train:
        train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_path=args.resume
        )

    # ── Nieskończony trening ──
    elif args.infinite:
        _info("\nUruchamianie nieskończonego treningu...", args.quiet)
        _info("Naciśnij Ctrl+C aby wstrzymać i wyświetlić menu opcji.\n", args.quiet)
        infinite_train(
            batch_size=args.batch_size,
            model_path=args.resume,
            checkpoint_interval=args.checkpoint_interval
        )

    # ── Pojedyncza litera ──
    elif args.image:
        _require_file(args.image)
        _info(f"\nRozpoznawanie: {args.image}", args.quiet)
        saved_copy_path = save_image_to_today_folder(args.image)

        model = load_model(MODEL_PATH, device)
        predicted_char, confidence, probs = predict_image(args.image, model, device, args)
        active_labels = get_active_chars()

        if args.json:
            payload = build_image_result_json(
                image_path=args.image,
                saved_copy_path=saved_copy_path,
                predicted_char=predicted_char,
                confidence=confidence,
                probs=probs,
                device=str(device),
            )
            if not args.quiet:
                print(dump_json(payload, pretty=args.pretty))
            out_path = _get_output_path(args, saved_copy_path)
            write_json(out_path, payload, pretty=args.pretty)
            _info(f"Zapisano JSON: {out_path}", args.quiet)
        else:
            output_handler = create_output_handler(args, source_image=args.image)
            result = OCRResult(predicted_char, confidence, probs, mode="single", class_labels=active_labels)
            output_handler.output(result)
            if not args.quiet:
                visualize_prediction(args.image, predicted_char, confidence, args)

    # ── Wyraz (jedna linia) ──
    elif args.word:
        _require_file(args.word)
        _info(f"\nRozpoznawanie wyrazu: {args.word}", args.quiet)
        saved_copy_path = save_image_to_today_folder(args.word)

        model = load_model(MODEL_PATH, device)
        word, avg_word_confidence, class_confidence = predict_word(args.word, model, device, args)

        if args.json:
            payload = build_word_result_json(
                image_path=args.word,
                saved_copy_path=saved_copy_path,
                word=word,
                device=str(device),
            )
            if not args.quiet:
                print(dump_json(payload, pretty=args.pretty))
            out_path = _get_output_path(args, saved_copy_path)
            write_json(out_path, payload, pretty=args.pretty)
            _info(f"Zapisano JSON: {out_path}", args.quiet)
        else:
            print_word_result(word, avg_word_confidence, class_confidence)

    # ── Tekst wieloliniowy ──
    elif args.lines:
        _require_file(args.lines)
        _info(f"\nRozpoznawanie tekstu: {args.lines}", args.quiet)
        saved_copy_path = save_image_to_today_folder(args.lines)

        model = load_model(MODEL_PATH, device)
        text, words_with_confidence, class_confidence = predict_segments(args.lines, model, device, args)

        if args.json:
            payload = build_lines_result_json(
                image_path=args.lines,
                saved_copy_path=saved_copy_path,
                text=text,
                device=str(device),
            )
            if not args.quiet:
                print(dump_json(payload, pretty=args.pretty))
            out_path = _get_output_path(args, saved_copy_path)
            write_json(out_path, payload, pretty=args.pretty)
            _info(f"Zapisano JSON: {out_path}", args.quiet)
        else:
            print_text_result(text, words_with_confidence, class_confidence)

    # ── Wiele obrazów ──
    elif args.multi:
        model = load_model(MODEL_PATH, device)
        _info(f"\nRozpoznawanie {len(args.multi)} obrazu(-ów):", args.quiet)

        results = []
        for img_path in args.multi:
            if not os.path.exists(img_path):
                _info(f"  Pominięto (nie znaleziono): {img_path}", args.quiet)
                continue

            save_image_to_today_folder(img_path)
            predicted_char, confidence, probs = predict_image(img_path, model, device, args)
            results.append({
                "file": img_path,
                "char": predicted_char,
                "confidence": confidence,
                "probs": probs
            })

            if not args.json and not args.quiet:
                print(f"  {img_path}  →  '{predicted_char}' ({confidence:.1f}%)")

        if args.json:
            payload = build_multi_result_json(results=results, device=str(device))
            if not args.quiet:
                print(dump_json(payload, pretty=args.pretty))
            out_path = args.output if args.output else "results.json"
            write_json(out_path, payload, pretty=args.pretty)
            _info(f"Zapisano JSON: {out_path}", args.quiet)

        elif args.output:
            # Zapis do pliku tekstowego
            from pathlib import Path
            lines = [f"{r['file']}: {r['char']} ({r['confidence']:.1f}%)" for r in results]
            Path(args.output).write_text("\n".join(lines), encoding="utf-8")
            _info(f"Zapisano wyniki: {args.output}", args.quiet)

    # ── Brak argumentów → pomoc ──
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
