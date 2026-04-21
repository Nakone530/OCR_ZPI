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
import logging
import os
import sys

import torch

from typing import Optional, List
from ocr.config import MODEL_PATH
from ocr.inference import get_active_chars, load_model, predict_image, predict_segments, predict_word
from ocr.output import OCRResult, create_output_handler
from ocr.trainer import download_dataset, train_model, infinite_train
from ocr.utils import save_image_to_today_folder
from ocr.json_output import (
    build_image_result_json,
    build_word_result_json,
    build_lines_result_json,
    build_multi_result_json,
    dump_json,
    write_json,
)
from ocr.display import (
    print_multi_result,
    print_single_result,
    print_text_result,
    print_top5,
    print_word_result,
    print_text_result,
    visualize_prediction,
)
#--State



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

    # Parametry segmentacji watershed
    parser.add_argument("--ws-fg-ratio", type=float, default=0.45,
                        help="Próg foreground dla watershed (ułamek max distance, domyślnie: 0.45)")
    parser.add_argument("--ws-split-aspect", type=float, default=1.15,
                        help="Kiedy komponent uznać za sklejony: warunek szerokość > ratio * wysokość (domyślnie: 1.15)")
    parser.add_argument("--ws-min-comp-area", type=int, default=30,
                        help="Minimalne pole komponentu, aby był kandydatem na literę (domyślnie: 30)")
    parser.add_argument("--ws-split-min-area", type=int, default=250,
                        help="Minimalne pole komponentu, od którego próbujemy podział watershed (domyślnie: 250)")
    parser.add_argument("--ws-min-box-w", type=int, default=3,
                        help="Minimalna szerokość boxa litery po segmentacji (domyślnie: 3)")
    parser.add_argument("--ws-min-box-h", type=int, default=5,
                        help="Minimalna wysokość boxa litery po segmentacji (domyślnie: 5)")
    parser.add_argument("--ws-min-box-area", type=int, default=20,
                        help="Minimalne pole boxa litery po segmentacji (domyślnie: 20)")
    parser.add_argument("--ws-merge-gap", type=int, default=4,
                        help="Maksymalna przerwa pozioma między fragmentami do scalenia (domyślnie: 4)")
    parser.add_argument("--ws-merge-height-ratio", type=float, default=1.8,
                        help="Maksymalny stosunek wysokości fragmentów do scalenia (domyślnie: 1.8)")
    parser.add_argument("--ws-merge-vert-dist", type=int, default=4,
                        help="Maksymalna odległość pionowa do scalenia fragmentów (domyślnie: 4)")

    # Parametr modelu
    parser.add_argument("--model-path", type=str, default=None, metavar="PLIK",
                        help="Ścieżka do wytrenowanego modelu (domyślnie: model_ocr.pth)")

    # Parametry wyjścia
    parser.add_argument("--output", "-o", type=str, metavar="PLIK",
                        help="Zapisz wynik do pliku (txt/json)")
    parser.add_argument("--output-format", "-f", type=str, default="console",
                        choices=["console", "txt", "json"],
                        help="Format wyjścia (domyślnie: console)")

    debug_mode = parser.add_mutually_exclusive_group()
    debug_mode.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help=(
            "Tryb debug dla -i/-w/-l: pokazuje podział liter i szczegóły "
            "klasyfikacji modelu"
        ),
    )
    parser.add_argument(
        "--debug-show-crops",
        action="store_true",
        help=(
            "W trybie --debug pokazuje wycinki liter (obrazy 28x28) "
            "podawane do modelu"
        ),
    )
    parser.add_argument(
        "--debug-save-crops",
        nargs="?",
        const="auto",
        default=None,
        metavar="KATALOG",
        help=(
            "W trybie --debug zapisuje wycinki liter podawane do modelu; "
            "bez wartości zapisuje do automatycznego katalogu"
        ),
    )
    # Zachowane wyłącznie dla kompatybilności wstecznej.
    debug_mode.add_argument("--quiet", "-q", action="store_true", help=argparse.SUPPRESS)
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

#-- state -> args

def build_args(state):
    args = []

    if state["mode"] == "train":
        args.append("--train")
        args += ["--epochs", str(state["epochs"])]
        args += ["--batch-size", str(state["batch_size"])]
        
    if state["denoise"]:
        args.append("--denoise")

    elif state["mode"] == "image":
        args += ["--image", state["input_path"]]

    if state["json"]:
        args.append("--json")

    return args

# ── Pomocnik walidacji pliku ───────────────────────────────────────────────────

def _require_file(path: str, info=None):
    if info is None:
        info = print
    
    if not os.path.exists(path):
        info(f"Błąd: Nie znaleziono pliku {path}")
        sys.exit(1)



#--args
def make_args() -> None:
    parser = build_parser()
    args = parser.parse_args()
    return args

# ── Main ───────────────────────────────────────────────────────────────────────

def make_info(buffer= None):
    if buffer is None:
        buffer = []

    def info(msg, quiet=False):
        buffer.append(msg)
        if not quiet:
            print(msg)
            
    return info, buffer

def main(args=None, info=None, buffor=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args is None:
        args = make_args()

    if info is None:
        info, buffor = make_info()

    if buffor is None:
        buffor = []
    
    # Ustal ścieżkę modelu: użyj args.model_path jeśli jest dostępne, inaczej MODEL_PATH
    model_path = getattr(args, 'model_path', None) or MODEL_PATH
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Używane urządzenie: {device}")
    info(f"Używany model: {model_path}")
    parser = build_parser()

    # ── Przygotowanie danych ──
    if args.prepare:
        download_dataset()
        print("\nDane przygotowane!")

    # ── Trening ──
    if args.train:
        train_model(epochs=args.epochs, batch_size=args.batch_size, model_path=args.resume, info=info)

    # ── Nieskończony trening ──
    elif args.infinite:
        info("\nUruchamianie nieskończonego treningu...")
        info("Naciśnij Ctrl+C aby wstrzymać i wyświetlić menu opcji.\n")
        infinite_train(
            batch_size=args.batch_size,
            model_path=args.resume,
            checkpoint_interval=args.checkpoint_interval,
            info=info
        )

    # ── Pojedyncza litera ──
    elif args.image:
        _require_file(args.image, info)
        info(f"\nRozpoznawanie: {args.image}")

        saved_copy_path = save_image_to_today_folder(args.image, info)

        model = load_model(model_path, device, info)

        result = predict_image(args.image, model, device, args)

        text = result["text"]
        confidence = result["confidence"]
        probs = result["probs"]

        active_labels = get_active_chars()

        output_handler = create_output_handler(args, source_image=args.image)

        ocr_result = OCRResult(
            text=text,
            confidence=confidence,
            probs=probs,
            mode="word",
            class_labels=active_labels
        )

        output_handler.output(ocr_result, info)

        if args.debug:
            visualize_prediction(args.image, text, confidence, args, info)

        if args.json:
            payload = build_image_result_json(
                image_path=args.image,
                saved_copy_path=saved_copy_path,
                predicted_char=text,
                confidence=confidence,
                probs=probs,
                device=str(device),
            )

            info(dump_json(payload, pretty=args.json_pretty))

            out_path = args.json_path or os.path.splitext(saved_copy_path)[0] + ".json"
            write_json(out_path, payload, pretty=args.json_pretty)

        else:
            print(f"TEXT: {text}")
            print(f"CONFIDENCE: {confidence:.2f}%")

    # ── Wyraz ──
    elif args.word:
        _require_file(args.word, info)

        info(f"\nRozpoznawanie wyrazu: {args.word}")
        saved_copy_path = save_image_to_today_folder(args.word, info)

        model = load_model(model_path, device, info)
        word, avg_word_confidence, class_confidence = predict_word(args.word, model, device, args)
        if args.json:
            payload = build_word_result_json(
                image_path=args.word,
                saved_copy_path=saved_copy_path,
                word=word,
                device=str(device),
            )
            info(dump_json(payload, pretty=args.json_pretty))
            out_path = args.json_path
            if out_path is None:
                out_path = os.path.splitext(saved_copy_path)[0] + ".json"
            write_json(out_path, payload, pretty=args.json_pretty)
        else:
            print_word_result(word, avg_word_confidence, class_confidence, info)


    # ── Tekst wieloliniowy ──
    elif args.lines:
        _require_file(args.lines, info)
        info(f"\nRozpoznawanie tekstu: {args.lines}")
        
        saved_copy_path = save_image_to_today_folder(args.lines, info)

        model = load_model(model_path, device, info)
        text, words_with_confidence, class_confidence = predict_segments(args.lines, model, device, args)


        if args.json:
            payload = build_lines_result_json(
                image_path=args.lines,
                saved_copy_path=saved_copy_path,
                text=text,
                device=str(device),
            )
            info(dump_json(payload, pretty=args.json_pretty))
            out_path = args.json_path
            if out_path is None:
                out_path = os.path.splitext(saved_copy_path)[0] + ".json"
            write_json(out_path, payload, pretty=args.json_pretty)
        else:
            print_text_result(text, words_with_confidence, class_confidence, info)



    # ── Wiele zdjęć ──
    elif args.multi:
        model = load_model(model_path, device, info)
        info(f"\nRozpoznawanie {len(args.multi)} pliku(-ów):")
        
        results = []
        
        for img_path in args.multi:
            if not os.path.exists(img_path):
                info(f"  Pominięto (nie znaleziono): {img_path}")
                continue

            saved_copy_path = save_image_to_today_folder(img_path, info)
            predicted_char, confidence, probs = predict_image(img_path, model, device, args)
            results.append({
                "file": img_path,
                "char": predicted_char,
                "confidence": confidence,
                "probs": probs
            })


            if not args.json:
                print_multi_result(img_path, predicted_char, confidence, info)

        if args.json:
            payload = build_multi_result_json(
                results=results,
                device=str(device),
            )
            info(dump_json(payload, pretty=args.json_pretty))
            out_path = args.json_path
            if out_path is None:
                # multi: zapisz w bieżącym katalogu jako wynik.json
                out_path = "results.json"
            write_json(out_path, payload, pretty=args.json_pretty)

        else:
            for result in results:
                info(f"  {result['file']}  →  '{result['char']}' ({result['confidence']:.1f}%)")

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
        info("\n" + "=" * 60)
        info("PRZYKŁADY UŻYCIA:")
        info("=" * 60)
        info("  python main.py --prepare")
        info("  python main.py --train --epochs 15")
        info("  python main.py --train --epochs 20 --batch-size 64")
        info("  python main.py --train --resume checkpoint.pth")
        info("")
        info("NIESKOŃCZONY TRENING:")
        info("  python main.py --infinite")
        info("  python main.py --infinite --batch-size 64")
        info("  python main.py --infinite --checkpoint-interval 10")
        info("  python main.py --infinite --resume checkpoint.pth")
        info("")
        info("ROZPOZNAWANIE:")
        info("  python main.py --image litera.png")
        info("  python main.py --image litera.png --denoise")
        info("  python main.py --word wyraz.png")
        info("  python main.py --lines tekst.png")
        info("  python main.py --multi a.png b.png c.png")
        info("=" * 60)
        
    all_text = "\n".join(buffor)
    return all_text
       
if __name__ == "__main__":
    main()
