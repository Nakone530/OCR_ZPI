"""
Punkt wejścia programu OCR.

Laczy wszystkie moduly:
  ocr.config    - stale konfiguracyjne
  ocr.model     - architektura sieci CNN
  ocr.trainer   - pobieranie danych i trening
  ocr.inference - rozpoznawanie znakow / wyrazow / tekstu
  ocr.display   - wyswietlanie i wizualizacja wynikow
  ocr.utils     - narzedzia pomocnicze
"""

import argparse
import json
import logging
import os
import sys

import torch

from ocr.config import MODEL_PATH
from ocr.inference import compute_accuracy, get_active_chars, load_model, predict_image, predict_letter, predict_segments, predict_word, process_folder
from ocr.output import OCRResult, create_output_handler
from ocr.trainer import train_model, infinite_train
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



# ── Pomocniki ─────────────────────────────────────────────────────────────────

def _print_accuracy(predicted_text: str, reference_path: str, info=None) -> None:
    if info is None:
        info = print
    try:
        reference = open(reference_path, encoding="utf-8").read()
    except OSError as e:
        info(f"Błąd odczytu pliku referencyjnego: {e}")
        return
    acc = compute_accuracy(predicted_text, reference)
    info("\n" + "=" * 60)
    info(f"  Dokładność OCR: {acc:.2f}%")
    info(f"  Predykcja:  {predicted_text.strip()[:80]}")
    info(f"  Referencja: {reference.strip()[:80]}")
    info("=" * 60)


# -- Funkcje pomocnicze dla wyświetlania rozmieszczenia tekstu ------------------

def display_text_layout(words, info):
    """Wyświetla tekst w oryginalnym rozmieszczeniu (z podziałem na linie)."""
    if not words:
        return
    
    # Filtruj słowa bez bbox
    words_with_bbox = [w for w in words if w.get("bbox") is not None]
    
    if not words_with_bbox:
        return
    
    # Grupuj po Y (podstawie linii)
    lines = []
    current_line = []
    current_y = None
    y_threshold = 20  # piksele tolerancji dla tej samej linii
    
    # Sortuj po Y
    sorted_words = sorted(
        words_with_bbox,
        key=lambda w: w["bbox"][1] if w.get("bbox") else 0
    )
    
    for word in sorted_words:
        bbox = word.get("bbox")
        if not bbox:
            continue
        
        y = bbox[1]
        if current_y is None or abs(y - current_y) <= y_threshold:
            current_line.append(word)
            if current_y is None:
                current_y = y
        else:
            if current_line:
                lines.append(current_line)
            current_line = [word]
            current_y = y
    
    if current_line:
        lines.append(current_line)
    
    # Wyświetl linie
    info("\n" + "=" * 60)
    info("TEKST W ROZMIESZCZENIU:")
    info("=" * 60)
    
    for line in lines:
        # Sortuj słowa w linii po X (od lewej do prawej)
        line_sorted = sorted(
            line,
            key=lambda w: w.get("bbox", [0, 0, 0, 0])[0] if w.get("bbox") else 0
        )
        
        line_text = " ".join(w["text"] for w in line_sorted)
        info(line_text)
    
    info("=" * 60)


# -- Parsowanie argumentow ------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OCR - Rozpoznawanie znakow (z opcjonalnym odszumianiem)"
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", "-p", action="store_true", help="Pobierz i przygotuj dane")
    mode.add_argument("--train", "-t", action="store_true", help="Trenuj model (okreslona liczba epok)")
    mode.add_argument("--infinite", action="store_true", help="Nieskonczony trening do przerwania (Ctrl+C)")
    mode.add_argument("--image", "-i", type=str, metavar="PLIK", help="Rozpoznaj pojedyncza litere")
    mode.add_argument("--word", "-w", type=str, metavar="PLIK", help="Rozpoznaj wyraz (jedna linia)")
    mode.add_argument("--lines", "-l", type=str, metavar="PLIK", help="Rozpoznaj tekst wieloliniowy")
    mode.add_argument("--multi", "-m", type=str, nargs="+", metavar="PLIK", help="Rozpoznaj wiele zdjec pojedynczych liter")
    mode.add_argument("--annotate", type=str, metavar="PLIK", help="Wycinki: popraw bboxy i zapisz wycinki + adnotacje")
    mode.add_argument("--folder", type=str, metavar="PLIK", help="Rozpoznaj zdjęcia w folderze")
    mode.add_argument("--page", type=str, metavar="PLIK", help="Separacja zdjęcia na wyrazy oraz ich rozpoznanie")
    # Porównanie z referencją
    parser.add_argument("--accuracy", "-a", type=str, default=None, metavar="PLIK",
                        help="Plik z referencyjną transkrypcją; oblicza procentowe podobieństwo wyniku OCR do referencji")

    parser.add_argument("--epochs", "-e", type=int, default=10, help="Liczba epok (domyslnie: 10)")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Rozmiar batcha (domyslnie: 32)")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Co ile epok zapisywac checkpoint w trybie infinite (domyslnie: 5)",
    )
    parser.add_argument("--resume", "-r", type=str, metavar="PLIK", help="Wznow trening z checkpointu")

    parser.add_argument("--denoise", action="store_true", help="Wlacz odszumianie")
    parser.add_argument(
        "--denoise-method",
        default="nlm-color",
        choices=["nlm-color", "median", "bilateral", "gaussian"],
        help="Metoda odszumiania (domyslnie: nlm-color)",
    )
    parser.add_argument("--h", type=int, default=10, help="Sila NLM - luminancja")
    parser.add_argument("--hColor", type=int, default=10, help="Sila NLM - kolor")
    parser.add_argument("--ksize", type=int, default=3, help="Rozmiar jadra dla median/gaussian (3,5,7...)")

    parser.add_argument("--ws-fg-ratio", type=float, default=0.45)
    parser.add_argument("--ws-split-aspect", type=float, default=1.15)
    parser.add_argument("--ws-min-comp-area", type=int, default=30)
    parser.add_argument("--ws-split-min-area", type=int, default=250)
    parser.add_argument("--ws-min-box-w", type=int, default=3)
    parser.add_argument("--ws-min-box-h", type=int, default=5)
    parser.add_argument("--ws-min-box-area", type=int, default=20)
    parser.add_argument("--ws-merge-gap", type=int, default=4)
    parser.add_argument("--ws-merge-height-ratio", type=float, default=1.8)
    parser.add_argument("--ws-merge-vert-dist", type=int, default=4)

    parser.add_argument("--model-path", type=str, default=None, metavar="PLIK", help="Sciezka do wytrenowanego modelu")
    parser.add_argument("--annotation-dir", type=str, default="inference", metavar="KATALOG", help="Katalog wyjsciowy dla trybu --annotate")
    parser.add_argument("--no-edit", action="store_true", help="W trybie --annotate wylacz interaktywna edycje bboxow")
    parser.add_argument("--non-interactive", action="store_true", help="W trybie --annotate pomin pytania input() i zapisz automatycznie")

    parser.add_argument("--output", "-o", type=str, metavar="PLIK", help="Zapisz wynik do pliku (txt/json)")
    parser.add_argument(
        "--output-format",
        "-f",
        type=str,
        default="console",
        choices=["console", "txt", "json"],
        help="Format wyjscia (domyslnie: console)",
    )

    debug_mode = parser.add_mutually_exclusive_group()
    debug_mode.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Tryb debug dla -i/-w/-l: pokazuje podzial liter i szczegoly klasyfikacji modelu",
    )
    parser.add_argument("--debug-show-crops", action="store_true")
    parser.add_argument("--debug-save-crops", nargs="?", const="auto", default=None, metavar="KATALOG")
    debug_mode.add_argument("--quiet", "-q", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--json", action="store_true", help="Wypisz wynik jako JSON")
    parser.add_argument("--json-pretty", action="store_true", help="Sformatuj JSON")
    parser.add_argument("--json-path", type=str, default=None, metavar="PLIK", help="Zapisz wynik JSON do pliku")

    return parser


def build_args(state):
    args = []
    if state["mode"] == "train":
        args.append("--train")
        args += ["--epochs", str(state["epochs"])]
        args += ["--batch-size", str(state["batch_size"])]
    elif state["mode"] == "image":
        args += ["--image", state["input_path"]]

    if state.get("denoise"):
        args.append("--denoise")
    if state["mode"] == "crnn":
        args += ["--crnn", state["input_path"]]

    if state["json"]:
        args.append("--json")
    return args


def _require_file(path: str, info=None):
    if info is None:
        info = print
    if not os.path.exists(path):
        info(f"Blad: Nie znaleziono pliku {path}")
        sys.exit(1)


def make_args():
    parser = build_parser()
    return parser.parse_args()


def make_info(buffer=None):
    if buffer is None:
        buffer = []

    def info(msg, quiet=False):
        buffer.append(msg)
        if not quiet:
            print(msg)

    return info, buffer


# -- Main ----------------------------------------------------------------------

def main(args=None, info=None, buffor=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args is None:
        args = make_args()
    if info is None:
        info, buffor = make_info()
    if buffor is None:
        buffor = []

    model_path = getattr(args, "model_path", None) or MODEL_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    info(f"Uzywane urzadzenie: {device}")
    info(f"Uzywany model: {model_path}")

    parser = build_parser()
    
    # ── Trening ──
    if args.train:
        train_model(epochs=args.epochs, batch_size=args.batch_size, model_path=args.resume, info=info, args=args)

    elif args.infinite:
        info("\nUruchamianie nieskonczonego treningu...")
        info("Nacisnij Ctrl+C aby wstrzymac i wyswietlic menu opcji.\n")
        infinite_train(
            batch_size=args.batch_size,
            model_path=args.resume,
            checkpoint_interval=args.checkpoint_interval,
            info=info,
        )

    elif args.annotate:
        _require_file(args.annotate, info)
        info(f"\nUruchamianie adnotacji : {args.annotate}")
        from ocr.bbox_annotator import process_letter

        output_dir = process_letter(
            image_path=args.annotate,
            base_dir=args.annotation_dir,
            enable_box_edit=not args.no_edit,
            non_interactive=args.non_interactive,
        )
        if output_dir:
            info(f"Adnotacje zapisane w: {output_dir}")

    elif args.page:
        _require_file(args.page, info)
        info(f"\nUruchamianie adnotacji : {args.page}")
        from ocr.bbox_annotator import process_letter

        output_dir = process_letter(
            image_path=args.page,
            base_dir=args.annotation_dir,
            enable_box_edit=not args.no_edit,
            non_interactive=args.non_interactive,
        )
        if output_dir:
            info(f"Adnotacje zapisane w: {output_dir}")

        results = process_folder(output_dir, args, model_path, device, info)
        
        # Przygotuj dane do wyświetlania w rozmieszczeniu
        table_rows = []
        layout_words = []  # słowa z informacją o linii
        
        for r in results:
            if "error" in r:
                table_rows.append({
                    "key": None,
                    "file": r["file"],
                    "text": f"ERROR: {r['error']}",
                    "confidence": None
                })
                continue

            filename = os.path.basename(r["file"])      # word_061.png
            name, _ = os.path.splitext(filename)        # word_061

            table_rows.append({
                "key": name,                            # klucz sortowania
                "file": r["file"],
                "text": r["text"],
                "confidence": r["confidence"]
            })
            
            # Dodaj do listy dla wyświetlania rozmieszczenia
            layout_words.append({
                "text": r["text"],
                "confidence": r["confidence"],
                "bbox": r.get("bbox")
            })
        
        table_rows.sort(key=lambda r: r["key"] if r["key"] else "")
        
        # Wyświetl tabelę
        info(f"\n{'NAME':<12} {'TEXT':<20} {'CONF':<10}")
        info("-" * 45)

        for r in table_rows:
            if r["confidence"] is None:
                info(f"{r['key']:<12} {r['text']:<20} {'-':<10}")
            else:
                info(f"{r['key']:<12} {r['text']:<20} {r['confidence']:.2f}%")
        
        # Wyświetl w rozmieszczeniu
        display_text_layout(layout_words, info)
        


    elif args.folder:
        results = process_folder(args.folder, args, model_path, device, info)
        rows = []
        for r in results:
            if "error" in r:
                rows.append({
                    "key": None,
                    "file": r["file"],
                    "text": f"ERROR: {r['error']}",
                    "confidence": None
                })
                continue

            filename = os.path.basename(r["file"])      # word_061.png
            name, _ = os.path.splitext(filename)        # word_061

            rows.append({
                "key": name,                            # klucz sortowania
                "file": r["file"],
                "text": r["text"],
                "confidence": r["confidence"]
            })
            rows.sort(key=lambda r: r["key"])
        info(f"\n{'NAME':<12} {'TEXT':<20} {'CONF':<10}")
        info("-" * 45)

        for r in rows:
            if r["confidence"] is None:
                info(f"{r['key']:<12} {r['text']:<20} {'-':<10}")
            else:
                info(f"{r['key']:<12} {r['text']:<20} {r['confidence']:.2f}%")
        
    elif args.image:
        _require_file(args.image, info)
        info(f"\nRozpoznawanie: {args.image}")

        saved_copy_path = save_image_to_today_folder(args.image, info)
        model = load_model(model_path, device, info)

        result = predict_image(args.image, model, device, args)
        text = result["text"]
        confidence = result["confidence"]
        probs = result["probs"]

        output_handler = create_output_handler(args, source_image=args.image)
        output_handler.output(
            OCRResult(text=text, confidence=confidence, probs=probs, mode="single", class_labels=get_active_chars()),
            info,
        )

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
            out_path = args.json_path or (os.path.splitext(saved_copy_path)[0] + ".json")
            write_json(out_path, payload, pretty=args.json_pretty)
        else:
            info(f"TEXT: {text}")
            info(f"CONFIDENCE: {confidence:.2f}%")

        if args.accuracy:
            _print_accuracy(text, args.accuracy, info)

    # ── CRNN ──
    elif args.crnn:
        _require_file(args.crnn, info)
        info(f"\nRozpoznawanie CRNN: {args.crnn}")
        save_image_to_today_folder(args.crnn, info)

        model = load_model(model_path, device, info)
        result = predict_image(args.crnn, model, device, args)
        text = result["text"]
        confidence = result["confidence"]

        info(f"Rozpoznany tekst: '{text}' ({confidence:.1f}%)")
        if args.json:
            payload = {"file": args.crnn, "text": text, "confidence": confidence}
            info(dump_json(payload, pretty=args.json_pretty))

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

        if args.accuracy:
            _print_accuracy(word, args.accuracy, info)

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
            out_path = args.json_path or (os.path.splitext(saved_copy_path)[0] + ".json")
            write_json(out_path, payload, pretty=args.json_pretty)
        else:
            print_text_result(text, words_with_confidence, class_confidence, info)

        if args.accuracy:
            _print_accuracy(text, args.accuracy, info)



    # ── Wiele zdjęć ──
    elif args.multi:
        model = load_model(model_path, device, info)
        info(f"\nRozpoznawanie {len(args.multi)} pliku(-ow):")

        results = []
        for img_path in args.multi:
            if not os.path.exists(img_path):
                info(f"  Pominieto (nie znaleziono): {img_path}")
                continue

            save_image_to_today_folder(img_path, info)
            result = predict_image(img_path, model, device, args)
            results.append(
                {
                    "file": img_path,
                    "char": result["text"],
                    "confidence": result["confidence"],
                    "probs": result["probs"],
                }
            )

            if not args.json:
                print_multi_result(img_path, result["text"], result["confidence"], info)

        if args.json:
            payload = build_multi_result_json(results=results, device=str(device))
            info(dump_json(payload, pretty=args.json_pretty))
            out_path = args.json_path or "results.json"
            write_json(out_path, payload, pretty=args.json_pretty)
        else:
            for result in results:
                info(f"  {result['file']}  ->  '{result['char']}' ({result['confidence']:.1f}%)")

    else:
        parser.print_help()

    return "\n".join(buffor)


if __name__ == "__main__":
    main()
