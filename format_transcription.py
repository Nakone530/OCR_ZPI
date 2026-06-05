from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def split_words(text: str) -> List[str]:
    return text.split()


def split_lines(text: str) -> List[str]:
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def format_transcription(words: List[str], prefix: str = "word_") -> str:
    width = max(3, len(str(max(0, len(words) - 1))))
    lines = [f"{prefix}{index:0{width}d} {word}" for index, word in enumerate(words)]
    return "\n".join(lines)


def convert_file(input_path: Path, output_path: Path, prefix: str = "word_", treat_lines: bool = False) -> str:
    text = input_path.read_text(encoding="utf-8")
    items = split_lines(text) if treat_lines else split_words(text)
    output = format_transcription(items, prefix=prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Konwertuje zwykly tekst na format word_000 <slowo>.")
    parser.add_argument("input", type=Path, help="Plik z wejściową transkrypcją")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Plik wyjściowy (format trans). Jeśli nie podasz, zapisze do trans/<nazwa_wejścia>",
    )
    parser.add_argument("--prefix", default="word_", help="Prefix numeracji (domyślnie: word_)")
    parser.add_argument("--lines", action="store_true", help="Traktuj każdą linię wejścia jako oddzielny wpis (nie dziel na słowa)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_path = args.output if args.output is not None else Path("trans") / args.input.stem
    out = convert_file(args.input, output_path, prefix=args.prefix, treat_lines=args.lines)
    print(out)


if __name__ == "__main__":
    main()