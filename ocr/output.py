"""
Moduł obsługi różnych formatów wyjścia OCR.

Formaty:
  - console: standardowe wypisanie w konsoli
  - txt: zapis do pliku tekstowego
  - json: zapis do pliku JSON z prawdopodobieństwami

Wyniki są automatycznie zapisywane w folderze 'wynik/' razem z kopią zdjęcia.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from .config import CHARS

# Folder na wyniki
RESULT_DIR = Path("wynik")


class OCRResult:
    """Klasa przechowująca wynik OCR."""
    
    def __init__(
        self,
        text: str,
        confidence: Optional[float] = None,
        probs: Optional[torch.Tensor] = None,
        mode: str = "single"  # single, word, lines, multi
    ):
        self.text = text
        self.confidence = confidence
        self.probs = probs
        self.mode = mode
    
    def get_all_probs(self) -> dict[str, float]:
        """Zwraca słownik wszystkich prawdopodobieństw dla każdego znaku."""
        if self.probs is None:
            return {}
        return {
            CHARS[i]: round(self.probs[i].item() * 100, 2)
            for i in range(len(CHARS))
        }
    
    def get_top_n(self, n: int = 5) -> list[tuple[str, float]]:
        """Zwraca top N predykcji."""
        if self.probs is None:
            return [(self.text, self.confidence or 0.0)]
        
        top_probs, top_indices = torch.topk(self.probs, min(n, len(CHARS)))
        return [
            (CHARS[idx.item()], round(prob.item() * 100, 2))
            for prob, idx in zip(top_probs, top_indices)
        ]


class OutputHandler:
    """Obsługuje różne formaty wyjścia i zapis do folderu wynik/."""
    
    def __init__(
        self,
        output_path: Optional[str] = None,
        output_format: str = "console",
        verbose: bool = True,
        source_image: Optional[str] = None,
        save_to_result_dir: bool = True
    ):
        self.output_path = output_path
        self.output_format = output_format
        self.verbose = verbose
        self.source_image = source_image
        self.save_to_result_dir = save_to_result_dir
        self.result_subdir: Optional[Path] = None
    
    def _prepare_result_dir(self) -> Path:
        """Tworzy podfolder w wynik/ z timestampem."""
        RESULT_DIR.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subdir = RESULT_DIR / timestamp
        subdir.mkdir(exist_ok=True)
        
        self.result_subdir = subdir
        return subdir
    
    def _copy_source_image(self, subdir: Path) -> Optional[Path]:
        """Kopiuje źródłowe zdjęcie do folderu wyników."""
        if not self.source_image:
            return None
        
        source = Path(self.source_image)
        if not source.exists():
            return None
        
        dest = subdir / f"input{source.suffix}"
        shutil.copy2(source, dest)
        return dest
    
    def _get_result_filename(self, ext: str) -> Path:
        """Zwraca ścieżkę do pliku wynikowego."""
        if self.result_subdir:
            return self.result_subdir / f"wynik{ext}"
        return Path(f"wynik{ext}")
    
    def output(self, result: OCRResult) -> str:
        """Główna metoda - kieruje wynik do odpowiedniego formatu."""
        # Przygotuj folder wyników jeśli zapisujemy
        if self.save_to_result_dir and self.output_format != "console":
            subdir = self._prepare_result_dir()
            copied_img = self._copy_source_image(subdir)
            if copied_img and self.verbose:
                print(f"Skopiowano zdjęcie do: {copied_img}")
        
        if self.output_format == "json":
            return self._output_json(result)
        elif self.output_format == "txt":
            return self._output_txt(result)
        else:  # console
            return self._output_console(result)
    
    def _output_console(self, result: OCRResult) -> str:
        """Wypisuje wynik w konsoli (domyślne zachowanie)."""
        output_lines = []
        
        if result.mode == "single":
            output_lines.append("=" * 40)
            output_lines.append(f"WYNIK: '{result.text}' (pewność: {result.confidence:.1f}%)")
            output_lines.append("=" * 40)
            
            if result.probs is not None:
                output_lines.append("\nTop 5 predykcji:")
                for rank, (char, prob) in enumerate(result.get_top_n(5), start=1):
                    output_lines.append(f"  {rank}. '{char}' – {prob:.1f}%")
        
        elif result.mode == "word":
            output_lines.append(f"\nRozpoznany wyraz: '{result.text}'")
        
        elif result.mode == "lines":
            output_lines.append(f"\nRozpoznany tekst:\n{result.text}")
        
        elif result.mode == "multi":
            output_lines.append(f"  '{result.text}' ({result.confidence:.1f}%)")
        
        output_str = "\n".join(output_lines)
        
        if self.verbose:
            print(output_str)
        
        return output_str
    
    def _output_txt(self, result: OCRResult) -> str:
        """Zapisuje wynik do pliku TXT."""
        lines = []
        
        if result.mode == "single":
            lines.append(f"Rozpoznany znak: {result.text}")
            lines.append(f"Pewność: {result.confidence:.1f}%")
            if result.probs is not None:
                lines.append("")
                lines.append("Top 5 predykcji:")
                for rank, (char, prob) in enumerate(result.get_top_n(5), start=1):
                    lines.append(f"  {rank}. '{char}' - {prob:.1f}%")
        
        elif result.mode == "word":
            lines.append(f"Rozpoznany wyraz: {result.text}")
        
        elif result.mode == "lines":
            lines.append("Rozpoznany tekst:")
            lines.append(result.text)
        
        elif result.mode == "multi":
            lines.append(f"{result.text} ({result.confidence:.1f}%)")
        
        output_str = "\n".join(lines)
        
        # Określ ścieżkę do zapisu
        if self.output_path:
            save_path = Path(self.output_path)
        elif self.result_subdir:
            save_path = self._get_result_filename(".txt")
        else:
            save_path = None
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(output_str, encoding="utf-8")
            if self.verbose:
                print(f"Zapisano wynik do: {save_path}")
                print(output_str)
        else:
            # Bez pliku - wypisz na stdout (dla pipe)
            print(output_str)
        
        return output_str
    
    def _output_json(self, result: OCRResult) -> str:
        """Zapisuje wynik do pliku JSON."""
        data = {
            "result": result.text,
            "mode": result.mode,
        }
        
        if result.confidence is not None:
            data["confidence"] = round(result.confidence, 2)
        
        if result.probs is not None:
            data["top5"] = [
                {"char": char, "probability": prob}
                for char, prob in result.get_top_n(5)
            ]
            data["all_probabilities"] = result.get_all_probs()
        
        output_str = json.dumps(data, indent=2, ensure_ascii=False)
        
        # Określ ścieżkę do zapisu
        if self.output_path:
            save_path = Path(self.output_path)
        elif self.result_subdir:
            save_path = self._get_result_filename(".json")
        else:
            save_path = None
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(output_str, encoding="utf-8")
            if self.verbose:
                print(f"Zapisano wynik JSON do: {save_path}")
                print(output_str)
        else:
            # Bez pliku - wypisz na stdout (dla pipe)
            print(output_str)
        
        return output_str


def create_output_handler(args, source_image: Optional[str] = None) -> OutputHandler:
    """Tworzy OutputHandler na podstawie argumentów CLI."""
    output_format = getattr(args, "output_format", "console")
    output_path = getattr(args, "output", None)
    quiet = getattr(args, "quiet", False)
    
    # Jeśli podano ścieżkę bez formatu, wykryj z rozszerzenia
    if output_path and output_format == "console":
        ext = Path(output_path).suffix.lower()
        if ext == ".json":
            output_format = "json"
        elif ext == ".txt":
            output_format = "txt"
    
    # Automatycznie zapisuj do folderu wynik/ gdy format != console
    save_to_result_dir = output_format != "console"
    
    return OutputHandler(
        output_path=output_path,
        output_format=output_format,
        verbose=not quiet,
        source_image=source_image,
        save_to_result_dir=save_to_result_dir
    )
