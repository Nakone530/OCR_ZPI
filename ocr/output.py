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
from . import info

# Folder na wyniki
RESULT_DIR = Path("wynik")


class OCRResult:
    """
    Klasa przechowująca wynik rozpoznawania OCR.
    
    Przechowuje rozpoznany tekst wraz z dodatkowymi informacjami
    o pewności predykcji i rozkładzie prawdopodobieństw.
    
    Atrybuty:
        text (str): Rozpoznany tekst/znak.
        confidence (float | None): Pewność predykcji w procentach.
        probs (torch.Tensor | None): Tensor prawdopodobieństw dla każdej klasy.
        mode (str): Tryb rozpoznawania ('single', 'word', 'lines', 'multi').
    
    Przykład:
        >>> result = OCRResult("A", confidence=95.5, probs=probs_tensor, mode="single")
        >>> result.get_top_n(3)
        [('A', 95.5), ('H', 2.3), ('R', 1.1)]
    """
    
    def __init__(
        self,
        text: str,
        confidence: Optional[float] = None,
        probs: Optional[torch.Tensor] = None,
        mode: str = "single",  # single, word, lines, multi
        class_labels: Optional[list[str]] = None,
    ):
        """
        Inicjalizuje obiekt OCRResult.
        
        Argumenty:
            text (str): Rozpoznany tekst lub znak.
            confidence (float, opcjonalnie): Pewność predykcji (0-100). Domyślnie None.
            probs (torch.Tensor, opcjonalnie): Tensor prawdopodobieństw. Domyślnie None.
            mode (str, opcjonalnie): Tryb rozpoznawania. Domyślnie "single".
                Dostępne tryby:
                - 'single': pojedyncza litera
                - 'word': wyraz (jedna linia)
                - 'lines': tekst wieloliniowy
                - 'multi': przetwarzanie wielu plików
        """
        self.text = text
        self.confidence = confidence
        self.probs = probs
        self.mode = mode
        self.class_labels = class_labels or list(CHARS)
    
    def get_all_probs(self) -> dict[str, float]:
        """
        Zwraca słownik prawdopodobieństw dla wszystkich znaków.
        
        Argumenty:
            Brak argumentów.
        
        Zwraca:
            dict[str, float]: Słownik gdzie kluczem jest znak (A-Z),
                              a wartością prawdopodobieństwo w procentach.
                              Pusty słownik jeśli probs is None.
        
        Przykład:
            >>> result.get_all_probs()
            {'A': 95.50, 'B': 0.23, 'C': 0.15, ...}
        """
        if self.probs is None:
            return {}
        limit = min(len(self.class_labels), len(self.probs))
        return {
            self.class_labels[i]: round(self.probs[i].item() * 100, 2)
            for i in range(limit)
        }
    
    def get_top_n(self, n: int = 5) -> list[tuple[str, float]]:
        """
        Zwraca N najbardziej prawdopodobnych predykcji.
        
        Argumenty:
            n (int, opcjonalnie): Liczba top predykcji do zwrócenia. Domyślnie 5.
        
        Zwraca:
            list[tuple[str, float]]: Lista krotek (znak, prawdopodobieństwo).
                                     Posortowana malejąco wg prawdopodobieństwa.
        
        Przykład:
            >>> result.get_top_n(3)
            [('A', 95.50), ('H', 2.30), ('R', 1.10)]
        """
        if self.probs is None:
            return [(self.text, self.confidence or 0.0)]
        
        top_probs, top_indices = torch.topk(self.probs, min(n, len(self.probs)))
        return [
            (
                self.class_labels[idx.item()] if idx.item() < len(self.class_labels) else f"<UNK:{idx.item()}>",
                round(prob.item() * 100, 2)
            )
            for prob, idx in zip(top_probs, top_indices)
        ]


class OutputHandler:
    """
    Obsługuje różne formaty wyjścia i zapis wyników OCR.
    
    Klasa odpowiedzialna za formatowanie i zapisywanie wyników rozpoznawania
    w różnych formatach (konsola, TXT, JSON) oraz za zarządzanie folderem wyników.
    
    Atrybuty:
        output_path (str | None): Ścieżka do pliku wyjściowego.
        output_format (str): Format wyjścia ('console', 'txt', 'json').
        verbose (bool): Czy wypisywać komunikaty na konsolę.
        source_image (str | None): Ścieżka do obrazu źródłowego.
        save_to_result_dir (bool): Czy zapisywać do folderu wynik/.
        result_subdir (Path | None): Podfolder z timestampem w wynik/.
    
    Przykład:
        >>> handler = OutputHandler(output_format="json", verbose=True)
        >>> handler.output(result)
    """
    
    def __init__(
        self,
        output_path: Optional[str] = None,
        output_format: str = "console",
        verbose: bool = True,
        source_image: Optional[str] = None,
        save_to_result_dir: bool = True
    ):
        """
        Inicjalizuje OutputHandler.
        
        Argumenty:
            output_path (str, opcjonalnie): Ścieżka do pliku wyjściowego. Domyślnie None.
            output_format (str, opcjonalnie): Format wyjścia. Domyślnie "console".
                Dostępne formaty: 'console', 'txt', 'json'.
            verbose (bool, opcjonalnie): Czy wypisywać komunikaty. Domyślnie True.
            source_image (str, opcjonalnie): Ścieżka do obrazu źródłowego (do skopiowania).
            save_to_result_dir (bool, opcjonalnie): Czy zapisywać do wynik/. Domyślnie True.
        """
        self.output_path = output_path
        self.output_format = output_format
        self.verbose = verbose
        self.source_image = source_image
        self.save_to_result_dir = save_to_result_dir
        self.result_subdir: Optional[Path] = None
    
    def _prepare_result_dir(self) -> Path:
        """
        Tworzy podfolder w wynik/ z timestampem.
        
        Tworzy strukturę katalogów wynik/YYYYMMDD_HHMMSS/ dla przechowywania
        wyników i kopii obrazu źródłowego.
        
        Argumenty:
            Brak argumentów.
        
        Zwraca:
            Path: Ścieżka do utworzonego podfolderu.
        
        Efekty uboczne:
            - Tworzy folder wynik/ jeśli nie istnieje
            - Tworzy podfolder z timestampem
            - Ustawia self.result_subdir
        """
        RESULT_DIR.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subdir = RESULT_DIR / timestamp
        subdir.mkdir(exist_ok=True)
        
        self.result_subdir = subdir
        return subdir
    
    def _copy_source_image(self, subdir: Path) -> Optional[Path]:
        """
        Kopiuje źródłowe zdjęcie do folderu wyników.
        
        Argumenty:
            subdir (Path): Folder docelowy (podfolder wynik/).
        
        Zwraca:
            Path | None: Ścieżka do skopiowanego pliku lub None jeśli
                         brak obrazu źródłowego lub plik nie istnieje.
        """
        if not self.source_image:
            return None
        
        source = Path(self.source_image)
        if not source.exists():
            return None
        
        dest = subdir / f"input{source.suffix}"
        shutil.copy2(source, dest)
        return dest
    
    def _get_result_filename(self, ext: str) -> Path:
        """
        Zwraca ścieżkę do pliku wynikowego z podanym rozszerzeniem.
        
        Argumenty:
            ext (str): Rozszerzenie pliku (np. ".txt", ".json").
        
        Zwraca:
            Path: Ścieżka do pliku wynikowego.
        """
        if self.result_subdir:
            return self.result_subdir / f"wynik{ext}"
        return Path(f"wynik{ext}")
    
    def output(self, result: OCRResult) -> str:
        """
        Główna metoda wyjścia - kieruje wynik do odpowiedniego formatu.
        
        Automatycznie tworzy folder wyników (jeśli potrzebny),
        kopiuje obraz źródłowy i formatuje wynik.
        
        Argumenty:
            result (OCRResult): Obiekt z wynikiem rozpoznawania.
        
        Zwraca:
            str: Sformatowany wynik jako string.
        
        Efekty uboczne:
            - Może tworzyć foldery i pliki
            - Może wypisywać na konsolę (jeśli verbose=True)
        """
        # Przygotuj folder wyników jeśli zapisujemy
        if self.save_to_result_dir and self.output_format != "console":
            subdir = self._prepare_result_dir()
            copied_img = self._copy_source_image(subdir)
            if copied_img and self.verbose:
                info(f"Skopiowano zdjęcie do: {copied_img}")
        
        if self.output_format == "json":
            return self._output_json(result)
        elif self.output_format == "txt":
            return self._output_txt(result)
        else:  # console
            return self._output_console(result)
    
    def _output_console(self, result: OCRResult) -> str:
        """
        Formatuje i wypisuje wynik w konsoli.
        
        Argumenty:
            result (OCRResult): Obiekt z wynikiem rozpoznawania.
        
        Zwraca:
            str: Sformatowany wynik.
        
        Efekty uboczne:
            Wypisuje na stdout (jeśli verbose=True).
        """
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
            info(output_str)

        return output_str

    def _output_txt(self, result: OCRResult) -> str:
        """
        Zapisuje wynik do pliku tekstowego TXT.
        
        Argumenty:
            result (OCRResult): Obiekt z wynikiem rozpoznawania.
        
        Zwraca:
            str: Zawartość zapisana do pliku.
        
        Efekty uboczne:
            - Tworzy/nadpisuje plik TXT
            - Wypisuje komunikat o zapisie (jeśli verbose=True)
        """
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
                info(f"Zapisano wynik do: {save_path}")
                info(output_str)
        else:
            # Bez pliku - wypisz na stdout (dla pipe)
            info(output_str)
        
        return output_str
    
    def _output_json(self, result: OCRResult) -> str:
        """
        Zapisuje wynik do pliku JSON.
        
        Format JSON zawiera rozpoznany tekst, tryb, pewność
        oraz szczegółowe prawdopodobieństwa (top5 i wszystkie).
        
        Argumenty:
            result (OCRResult): Obiekt z wynikiem rozpoznawania.
        
        Zwraca:
            str: JSON string zapisany do pliku.
        
        Efekty uboczne:
            - Tworzy/nadpisuje plik JSON
            - Wypisuje komunikat o zapisie (jeśli verbose=True)
        """
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
                info(f"Zapisano wynik JSON do: {save_path}")
                info(output_str)
        else:
            # Bez pliku - wypisz na stdout (dla pipe)
            info(output_str)
        
        return output_str


def create_output_handler(args, source_image: Optional[str] = None) -> OutputHandler:
    """
    Funkcja fabryczna tworząca OutputHandler na podstawie argumentów CLI.
    
    Automatycznie wykrywa format wyjścia z rozszerzenia pliku
    jeśli nie został jawnie podany.
    
    Argumenty:
        args: Obiekt argparse.Namespace z parametrami. Oczekiwane atrybuty:
            - output_format (str, opcjonalnie): Format wyjścia ('console', 'txt', 'json')
            - output (str, opcjonalnie): Ścieżka do pliku wyjściowego
            - quiet (bool, opcjonalnie): Czy wyłączyć komunikaty (verbose=False)
        source_image (str, opcjonalnie): Ścieżka do obrazu źródłowego do skopiowania.
    
    Zwraca:
        OutputHandler: Skonfigurowany obiekt OutputHandler.
    
    Przykład:
        >>> handler = create_output_handler(args, source_image="input.png")
        >>> handler.output(result)
    
    Uwaga:
        Jeśli output_format=="console" ale podano ścieżkę z rozszerzeniem
        .json lub .txt, format zostanie automatycznie wykryty.
    """
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
