"""
Rozszerzenie słownika autokorekty HTR o polskie formy fleksyjne z SJP.

Źródło: sjp.pl/sl/growy/ — lista słów do gier planszowych (literaki, scrabble),
zawiera ~3 miliony poprawnych form polskich, licencja GPL 2 + CC BY 4.0.

Użycie:
    python build_dictionary.py                  # domyślnie 50 000 słów
    python build_dictionary.py --max-words 100000
    python build_dictionary.py --max-words 0    # bez limitu (może być >1M, wymaga >2GB RAM)
    python build_dictionary.py --dry-run        # tylko statystyki, bez zapisu

Pamięć RAM (przy _EMBED_DIM=2048 w utils.py):
    50 000 słów  →  ~400 MB
    100 000 słów →  ~800 MB
    200 000 słów →  ~1.6 GB
"""

import argparse
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

SJP_INDEX_URL = "https://sjp.pl/sl/growy/"
DICT_PATH = Path("data/dictionary.json")

# Akceptuj tylko czysto polskie słowa: litery łacińskie + diakrytyki + apostrof + łącznik
_VALID_RE = re.compile(r"^[a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ'-]*[a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ]$")


def _fetch_latest_sjp_url() -> str:
    req = urllib.request.Request(SJP_INDEX_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", errors="replace")
    m = re.search(r'href=["\']?(sjp-\d{8}\.zip)["\']?', html)
    if not m:
        raise RuntimeError(
            f"Nie znaleziono pliku zip na {SJP_INDEX_URL}\n"
            "Sprawdź ręcznie czy strona jest dostępna."
        )
    return f"https://sjp.pl/sl/growy/{m.group(1)}"


def _download_sjp_words(url: str) -> list[str]:
    print(f"Pobieranie: {url} …", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    print(f"  Pobrano {len(data) / 1_048_576:.1f} MB", flush=True)

    zf = zipfile.ZipFile(io.BytesIO(data))
    txt_name = next((n for n in zf.namelist() if n.endswith(".txt")), None)
    if txt_name is None:
        raise RuntimeError(f"Brak pliku .txt w archiwum: {zf.namelist()}")
    return zf.read(txt_name).decode("utf-8", errors="replace").splitlines()


def _filter_words(words: list[str], min_len: int = 3, max_len: int = 20) -> list[str]:
    """
    Filtruje listę słów:
      - długość w zakresie [min_len, max_len]
      - tylko polskie litery (bez cyfr, znaków specjalnych)
      - deduplikacja (case-insensitive)
    Zachowuje oryginalną wielkość liter pierwszego wystąpienia.
    """
    result = []
    seen: set[str] = set()
    for w in words:
        w = w.strip()
        if len(w) < min_len or len(w) > max_len:
            continue
        if not _VALID_RE.match(w):
            continue
        key = w.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(w)
    return result


def _prioritize(words: list[str], existing_lower: set[str]) -> list[str]:
    """
    Sortuje słowa według priorytetu dla HTR:
      1. Krótkie słowa (3–8 znaków) — najczęstsze w tekstach
      2. Średnie (9–13)
      3. Długie (14–20)
    W obrębie każdej grupy: alfabetycznie.
    Słowa już w słowniku są pomijane (zostaną dodane przed wywołaniem tej funkcji).
    """
    def priority(w: str) -> tuple[int, str]:
        n = len(w)
        tier = 0 if n <= 8 else (1 if n <= 13 else 2)
        return (tier, w.lower())

    new_words = [w for w in words if w.lower() not in existing_lower]
    return sorted(new_words, key=priority)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rozszerzenie słownika autokorekty HTR o formy fleksyjne z SJP."
    )
    parser.add_argument(
        "--max-words", type=int, default=50_000,
        help="Docelowa łączna liczba słów w słowniku (0 = bez limitu, domyślnie: 50000)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Tylko statystyki — nie zapisuje pliku"
    )
    parser.add_argument(
        "--min-len", type=int, default=3,
        help="Minimalna długość słowa (domyślnie: 3)"
    )
    parser.add_argument(
        "--max-len", type=int, default=20,
        help="Maksymalna długość słowa (domyślnie: 20)"
    )
    args = parser.parse_args()

    # 1. Wczytaj istniejący słownik
    existing: list[str] = []
    if DICT_PATH.exists():
        with open(DICT_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Istniejący słownik: {len(existing)} słów")
    else:
        print(f"Brak pliku {DICT_PATH} — zostanie stworzony nowy")

    existing_lower = {w.lower() for w in existing}

    # 2. Pobierz i odfiltruj słowa z SJP
    try:
        url = _fetch_latest_sjp_url()
    except Exception as e:
        print(f"Błąd przy pobieraniu strony SJP: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        raw = _download_sjp_words(url)
    except Exception as e:
        print(f"Błąd przy pobieraniu pliku: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wierszy w pliku SJP: {len(raw):,}")
    filtered = _filter_words(raw, args.min_len, args.max_len)
    print(f"Po filtracji (dł. {args.min_len}–{args.max_len}, tylko polskie znaki): {len(filtered):,}")

    # 3. Scal: istniejące słowa zachowane w całości, nowe posortowane według priorytetu
    prioritized_new = _prioritize(filtered, existing_lower)
    merged = existing + prioritized_new

    # 4. Przytnij do limitu (jeśli ustawiony)
    if args.max_words > 0 and len(merged) > args.max_words:
        cut = len(merged) - args.max_words
        merged = merged[: args.max_words]
        print(f"Przycięto {cut:,} słów do limitu {args.max_words:,}")

    print(f"\nWynik:")
    print(f"  Słów łącznie:        {len(merged):,}")
    print(f"  Z istniejącego:      {len(existing):,}")
    print(f"  Nowych z SJP:        {len(merged) - len(existing):,}")
    ram_mb = len(merged) * 2048 * 4 / 1_048_576
    print(f"  Szacowany RAM cache: {ram_mb:.0f} MB  (przy _EMBED_DIM=2048)")

    if args.dry_run:
        print("\n--dry-run: plik NIE został zapisany")
        return

    # 5. Zapisz
    DICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)
    print(f"\nZapisano: {DICT_PATH}")
    print("Przy następnym uruchomieniu programu cache embeddingów zostanie przebudowany.")


if __name__ == "__main__":
    main()
