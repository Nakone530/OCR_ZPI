"""
Benchmark autokorekty DictCorrect.

Uruchom: python test_correct.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
from ocr.utils import DictCorrect

CASES = [
    # (wejście HTR, confidence%, oczekiwane słowo)
    # --- typowe błędy modelu ---
    ("dorn",      50,  "dom"),
    ("zycie",     60,  "życie"),
    ("czlowiek",  50,  "człowiek"),
    ("Sloce",     55,  "Słońce"),
    ("jestm",     40,  "jestem"),
    ("bardzо",    70,  "bardzo"),   # cyrylica 'о' zamiast łacińskiego
    # --- słowa poprawne (nie powinny być zmieniane) ---
    ("dom",       95,  "dom"),
    ("nie",       90,  "nie"),
    ("człowiek",  85,  "człowiek"),
    # --- wysokie confidence, poprawne słowo poza słownikiem ---
    ("Kowalski",  92,  "Kowalski"),
]

ok = 0
for inp, conf, expected in CASES:
    result = DictCorrect(inp, conf)
    status = "OK" if result.lower() == expected.lower() else "FAIL"
    if status == "OK":
        ok += 1
    print(f"[{status}] '{inp}' (conf={conf}%) -> '{result}'  (oczekiwano: '{expected}')")

print(f"\n{ok}/{len(CASES)} poprawnych")
