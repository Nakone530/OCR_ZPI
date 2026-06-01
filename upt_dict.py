import json
import re
import argparse
from pathlib import Path


def extract_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĄąĆćĘęŁłŃńÓóŚśŹźŻż0-9'-]+", text)


def update_dictionary_from_jsonl(jsonl_path: Path, dictionary_set: set):
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = data.get("text", "")

            for word in extract_words(text):
                word = word.strip()
                if word:
                    dictionary_set.add(word)


def normalize_dictionary_lowercase(dictionary_set: set) -> set:
    """
    Zamienia wszystkie wpisy na lowercase i usuwa duplikaty wynikające z wielkości liter.
    """
    return {word.lower() for word in dictionary_set if isinstance(word, str)}


def update_dictionary(folder_path: str, dictionary_path: str):
    folder = Path(folder_path)

    # load dictionary
    if Path(dictionary_path).exists():
        with open(dictionary_path, "r", encoding="utf-8") as f:
            dictionary = json.load(f)
    else:
        dictionary = []

    dictionary_set = set(dictionary)

    # find jsonl files recursively
    jsonl_files = list(folder.rglob("*.jsonl"))
    print(f"Znaleziono {len(jsonl_files)} plików .jsonl")

    for file_path in jsonl_files:
        update_dictionary_from_jsonl(file_path, dictionary_set)

    # NORMALIZACJA NA KOŃCU
    dictionary_set = normalize_dictionary_lowercase(dictionary_set)

    # sort
    sorted_dictionary = sorted(dictionary_set)

    # save
    with open(dictionary_path, "w", encoding="utf-8") as f:
        json.dump(sorted_dictionary, f, ensure_ascii=False, indent=1)

    print(f"Zaktualizowano słownik: {len(sorted_dictionary)} słów")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--dict", required=True)

    args = parser.parse_args()

    update_dictionary(args.folder, args.dict)


if __name__ == "__main__":
    main()
