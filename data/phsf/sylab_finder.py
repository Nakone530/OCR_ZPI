import argparse
from collections import Counter
import pyphen
import re
import csv

parser = argparse.ArgumentParser()
parser.add_argument("text_file", help="Plik tekstowy do analizy")
args = parser.parse_args()

dic = pyphen.Pyphen(lang="pl_PL")
counter = Counter()

syl_dir = "syllables.csv"

SAVE_EVERY = 10000
processed = 0

with open(args.text_file, encoding="utf-8") as f:
    for line in f:
        for word in line.split():
            
            # usuń interpunkcję, zostaw litery i cyfry
            word = re.sub(r"[^\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]", "", word)
            word = word.lower()

            if not word:
                continue

            syllables = dic.inserted(word).split("-")
            counter.update(syllables)
            syllables = dic.inserted(word).split("-")
            counter.update(syllables)

            processed += 1

            VOWELS = set("aąeęiouóy")



            if processed % SAVE_EVERY == 0:
                print(f"Zapis po {processed} słowach...")

                with open(syl_dir, "w", encoding="utf-8") as out:
                    for syl, count in counter.most_common():
                        out.write(f"{syl};{count}\n")
                        

# końcowy zapis
with open(syl_dir, "w", encoding="utf-8") as out:
    for syl, count in counter.most_common():
        out.write(f"{syl};{count}\n")
