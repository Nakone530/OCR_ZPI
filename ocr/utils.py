"""
Narzędzia pomocnicze:
  - transformacje obrazów
  - odszumianie filtrem bilateralnym (OpenCV)
  - ładowanie obrazów (PNG/JPG/PDF) z opcjonalnym odszumianiem
  - zapis obrazu do folderu z dzisiejszą datą
"""

import os
from datetime import date
from pathlib import Path
import random
from difflib import SequenceMatcher
import json
try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None
import numpy as np
from PIL import Image
try:
    from pdf2image import convert_from_path  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    convert_from_path = None
try:
    import fitz  # type: ignore  # PyMuPDF
except ModuleNotFoundError:  # pragma: no cover
    fitz = None
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F
import csv
from datetime import datetime
from typing import Any
from datetime import date

from pathlib import Path
import matplotlib.pyplot as plt
import itertools
from .config import IMAGE_SIZE, MEAN, STD, CHARS, AuxCHARS, MODEL_PATH, NUM_CLASSES, char2idx, idx2char, VERSION_RE
from .model import MainModel, AuxModel
from . import info

def load_dictionary() -> set:
    """Zwraca zbi\u00f3r znanych polskich s\u0142\u00f3w."""
    return {
        # Przyimki / sp\u00f3jniki / zaimki
        "w", "z", "na", "do", "po", "o", "za", "przez", "nad", "pod", "przed", "za",
        "mi\u0119dzy", "przy", "od", "dla", "bez", "u", "mimo", "wed\u0142ug",
        "podczas", "pomimo", "wobec", "dzi\u0119ki", "przeciw", "naprzeciw",
        "i", "a", "ale", "lub", "albo", "\u017ce", "bo", "wi\u0119c", "jednak",
        "oraz", "ani", "czy", "je\u015bli", "gdy", "poniewa\u017c", "dlatego", "tote\u017c",
        "ten", "ta", "to", "ci", "te", "tamten", "tamta", "tamto", "tamci", "tamte",
        "m\u00f3j", "tw\u00f3j", "jego", "jej", "nasz", "wasz", "ich", "sw\u00f3j",
        "kto\u015b", "co\u015b", "nikt", "nic", "ka\u017cdy", "wszyscy", "wszystko",
        "ja", "ty", "on", "ona", "ono", "my", "wy", "oni", "one",
        "mnie", "ciebie", "siebie", "sobie", "si\u0119",

        # Liczebniki
        "jeden", "dwa", "trzy", "cztery", "pi\u0119\u0107", "sze\u015b\u0107",
        "siedem", "osiem", "dziewi\u0119\u0107", "dziesi\u0119\u0107",
        "jedena\u015bcie", "dwana\u015bcie", "dwadzie\u015bcia", "trzydzie\u015bci",
        "sto", "dwie\u015bcie", "trzysta", "czterysta", "pi\u0119\u0107set",
        "tysi\u0105c", "tysi\u0105ce", "milion", "miliard",
        "pierwszy", "drugi", "trzeci", "czwarty", "pi\u0105ty",
        "sz\u00f3sty", "si\u00f3dmy", "\u00f3smy", "dziewi\u0105ty", "dziesi\u0105ty",
        "ostatni", "pierwszych", "ostatnich",

        # Czasowniki (bezokoliczniki i formy)
        "by\u0107", "jest", "s\u0105", "by\u0142", "by\u0142a", "by\u0142o", "byli",
        "by\u0142y", "b\u0119d\u0119", "b\u0119dziesz", "b\u0119dzie", "b\u0119dziemy",
        "b\u0119dziecie", "b\u0119d\u0105", "by\u0107", "b\u0105d\u017a",
        "jestem", "jeste\u015b", "jeste\u015bmy", "jeste\u015bcie",
        "mie\u0107", "ma", "mam", "masz", "mamy", "macie", "maj\u0105",
        "mia\u0142", "mia\u0142a", "mia\u0142o", "mieli", "mia\u0142y",
        "b\u0119dzie", "b\u0119d\u0105",
        "m\u00f3c", "mog\u0119", "mo\u017cesz", "mo\u017ce", "mo\u017cemy", "mo\u017cecie",
        "mog\u0105", "m\u00f3g\u0142", "mog\u0142a", "mog\u0142o", "mogli",
        "chcie\u0107", "chc\u0119", "chcesz", "chce", "chcemy", "chcecie",
        "chc\u0105", "chcia\u0142", "chcia\u0142a", "chcieli",
        "wiedzie\u0107", "wiem", "wiesz", "wie", "wiemy", "wiecie", "wiedz\u0105",
        "wiedzia\u0142", "wiedzia\u0142a", "wiedzieli",
        "m\u00f3wi\u0107", "m\u00f3wi\u0119", "m\u00f3wisz", "m\u00f3wi", "m\u00f3wimy",
        "m\u00f3wicie", "m\u00f3wi\u0105", "m\u00f3wi\u0142", "m\u00f3wi\u0142a",
        "robi\u0107", "robi\u0119", "robisz", "robi", "robimy", "robicie",
        "robi\u0105", "robi\u0142", "robi\u0142a", "robili",
        "zrobi\u0107", "zrobi\u0119", "zrobisz", "zrobi", "zrobimy",
        "zrobi\u0142", "zrobi\u0142a", "zrobili",
        "i\u015b\u0107", "id\u0119", "idziesz", "idzie", "idziemy", "idziecie",
        "id\u0105", "szed\u0142", "sz\u0142a", "szli",
        "p\u00f3j\u015b\u0107", "p\u00f3jd\u0119", "p\u00f3jdziesz", "p\u00f3jdzie",
        "poszed\u0142", "posz\u0142a", "poszli",
        "sta\u0107", "stoj\u0119", "stoimy", "stoisz", "stoi", "stoj\u0105",
        "sta\u0142", "sta\u0142a",
        "le\u017ce\u0107", "le\u017c\u0119", "le\u017cy", "le\u017c\u0105", "le\u017ca\u0142",
        "siedzie\u0107", "siedz\u0119", "siedzisz", "siedzi", "siedzimy",
        "siedz\u0105", "siedzia\u0142", "siedzia\u0142a",
        "czyta\u0107", "czytam", "czytasz", "czyta", "czytamy", "czytacie",
        "czytaj\u0105", "czyta\u0142", "czyta\u0142a", "czytali",
        "pisa\u0107", "pisz\u0119", "piszesz", "pisze", "piszemy", "piszecie",
        "pisz\u0105", "pisa\u0142", "pisa\u0142a", "pisali",
        "widzie\u0107", "widz\u0119", "widzisz", "widzi", "widzimy", "widzicie",
        "widz\u0105", "widzia\u0142", "widzia\u0142a",
        "s\u0142ysze\u0107", "s\u0142ysz\u0119", "s\u0142yszysz", "s\u0142yszy",
        "s\u0142yszymy", "s\u0142ysz\u0105", "s\u0142ysza\u0142",
        "my\u015ble\u0107", "my\u015bl\u0119", "my\u015blisz", "my\u015bli",
        "my\u015blimy", "my\u015bl\u0105", "my\u015bla\u0142", "my\u015bla\u0142a",
        "je\u015b\u0107", "jem", "jesz", "je", "jemy", "jecie", "jedz\u0105",
        "jad\u0142", "jad\u0142a", "jedli",
        "pi\u0107", "pij\u0119", "pijesz", "pije", "pijemy", "pijecie",
        "pij\u0105", "pi\u0142", "pi\u0142a", "pili",
        "spa\u0107", "\u015bpi\u0119", "\u015bpisz", "\u015bpi", "\u015bpimy",
        "\u015bpi\u0105", "spa\u0142", "spa\u0142a",
        "gra\u0107", "gram", "grasz", "gra", "gramy", "gracie", "graj\u0105",
        "gra\u0142", "gra\u0142a", "grali",
        "bra\u0107", "bior\u0119", "bierzesz", "bierze", "bierzemy", "bierzecie",
        "bior\u0105", "bra\u0142", "bra\u0142a",
        "da\u0107", "dam", "dasz", "da", "damy", "dacie", "dadz\u0105",
        "da\u0142", "da\u0142a", "dali",
        "kupi\u0107", "kupi\u0119", "kupisz", "kupi", "kupimy", "kupicie",
        "kupi\u0105", "kupi\u0142", "kupi\u0142a",
        "szuka\u0107", "szukam", "szukasz", "szuka", "szukamy", "szukacie",
        "szukaj\u0105", "szuka\u0142", "szuka\u0142a",
        "znale\u017a\u0107", "znajd\u0119", "znajdziesz", "znajdzie",
        "znalaz\u0142", "znalaz\u0142a", "znale\u017ali",
        "czeka\u0107", "czekam", "czekasz", "czeka", "czekamy", "czekacie",
        "czekaj\u0105", "czeka\u0142", "czeka\u0142a",
        "lubi\u0107", "lubi\u0119", "lubisz", "lubi", "lubimy", "lubicie",
        "lubi\u0105", "lubi\u0142", "lubi\u0142a",
        "kocha\u0107", "kocham", "kochasz", "kocha", "kochamy", "kochacie",
        "kochaj\u0105", "kocha\u0142", "kocha\u0142a",
        "pracowa\u0107", "pracuj\u0119", "pracujesz", "pracuje", "pracujemy",
        "pracujecie", "pracuj\u0105", "pracowa\u0142", "pracowa\u0142a",
        "mieszka\u0107", "mieszkam", "mieszkasz", "mieszka", "mieszkamy",
        "mieszkaj\u0105", "mieszka\u0142", "mieszka\u0142a",
        "umie\u0107", "umiem", "umiesz", "umie", "umiemy", "umiej\u0105",
        "umia\u0142", "umia\u0142a",
        "rozumie\u0107", "rozumiem", "rozumiesz", "rozumie", "rozumiemy",
        "rozumiej\u0105", "rozumia\u0142", "rozumia\u0142a",
        "musie\u0107", "musz\u0119", "musisz", "musi", "musimy", "musicie",
        "musz\u0105", "musia\u0142", "musia\u0142a",
        "wydaje", "wydaj\u0119", "wydajesz", "wydaj\u0105",
        "powiedzie\u0107", "powiem", "powiesz", "powie", "powiemy",
        "powiecie", "powiedz\u0105", "powiedzia\u0142", "powiedzia\u0142a",
        "mo\u017cna", "trzeba", "warto", "wida\u0107", "s\u0142ycha\u0107",

        # Rzeczowniki (cz\u0119ste)
        "cz\u0142owiek", "ludzie", "osoba", "kobieta", "m\u0119\u017cczyzna",
        "dziecko", "dzieci", "ch\u0142opiec", "dziewczyna",
        "rodzina", "rodzice", "matka", "ojciec", "mama", "tata",
        "brat", "siostra", "syn", "c\u00f3rka", "\u017cona", "m\u0105\u017c",
        "przyjaciel", "znajomy", "s\u0105siad", "go\u015b\u0107",
        "pan", "pani", "pa\u0144stwo", "towarzysz",
        "\u017cycie", "\u015bmier\u0107", "mi\u0142o\u015b\u0107", "przyja\u017a\u0144",
        "szcz\u0119\u015bcie", "rado\u015b\u0107", "smutek", "gniew", "strach",
        "nadzieja", "wiara", "pok\u00f3j", "wojna",
        "czas", "chwila", "moment", "dzie\u0144", "tydzie\u0144",
        "miesi\u0105c", "rok", "lata", "rana", "wiecz\u00f3r", "noc",
        "rano", "po\u0142udnie", "wiecz\u00f3r", "p\u00f3\u0142noc",
        "wiosna", "lato", "jesie\u0144", "zima",
        "stycze\u0144", "luty", "marzec", "kwiecie\u0144", "maj", "czerwiec",
        "lipiec", "sierpie\u0144", "wrzesie\u0144", "pa\u017adziernik",
        "listopad", "grudzie\u0144",
        "\u015bwiat", "ziemia", "niebo", "s\u0142o\u0144ce", "ksi\u0119\u017cyc",
        "gwiazda", "gwiazdy", "woda", "ogie\u0144", "powietrze",
        "g\u00f3ra", "morze", "las", "pole", "\u0142\u0105ka", "rzeka",
        "jezioro", "droga", "ulica", "miasto", "wie\u015b", "kraj",
        "dom", "mieszkanie", "pok\u00f3j", "drzwi", "okno", "\u015bciana",
        "pod\u0142oga", "dach", "kuchnia", "\u0142azienka", "ogr\u00f3d",
        "st\u00f3\u0142", "krzes\u0142o", "\u0142\u00f3\u017cko", "szafa", "biurko",
        "ksi\u0105\u017cka", "ksi\u0105\u017cki", "list", "obraz", "zdj\u0119cie",
        "d\u017awi\u0119k", "g\u0142os", "muzyka", "piosenka", "s\u0142owo",
        "j\u0119zyk", "litera", "tekst", "strona", "strony",
        "szko\u0142a", "uniwersytet", "klasa", "lekcja", "lekcje",
        "nauczyciel", "ucze\u0144", "uczennica", "student",
        "praca", "pracownik", "pracodawca", "zaw\u00f3d",
        "biuro", "fabryka", "sklep", "restauracja", "szpital",
        "samoch\u00f3d", "poci\u0105g", "samolot", "rower", "autobus",
        "g\u0142owa", "serce", "r\u0119ka", "noga", "oko", "oczy", "ucha",
        "usta", "d\u0142o\u0144", "palec", "w\u0142osy", "sk\u00f3ra",
        "jedzenie", "woda", "chleb", "mleko", "mi\u0119so",
        "warzywa", "owoce", "ciasto", "cukier", "s\u00f3l",
        "herbata", "kawa", "piwo", "wino", "sok",
        "ubranie", "buty", "kapelusz", "p\u0142aszcz",
        "historia", "historie", "literatura", "nauka", "sztuka",
        "muzyka", "poezja", "film", "teatr", "taniec",
        "nar\u00f3d", "pa\u0144stwo", "spo\u0142ecze\u0144stwo",
        "w\u0142adza", "prezydent", "rz\u0105d", "polityka",
        "prawo", "s\u0105d", "s\u0105dy", "wi\u0119zienie",
        "gospodarka", "pieni\u0105dze", "pieni\u0105dz", "cena", "bud\u017cet",
        "religia", "B\u00f3g", "ko\u015bci\u00f3\u0142", "ksi\u0105dz",
        "dusza", "niebo", "piek\u0142o", "anio\u0142",
        "my\u015bl", "my\u015bli", "pomys\u0142", "idea", "plan",
        "problem", "sprawa", "kwestia", "temat", "przyczyna",
        "skutek", "rezultat", "wynik", "wniosek", "wnioski",
        "cel", "zadanie", "obowi\u0105zek", "rola", "funkcja",
        "system", "metoda", "spos\u00f3b", "proces",
        "informacja", "dane", "wiedza", "fakt",
        "badanie", "badania", "eksperyment", "analiza",
        "ksi\u0119ga", "zapis", "dokument", "dokumenty",  "formularz",

        # Przymiotniki (cz\u0119ste)
        "dobry", "dobra", "dobre", "dobrzy", "dobre",
        "z\u0142y", "z\u0142a", "z\u0142e", "\u017ali", "z\u0142e",
        "du\u017cy", "du\u017ca", "du\u017ce", "du\u017cy",
        "ma\u0142y", "ma\u0142a", "ma\u0142e", "mali",
        "nowy", "nowa", "nowe", "nowi",
        "stary", "stara", "stare", "starzy",
        "m\u0142ody", "m\u0142oda", "m\u0142ode", "m\u0142odzi",
        "pi\u0119kny", "pi\u0119kna", "pi\u0119kne", "pi\u0119kni",
        "brzydki", "brzydka", "brzydkie",
        "wielki", "wielka", "wielkie",
        "wa\u017cny", "wa\u017cna", "wa\u017cne", "wa\u017cni",
        "trudny", "trudna", "trudne", "trudni",
        "\u0142atwy", "\u0142atwa", "\u0142atwe",
        "szybki", "szybka", "szybkie", "szybcy",
        "wolny", "wolna", "wolne", "wolni",
        "d\u0142ugi", "d\u0142uga", "d\u0142ugie",
        "kr\u00f3tki", "kr\u00f3tka", "kr\u00f3tkie",
        "wysoki", "wysoka", "wysokie", "wysocy",
        "niski", "niska", "niskie", "niscy",
        "szeroki", "szeroka", "szerokie",
        "g\u0142\u0119boki", "g\u0142\u0119boka", "g\u0142\u0119bokie",
        "czysty", "czysta", "czyste", "czy\u015bci",
        "brudny", "brudna", "brudne",
        "ciep\u0142y", "ciep\u0142a", "ciep\u0142e",
        "zimny", "zimna", "zimne",
        "gor\u0105cy", "gor\u0105ca", "gor\u0105ce",
        "mi\u0119kki", "mi\u0119kka", "mi\u0119kkie",
        "twardy", "twarda", "twarde",
        "lekki", "lekka", "lekkie",
        "ci\u0119\u017cki", "ci\u0119\u017cka", "ci\u0119\u017ckie",
        "pusty", "pusta", "puste",
        "pe\u0142ny", "pe\u0142na", "pe\u0142ne",
        "silny", "silna", "silne", "silni",
        "s\u0142aby", "s\u0142aba", "s\u0142abe", "s\u0142abi",
        "bogaty", "bogata", "bogate",
        "biedny", "biedna", "biedne",
        "szcz\u0119\u015bliwy", "szcz\u0119\u015bliwa", "szcz\u0119\u015bliwe",
        "smutny", "smutna", "smutne",
        "weso\u0142y", "weso\u0142a", "weso\u0142e",
        "m\u0105dry", "m\u0105dra", "m\u0105dre",
        "g\u0142upi", "g\u0142upia", "g\u0142upie",
        "mi\u0142y", "mi\u0142a", "mi\u0142e",
        "prawdziwy", "prawdziwa", "prawdziwe",
        "fa\u0142szywy", "fa\u0142szywa", "fa\u0142szywe",
        "polski", "polska", "polskie", "polscy",
        "polskiego", "polskiej", "polskich", "polskim",
        "europejski", "europejska", "europejskie",
        "kolejny", "kolejna", "kolejne",
        "inny", "inna", "inne", "inni",
        "sam", "sama", "samo", "sami", "same",
        "taki", "taka", "takie", "tacy",
        "nasz", "nasza", "nasze", "nasi",
        "wasz", "wasza", "wasze",

        # Przys\u0142\u00f3wki
        "bardzo", "tak", "nie", "ju\u017c", "jeszcze", "zawsze",
        "nigdy", "cz\u0119sto", "rzadko", "znowu", "zn\u00f3w",
        "teraz", "potem", "p\u00f3\u017aniej", "nast\u0119pnie", "wcze\u015bniej",
        "zawsze", "nigdy", "czasami", "czasem",
        "tu", "tutaj", "tam", "gdzie", "wsz\u0119dzie",
        "dobrze", "\u017ale", "\u0142adnie", "brzydko", "cicho", "g\u0142o\u015bno",
        "szybko", "wolno", "ostro\u017cnie", "uwa\u017cnie",
        "naprawd\u0119", "prawie", "ledwie", "nawet", "te\u017c",
        "tylko", "w\u0142a\u015bnie", "w\u0142a\u015bciwie", "by\u0107mo\u017ce",
        "mo\u017ce", "chyba", "oczywi\u015bcie", "pewnie", "zapewne",
        "ponownie", "razem", "osobno", "raz", "dwa razy",
        "przynajmniej", "co najmniej", "oko\u0142o", "blisko", "daleko",
        "wszystkich", "zbyt", "do\u015b\u0107", "raczej", "niczym",

        # Dodatkowe popularne
        "kt\u00f3ry", "kt\u00f3ra", "kt\u00f3re", "kt\u00f3rzy",
        "kt\u00f3rego", "kt\u00f3rej", "kt\u00f3rym", "kt\u00f3rych",
        "co", "czego", "czemu", "czym", "kim", "komu",
        "kiedy", "gdzie", "dok\u0105d", "sk\u0105d", "dlaczego",
        "jak", "jaki", "jaka", "jakie", "jacy",
        "jaki\u015b", "jaka\u015b", "jakie\u015b", "jacy\u015b",
        "ile", "wielu", "niewielu", "kilka", "kilku",
        "kilkana\u015bcie", "kilkadziesi\u0105t", "kilkaset",
        "nieco", "nie", "ni", "nijak",

        # Wyra\u017cenia powitalne / grzeczno\u015bciowe
        "dzie\u0144", "dobry", "witaj", "witam", "cze\u015b\u0107",
        "dzi\u0119kuj\u0119", "dzi\u0119ki", "prosz\u0119", "przepraszam",
        "przepraszamy", "do widzenia", "do zobaczenia",
        "cze\u015b\u0107", "pa pa", "\u017cegna\u0144",
        "bardzo prosz\u0119", "nie ma za co",
        "przykro mi", "niestety", "oczywi\u015bcie",

        # Zwierz\u0119ta i zwierz\u0105tka
        "kot", "kota", "kotu", "kocie", "koty", "kot\u00f3w", "psa", "psie",
        "pies", "psem", "psy", "ptak", "ptaka", "ptaki", "ryba", "ryby",
        "sowa", "sowy", "krowa", "ko\u0144", "koniu", "konie",

        # Ro\u015bliny i przyroda
        "drzewo", "drzewa", "kwiat", "kwiaty", "li\u015b\u0107", "li\u015bcie",
        "trawa", "trawy", "owo", "owoc", "owoce", "warzywo", "warzywa",
        "pole", "pola", "sad", "ogrodzie", "ogr\u00f3d",

        # Jedzenie (dodatkowe)
        "mas\u0142o", "ser", "jajko", "jajka", "ry\u017c", "makaron",
        "ziemniaki", "pomidor", "sa\u0142ata", "kapusta", "marchewka",
        "jab\u0142ko", "jab\u0142ka", "banan", "pomara\u0144cza",
        "cukierki", "czekolada", "lody",

        # Miasto / budynki
        "budynek", "budynku", "budynki", "ko\u015bci\u00f3\u0142",
        "szko\u0142y", "szko\u0142\u0105", "szpitala", "apteka",
        "poczta", "dworzec", "stacja", "kino", "teatry",
        "muzeum", "park", "parku", "most", "mostu", "plac",

        # Cia\u0142o cz\u0142owieka (dodatkowe)
        "r\u0119ce", "r\u0105k", "nog\u0105", "nog\u0119", "nogami",
        "palce", "palc\u00f3w", "d\u0142onie", "d\u0142oni",
        "ramiona", "barki", "plecy", "kark", "szyja",
        "twarz", "g\u0119ba", "\u0107wiek", "brodzie", "czo\u0142o",
        "w\u0142osami", "warkocz", "brwi",

        # Ubrania
        "spodnie", "spodenki", "koszula", "koszule", "sukienka",
        "kurtka", "kurtk\u0119", "czapka", "r\u0119kawiczki",
        "skarpety", "but\u00f3w", "butem",

        # Pogoda i klimat
        "s\u0142o\u0144ce", "deszcz", "\u015bnieg", "mgr", "mg\u0142a",
        "chmura", "chmury", "wiatr", "wietrze", "burza",
        "grad", "upa\u0142", "mroz", "ciep\u0142o", "ch\u0142odno",

        # Kolory
        "czerwony", "czerwona", "czerwone", "niebieski", "niebieska",
        "zielony", "zielona", "zielone", "\u017c\u00f3\u0142ty", "\u017c\u00f3\u0142ta",
        "bia\u0142y", "bia\u0142a", "bia\u0142e", "czarny", "czarna", "czarne",
        "szary", "szara", "szare", "fioletowy", "pomara\u0144czowy",
        "granatowy", "turkusowy", "r\u00f3\u017cowy", "r\u00f3\u017cowa",

        # Cechy fizyczne
        "gruby", "gruba", "grube", "chudy", "chuda", "chude",
        "prosty", "prosta", "proste", "krzywy", "krzywa", "krzywe",
        "okr\u0105g\u0142y", "okr\u0105g\u0142a", "kwadratowy", "tr\u00f3jk\u0105tny",
        "g\u0142adki", "g\u0142adka", "chropowaty",

        # Czas (dodatkowe)
        "sekunda", "minuta", "godzina", "godzin\u0119",
        "poniedzia\u0142ek", "wtorek", "\u015broda", "czwartek",
        "pi\u0105tek", "sobota", "niedziela",
        "weekend", "urlop", "wakacje", "ferie",

        # Czynno\u015bci codzienne
        "my\u0107", "myje", "myj\u0119", "myjemy",
        "pra\u0107", "pierze", "pranie",
        "sprz\u0105ta\u0107", "sprz\u0105tam", "sprz\u0105ta",
        "gotowa\u0107", "gotuj\u0119", "gotuje", "gotujemy",
        "rysowa\u0107", "rysuj\u0119", "rysuje", "malowa\u0107",
        "\u015bpiewa\u0107", "\u015bpiewam", "\u015bpiewa",
        "bawi\u0107", "bawi\u0119", "bawi", "bawi\u0105",
        "ta\u0144czy\u0107", "ta\u0144cz\u0119", "ta\u0144czy",
        "biega\u0107", "biegam", "biega",

        # Emocje i stany
        "zaskoczony", "zdziwiony", "zmartwiony", "zaniepokojony",
        "zm\u0119czony", "zm\u0119czona", "chory", "chora", "chore",
        "zdrowy", "zdrowa", "zdrowe", "g\u0142odny", "g\u0142odna",
        "spragniony", "senny", "senna", "samotny",

        # Cz\u0119\u015bci zdania
        "kt\u00f3rym", "kt\u00f3rej", "kt\u00f3rzy", "kt\u00f3rych",
        "kt\u00f3rego", "jakiego", "jakiej", "jakim", "jakich",
        "wszystkim", "wszystkich", "wszystkimi",
        "takiego", "takiej", "takim", "takich",
        "naszego", "naszej", "naszym", "naszych",
        "waszego", "waszej", "waszym", "waszych",
        "tego", "tej", "tym", "tych", "temi",

        # Partyku\u0142y, \u0141\u0105czniki, Modyfikatory
        "nawet", "przynajmniej", "w\u0142a\u015bnie", "w\u0142a\u015bciwie",
        "oczywi\u015bcie", "naturalnie", "ewentualnie",
        "ewidentnie", "zdecydowanie", "zw\u0142aszcza",
        "szczeg\u00f3lnie", "g\u0142\u00f3wnie", "przede wszystkim",
        "na przyk\u0142ad", "mi\u0119dzy innymi", "w dodatku",
        "ponadto", "wr\u0119cz", "niemal", "\u017cadnego",
        "co\u015b", "kto\u015b", "cokolwiek",

        # S\u0142owa na co dzie\u0144
        "dzwonek", "klucz", "klucze", "portfel", "parasol",
        "szczoteczka", "r\u0119cznik", "po\u015bciel",
        "koc", "poduszka", "poduszki",
        "\u015bwieca", "\u015bwiece", "lampa", "lampy",
        "zegar", "zegarek", "kalendarz",
        "lustro", "lusterko", "szafka", "szafki",

        # Podr\u00f3\u017ce
        "podr\u00f3\u017c", "podr\u00f3\u017cy", "wycieczka", "wycieczki",
        "wakacji", "urlopie", "mapa", "bilety",
        "hotel", "hotelu", "pokoju", "recepcja",
        "kierowca", "pasa\u017cer", "pilot", "przewodnik",

        # Praca i edukacja
        "sekretarka", "sekretarz", "dyrektor", "dyrektorka",
        "kierownik", "kierowniczka", "szef", "szefowa",
        "sta\u017c", "etat", "pensja", "wyp\u0142ata",
        "egzamin", "egzaminy", "kolokwium",
        "dyplom", "\u015bwiadectwo", "certyfikat",
        "biblioteka", "biblioteki", "ksi\u0119garnia",
        "zaj\u0119cia", "wyk\u0142ady", "seminarium",
        "ocena", "stopie\u0144", "stopnie",

        # Polska specyfika
        "warszawa", "krak\u00f3w", "gda\u0144sk", "wroc\u0142aw",
        "pozna\u0144", "\u0142\u00f3d\u017a", "katowice", "lublin",
        "warszawy", "warszawie", "krakowa", "krakowie",
        "polak", "polka", "polacy",
        "wis\u0142a", "morze ba\u0142tyckie",
        "sejm", "senat", "prezydent", "premier",
        "rz\u0105du", "rz\u0105dem", "rz\u0105dowy",
        "samorz\u0105d", "gmina", "powiat", "wojew\u00f3dztwo",
        "wolno\u015b\u0107", "niepodleg\u0142o\u015b\u0107", "solidarno\u015b\u0107",
        "powstanie", "powstania",

        # Przyimki z\u0142o\u017cone
        "pomi\u0119dzy", "wewn\u0105trz", "zewn\u0105trz",
        "naprzeciwko", "naprzeciw", "wok\u00f3\u0142", "dooko\u0142a",
        "opr\u00f3cz", "zamiast", "pomimo", "wzd\u0142u\u017c",
        "wg", "wedle", "wszech",

        # Rzadkie ale wa\u017cne
        "b\u00f3g", "boga", "bogu", "bo\u017ce",
        "anio\u0142y", "diabe\u0142", "diabli",
        "niebiosa", "piekielny",
        "krzy\u017c", "krzy\u017ca",
        "modlitwa", "modlitwy", "modli\u0107 si\u0119",
        "ko\u015bcio\u0142a", "ko\u015bcio\u0142em", "parafia",

        # Moda i uroda
        "makija\u017c", "szminka", "puder", "krem",
        "perfumy", "woda toaletowa",
        "fryzura", "fryzury", "grzywka",
        "manicure", "pedicure",

        # Zdrowie
        "grypa", "przezi\u0119bienie", "kaszel", "gor\u0105czka",
        "b\u00f3l", "b\u00f3lu", "b\u00f3lem",
        "tabletka", "tabletki", "lek", "leki", "lekarstwa",
        "doktor", "lekarz", "lekarza", "piel\u0119gniarka",
        "wizyta", "badania", "zabieg", "operacja",
        "rehabilitacja", "masa\u017c",

        # Sport
        "pi\u0142ka", "pi\u0142k\u0105", "bramka", "bramki",
        "mecz", "meczu", "zawody", "zawodnik",
        "trening", "\u0107wiczenia", "hala sportowa",
        "basen", "boisko", "stadion",
        "bieganie", "p\u0142ywanie", "jazda", "spacer",

        # technika
        "komputer", "komputera", "laptop", "klawiatura",
        "myszka", "ekran", "monitor", "drukarka",
        "telefon", "telefony", "smartfon",
        "aparat", "kamery", "s\u0142uchawki",
        "bateria", "\u0142adowarka", "kabel",
        "strona internetowa", "email",
        "program", "aplikacja", "system",
        "danych", "plik", "pliki", "folder",

        # Sztuka
        "obraz", "obrazy", "malarz", "malarstwo",
        "rze\u017aba", "rze\u017aby", "rze\u017abiarz",
        "wystawa", "wystawy", "galeria",
        "koncert", "koncerty", "orkiestra",
        "aktor", "aktorka", "aktorzy",
        "re\u017cyser", "spektakl",
    }
# ── Transformacje ──────────────────────────────────────────────────────────────

def aux_transform():
    """
    pipeline transformacji obrazu do inferencji (bez augmentacji danych).
    
    Pipeline zawiera:
      - Konwersja do skali szarości (1 kanał)
      - Zmiana rozmiaru do IMAGE_SIZE x IMAGE_SIZE
      - Konwersja do tensora PyTorch
      - Normalizacja wartości pikseli (mean=0.5, std=0.5)
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((24, 24)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

def base_transform():
    """
    pipeline transformacji obrazu do inferencji (bez augmentacji danych).
    
    Pipeline zawiera:
      - Konwersja do skali szarości (1 kanał)
      - Zmiana rozmiaru do IMAGE_SIZE x IMAGE_SIZE
      - Konwersja do tensora PyTorch
      - Normalizacja wartości pikseli (mean=0.5, std=0.5)
    """
    return [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]

def get_inf_transform(args) -> transforms.Compose:
    """
    Zwraca pipeline transformacji obrazu do inferencji (bez augmentacji danych).
    
    Pipeline zawiera:
      - Konwersja do skali szarości (1 kanał)
      - Zmiana rozmiaru do IMAGE_SIZE x IMAGE_SIZE
      - Konwersja do tensora PyTorch
      - Normalizacja wartości pikseli (mean=0.5, std=0.5)
    
    Zwraca:
        transforms.Compose: Złożona transformacja gotowa do użycia
                            na obrazach PIL podczas predykcji.
    
    Przykład:
        >>> transform = get_transform()
        >>> tensor = transform(pil_image)
    """
    pack = [ResizeWithAspect(),
            RandomOtsu(Otsu()),]
    if getattr(args, "denoise", False):
        pack.append(trans_denoise_bil())

    pack.extend(base_transform())
    return transforms.Compose(pack)


def get_train_transform(args=None, denoise_prob: float = 0.3, max_padding: int = 20) -> transforms.Compose:
    """
    Zwraca pipeline transformacji obrazu do trenowania modelu (z augmentacją danych).

    Args:
        args: nieużywany, zachowany dla kompatybilności wstecznej
        denoise_prob: prawdopodobieństwo losowego odszumiania (0.0 = wyłączone, 1.0 = zawsze)
        max_padding: maksymalny padding w pikselach dla RandomPadding
    """
    pack = [
        transforms.RandomRotation(5),
        RandomDenoise(p=denoise_prob),
        RandomPadding(max_pad=max_padding),
        ResizeWithAspect(),
        RandomOtsu(Otsu())
    ]
    pack.extend(base_transform())
    return transforms.Compose(pack)


class RandomPadding:
    def __init__(self, max_pad=20):
        self.max_pad = max_pad

    def __call__(self, img):
        left = random.randint(0, self.max_pad)
        top = random.randint(0, self.max_pad)
        right = random.randint(0, self.max_pad)
        bottom = random.randint(0, self.max_pad)
        return F.pad(img, (left, top, right, bottom), fill=255)

class RandomDenoise:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return trans_denoise_bil()(img)
        return img


class ResizeWithAspect:
    def __init__(self, size=(32, 128), fill=255):
        self.h, self.w = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = min(self.w / w, self.h / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = F.resize(img, (new_h, new_w))

        pad_w = self.w - new_w
        pad_h = self.h - new_h

        left = pad_w // 2
        top = pad_h // 2

        return F.pad(img, (left, top, pad_w - left, pad_h - top), fill=self.fill)

class Otsu:
    def __call__(self, img):
        arr = np.array(img.convert("L"))
        _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(th)


class RandomOtsu:
    def __init__(self, otsu, p=0.1):
        self.otsu = otsu
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return self.otsu(img)
        return img
    
def preprocess_letter(img: np.ndarray) -> np.ndarray:
    """
    Przetwarza obraz pojedynczej litery przed klasyfikacją.
    
    Operacje wykonywane na obrazie:
      1. Dodaje biały padding (10px) wokół obrazu
      2. Tworzy kwadratowy canvas o rozmiarze max(wysokość, szerokość)
      3. Centruje literę na canvasie z białym tłem
      4. Skaluje wynikowy obraz do rozmiaru 24x24 pikseli
    
    Argumenty:
        img (np.ndarray): Obraz litery w skali szarości jako tablica numpy.
                          Oczekiwany format: (wysokość, szerokość), dtype uint8.
    
    Zwraca:
        np.ndarray: Przetworzony obraz litery o wymiarach 28x28 pikseli.
    
    Przykład:
        >>> letter = preprocess_letter(letter_array)
        >>> letter.shape
        (24, 24)
    """
    pad = 10
    img = np.pad(img, pad, mode='constant', constant_values=255)

    h, w = img.shape
    size = max(h, w)
    
    new_img = np.full((size, size), 255, dtype=img.dtype)
    
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    
    new_img[y_offset:y_offset+h, x_offset:x_offset+w] = img
    
    new_img = cv2.resize(new_img, (24, 24))
    return new_img

# ── Odszumianie ────────────────────────────────────────────────────────────────

def denoise_pil(img_pil: Image.Image) -> Image.Image:
    """
    Odszumia obraz PIL filtrem bilateralnym.

    Argumenty:
        img_pil (Image.Image): Obraz wejściowy w formacie PIL.
    
    Zwraca:
        Image.Image: Odszumiony obraz w formacie PIL.
    
    Przykład:
        >>> denoised = denoise_pil(img)
    """
    if cv2 is None:
        raise ModuleNotFoundError(
            "Brak modułu 'cv2'. Zainstaluj: pip install opencv-python "
            "(wymagane tylko dla opcji --denoise)."
        )
    rgb = img_pil.convert("RGB")
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    out = cv2.bilateralFilter(bgr, 9, 75, 75)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

class trans_denoise_bil:
    def __init__(self, d=9, sigma_color=75, sigma_space=75):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, img: Image.Image):
        rgb = img.convert("RGB")
        arr = np.array(rgb)

        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        out = cv2.bilateralFilter(bgr, self.d, self.sigma_color, self.sigma_space)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        return Image.fromarray(out)
# ── Ładowanie obrazu ───────────────────────────────────────────────────────────

def load_image(image_path: str, mode: str = "L") -> Image.Image:
    """
    Ładuje obraz z pliku i konwertuje do wybranego trybu kolorów.
    
    Obsługuje formaty PNG, JPG/JPEG oraz PDF (pierwsza strona).
    Dla plików PDF wykorzystuje bibliotekę pdf2image z DPI=300.
    
    Argumenty:
        image_path (str): Ścieżka do pliku obrazu (PNG, JPG, JPEG, PDF).
        mode (str, opcjonalnie): Tryb kolorów PIL. Domyślnie "L" (skala szarości).
            Dostępne tryby: "L" (grayscale), "RGB", "RGBA", "1" (binarny).
    
    Zwraca:
        Image.Image: Załadowany obraz w formacie PIL w wybranym trybie.
    
    Przykład:
        >>> img = load_image("letter.png", mode="L")
        >>> img_rgb = load_image("document.pdf", mode="RGB")
    """
    suffix = Path(image_path).suffix.lower()
    if suffix == ".pdf":
        # Prefer pdf2image when available (higher DPI control),
        # but on Windows it requires Poppler binaries in PATH.
        if convert_from_path is not None:
            try:
                img = convert_from_path(image_path, dpi=300)[0]
                return img.convert(mode)
            except Exception:
                # fall back to PyMuPDF if Poppler is missing or pdf2image fails
                pass

        if fitz is None:
            raise ModuleNotFoundError(
                "Obsługa PDF wymaga dodatkowych zależności.\n"
                "- Opcja A (najprostsza): zainstaluj PyMuPDF: pip install pymupdf\n"
                "- Opcja B: użyj pdf2image + zainstaluj Poppler i dodaj do PATH.\n"
                "Błąd wygląda na brak Popplera (pdfinfo/pdftoppm) w systemie."
            )

        # PyMuPDF fallback: render first page to a bitmap
        doc = fitz.open(image_path)
        try:
            page = doc.load_page(0)
            # ~300 DPI equivalent: scale 300/72
            zoom = 300.0 / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        finally:
            doc.close()
    else:
        img = Image.open(image_path)

    return img.convert(mode)


def load_and_optionally_denoise(image_path: str, args, mode: str = "L") -> Image.Image:
    """
    Ładuje obraz i opcjonalnie stosuje odszumianie na podstawie argumentów CLI.
    
    Funkcja łączy ładowanie obrazu z opcjonalnym preprocessingiem (odszumianiem).
    Jeśli args.denoise jest True, stosowany jest filtr bilateralny.
    
    Argumenty:
        image_path (str): Ścieżka do pliku obrazu (PNG, JPG, PDF).
        args: Obiekt argparse.Namespace z parametrami. Oczekiwane atrybuty:
            - denoise (bool): Czy stosować odszumianie
        mode (str, opcjonalnie): Tryb kolorów wyjściowych. Domyślnie "L".
            Używaj "RGB" dla klasyfikacji lub "L" dla segmentacji.
    
    Zwraca:
        Image.Image: Załadowany (i opcjonalnie odszumiony) obraz w formacie PIL.
    
    Przykład:
        >>> img = load_and_optionally_denoise("document.png", args, mode="L")
    """
    suffix = Path(image_path).suffix.lower()
    if suffix == ".pdf":
        img = convert_from_path(image_path, dpi=300)[0]
    else:
        img = Image.open(image_path)

    if getattr(args, "denoise", False):
        img = denoise_pil(img)

    return img.convert(mode)

# ── ładowanie datasetu ────────────────────────────────────────────────────

def load_all_datasets(root_dir):
    all_data = []

    for author in os.listdir(root_dir):
        author_path = os.path.join(root_dir, author)

        if not os.path.isdir(author_path):
            continue

        json_path = os.path.join(author_path, "boxes.jsonl")

        if not os.path.exists(json_path):
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                # KLUCZOWE: dodaj pełną ścieżkę do obrazu
                item["image_path"] = os.path.join(author_path, item["crop_file"])

                all_data.append(item)

    return all_data

# ── Zapis do folderu z datą ────────────────────────────────────────────────────

def get_today_folder() -> str:
    """
    Zwraca ścieżkę do folderu z dzisiejszą datą, tworząc go jeśli nie istnieje.
    
    Folder jest tworzony w bieżącym katalogu roboczym z nazwą w formacie
    YYYY-MM-DD (np. "2024-01-15").
    
    Argumenty:
        Brak argumentów.
    
    Zwraca:
        str: Ścieżka do folderu z dzisiejszą datą (np. "./2024-01-15").
    
    Przykład:
        >>> folder = get_today_folder()
        >>> print(folder)
        './2024-01-15'
    """
    folder = os.path.join("./outputs/", date.today().strftime("%Y-%m-%d"))
    os.makedirs(folder, exist_ok=True)
    return folder

def save_image_to_today_folder(image_path: str, info=None) -> str:
    """
    Kopiuje obraz do folderu z dzisiejszą datą z automatyczną numeracją.
    
    Obrazy są zapisywane jako pliki PNG z kolejnymi numerami (1.png, 2.png, ...).
    Funkcja automatycznie wykrywa istniejące pliki i nadaje następny numer.
    
    Argumenty:
        image_path (str): Ścieżka do obrazu źródłowego do skopiowania.
        info (callable, opcjonalnie): Funkcja do logowania. Domyślnie print.
    
    Zwraca:
        str: Ścieżka do zapisanego pliku (np. "./2024-01-15/3.png").
    
    Efekty uboczne:
        - Tworzy folder z dzisiejszą datą (jeśli nie istnieje)
        - Zapisuje obraz jako plik PNG
        - Wypisuje komunikat o zapisie na konsolę
    
    Przykład:
        >>> path = save_image_to_today_folder("input/letter.jpg")
        Zapisano zdjęcie jako: ./2024-01-15/1.png
    """
    if info is None:
        info = print
    
    folder = get_today_folder()
    existing = [
        int(os.path.splitext(f)[0])
        for f in os.listdir(folder)
        if f.endswith(".png") and os.path.splitext(f)[0].isdigit()
    ]
    dest = os.path.join(folder, f"{max(existing, default=0) + 1}.png")
    load_image(image_path, mode="L").save(dest, format="PNG")
    info(f"Zapisano zdjęcie jako: {dest}")
    return dest


def save_image_to_temp_folder(image: Any, order: str) -> str:
    """
    Zapisuje obraz do folderu tymczasowego z timestampem w nazwie.
    
    Funkcja tworzy folder ./temp/ (jeśli nie istnieje) i zapisuje obraz
    z nazwą zawierającą timestamp i podany suffix (order).
    
    Argumenty:
        image: Obraz do zapisania. Może być:
            - PIL.Image.Image: Obraz PIL
            - np.ndarray: Tablica numpy (zostanie skonwertowana do PIL)
        order (str): Suffix dodawany do nazwy pliku przed rozszerzeniem.
                     Używany do oznaczenia kolejności lub typu obrazu.
    
    Zwraca:
        str: Ścieżka do zapisanego pliku (np. "./temp/20240115_143022_1.png").
    
    Efekty uboczne:
        - Tworzy folder ./temp/ jeśli nie istnieje
        - Zapisuje obraz jako plik PNG
    
    Przykład:
        >>> path = save_image_to_temp_folder(pil_image, "_letter1")
        >>> path = save_image_to_temp_folder(numpy_array, "_segment")
    """
    os.makedirs("./temp", exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + order + ".png"
    filepath = os.path.join("./temp", filename)

    if isinstance(image, Image.Image):
        image.save(filepath)
    else:
        # assume numpy array
        img = Image.fromarray(image)
        img.save(filepath)

    return filepath

# ── Ładowanie modelu ───────────────────────────────────────────────────────────

ACTIVE_CHARS = list(CHARS)


def get_active_chars() -> list[str]:
    """Zwraca aktualne mapowanie indeks->znak używane przez model."""
    return list(ACTIVE_CHARS)


def _set_active_chars(checkpoint: object | None = None) -> None:
    """Ustawia mapowanie indeks->znak na podstawie checkpointa lub domyślnej konfiguracji."""
    global ACTIVE_CHARS


    ACTIVE_CHARS = list(CHARS)


def _label_for_idx(idx: int) -> str:
    if 0 <= idx < len(ACTIVE_CHARS):
        return ACTIVE_CHARS[idx]
    return f"<UNK:{idx}>"

#print(matplotlib.get_backend())

def load_model(model_path: str = MODEL_PATH, device: torch.device = None, info=None, model_type = 1) -> nn.Module:
    if info is None:
        info = print

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(model_path):
        info(f"Wczytywanie modelu z {model_path}...")

        checkpoint = torch.load(model_path, map_location=device)

        _set_active_chars(checkpoint)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            info("Wczytano checkpoint")
        else:
            state_dict = checkpoint
            info("Wczytano state_dict")
        if(model_type == 1):
            model = MainModel(num_classes=len(CHARS) + 1)
        else:
            model = AuxModel(num_classes=len(AuxCHARS))
        model.load_state_dict(state_dict)
        info("Model wczytany!")

    else:
        info(f"UWAGA: Nie znaleziono modelu {model_path}")
        if(model_type == 1):
            model = MainModel(num_classes=len(CHARS) + 1)
        else:
            model = AuxModel(num_classes=len(AuxCHARS))

    model.to(device)
    model.eval()

    return model


#------Zarządzanie transkrypcjami------

def load_transcription(path):
    mapping = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue

            key, text = parts

            # usuń rozszerzenie jeśli jest
            key = key.replace(".png", "")

            mapping[key] = text

    return mapping

#------Zarządzanie modelami------------
def list_models(models_dir: str, version: str | None = None):
    models = [
        name for name in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, name))
    ]

    if version:
        models = [
            m for m in models
            if m.startswith(f"v{version}")
        ]

    models.sort()
    return models

def select_version():
    version = input("Wybierz wersję (ENTER = wszystkie): ").strip()
    return version if version else None

def select_models(models):
    print("\nDostępne modele:")
    for i, m in enumerate(models):
        print(f"[{i}] {m}")

    default_idx = int(input("\nWybierz DEFAULT model (index): "))
    selected = input("Wybierz ensemble (np. 0,1,2) lub ENTER = wszystkie: ")

    if selected.strip() == "":
        ensemble = models
    else:
        ensemble = [models[int(i)] for i in selected.split(",")]

    default_model = models[default_idx]

    return default_model, ensemble



def load_models(models_dir, selected_models, device, info):
    loaded = {}

    for m in selected_models:
        path = os.path.join(models_dir, m, "model.pth")
        info(f"Ładowanie modelu: {m}")
        loaded[m] = load_model(path, device, info)

    return loaded

from collections import Counter

def aggregate(results, default_model):
    votes = Counter()
    confidence_sum = {}

    for model_name, res in results.items():
        text = res["text"]

        conf = res["confidence"]
        if hasattr(conf, "item"):  # torch / numpy
            conf = conf.item()

        votes[text] += 1
        confidence_sum[text] = confidence_sum.get(text, 0) + conf

    top_text, top_count = votes.most_common(1)[0]

    if top_count >= 2:
        return top_text

    return results[default_model]["text"]

def generate_model_ensembles(models, min_size=3, max_size=5, mode=1):
    if max_size is None:
        max_size = len(models)

    max_size = min(max_size, len(models))

    if mode == 1 :
        for r in range(min_size, max_size + 1):
            for combo in itertools.combinations(models, r):
                yield combo
    elif mode == 2 :
        for r in range(min_size, max_size + 1):
            for combo in itertools.permutations(models, r):
                yield combo

#---cache----------
def parse_version(name: str):
    m = VERSION_RE.match(name)
    if not m:
        return None
    major = int(m.group(1))
    minor = int(m.group(2) or 0)
    return (major, minor)


def version_str(v):
    major, minor = v
    return f"v{major}" if minor == 0 else f"v{major}.{minor}"


def resolve_cache_path(models_dir="./models", cache_dir="./cache"):
    versions = []

    for d in os.listdir(models_dir):
        full = os.path.join(models_dir, d)
        if os.path.isdir(full):
            v = parse_version(d)
            if v:
                versions.append(v)

    if not versions:
        raise ValueError("Brak poprawnych wersji w ./models")

    versions.sort()

    v_min = versions[0]
    v_max = versions[-1]

    base = f"{version_str(v_min)}_{version_str(v_max)}.json"
    path = os.path.join(cache_dir, base)

    if not os.path.exists(path):
        return path

    i = 1
    while True:
        suffix = f"_a{i:02d}"
        new_path = os.path.join(cache_dir, base.replace(".json", f"{suffix}.json"))
        if not os.path.exists(new_path):
            return new_path
        i += 1




def load_cache(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist() if obj.ndim > 0 else obj.item()

    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]

    return obj


def save_results_csv(rows, path="results.csv"):
    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "file",
                "ensemble",
                "text",
                "confidence",
                "accuracy"
            ]
        )

        if not file_exists:
            writer.writeheader()

        for r in rows:

            writer.writerow({
                "key": r["key"],
                "file": r["file"],
                "ensemble": r["ensemble"],
                "text": r["text"],
                "confidence": r["confidence"],
                "accuracy": r["accuracy"]
            })
#--- Autokorekta------
def DictCorrect(
    text: str,
    threshold: float = 0.8,
) -> str:
    """
    Szuka najbardziej podobnego słowa w słowniku.
    
    Jeśli podobieństwo >= threshold:
        zwraca słowo ze słownika
    W przeciwnym razie:
        zwraca oryginalny tekst
    """
    dictionary = load_dictionary()
    text = text.strip()

    if not text:
        return text

    best_match = None
    best_score = 0.0

    text_lower = text.lower()

    for word in dictionary:
        score = SequenceMatcher(
            None,
            text_lower,
            word.lower()
        ).ratio()

        if score > best_score:
            best_score = score
            best_match = word

    if best_score >= threshold:
        return best_match

    return text
