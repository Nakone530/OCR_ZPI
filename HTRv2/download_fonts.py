import urllib.request
import os

FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")
os.makedirs(FONTS_DIR, exist_ok=True)

FONTS = {
    "AmaticSC": "https://github.com/google/fonts/raw/main/ofl/amaticsc/AmaticSC-Regular.ttf",
    "Caveat": "https://github.com/google/fonts/raw/main/ofl/caveat/Caveat%5Bwght%5D.ttf",
    "DancingScript": "https://github.com/google/fonts/raw/main/ofl/dancingscript/DancingScript%5Bwght%5D.ttf",
    "Pacifico": "https://github.com/google/fonts/raw/main/ofl/pacifico/Pacifico-Regular.ttf",
    "IndieFlower": "https://github.com/google/fonts/raw/main/ofl/indieflower/IndieFlower-Regular.ttf",
    "PatrickHand": "https://github.com/google/fonts/raw/main/ofl/patrickhand/PatrickHand-Regular.ttf",
    "Kalam": "https://github.com/google/fonts/raw/main/ofl/kalam/Kalam-Regular.ttf",
    "ArchitectsDaughter": "https://github.com/google/fonts/raw/main/ofl/architectsdaughter/ArchitectsDaughter-Regular.ttf",
    "GochiHand": "https://github.com/google/fonts/raw/main/ofl/gochihand/GochiHand-Regular.ttf",
    "Mali": "https://github.com/google/fonts/raw/main/ofl/mali/Mali-Regular.ttf",
}

for name, url in FONTS.items():
    dest = os.path.join(FONTS_DIR, f"{name}.ttf")
    if os.path.exists(dest):
        print(f"{name}.ttf already exists")
        continue
    try:
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  OK -> {dest}")
    except Exception as e:
        print(f"  FAILED: {e}")

print("\nDone. Fonts in", FONTS_DIR)
