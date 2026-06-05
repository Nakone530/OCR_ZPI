import urllib.request, os

fonts_dir = r"C:\Users\Ivan\Desktop\HTRv2\fonts"
added = []

urls = {
    "ShadowsIntoLight.ttf": "https://github.com/google/fonts/raw/main/ofl/shadowsintolight/ShadowsIntoLight.ttf",
    "PermanentMarker.ttf": "https://github.com/google/fonts/raw/main/ofl/permanentmarker/PermanentMarker-Regular.ttf",
    "GloriaHallelujah.ttf": "https://github.com/google/fonts/raw/main/ofl/gloriahallelujah/GloriaHallelujah-Regular.ttf",
    "Neucha.ttf": "https://github.com/google/fonts/raw/main/ofl/neucha/Neucha-Regular.ttf",
    "MarckScript.ttf": "https://github.com/google/fonts/raw/main/ofl/marckscript/MarckScript-Regular.ttf",
    "BadScript.ttf": "https://github.com/google/fonts/raw/main/ofl/badscript/BadScript-Regular.ttf",
    "Lobster.ttf": "https://github.com/google/fonts/raw/main/ofl/lobster/Lobster-Regular.ttf",
    "Satisfy.ttf": "https://github.com/google/fonts/raw/main/ofl/satisfy/Satisfy-Regular.ttf",
    "RougeScript.ttf": "https://github.com/google/fonts/raw/main/ofl/rougescript/RougeScript-Regular.ttf",
    "PatrickHandSC.ttf": "https://github.com/google/fonts/raw/main/ofl/patrickhandsc/PatrickHandSC-Regular.ttf",
}

for name, url in urls.items():
    dest = os.path.join(fonts_dir, name)
    if os.path.exists(dest):
        print(f"{name} already exists")
        continue
    try:
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  OK")
        added.append(name)
    except Exception as e:
        print(f"  FAILED: {e}")

print(f"\nAdded {len(added)} new fonts: {', '.join(added)}")
