import os
import random
from typing import Optional
import json
import csv

import numpy as np
from PIL import Image, ImageOps, ImageFilter


def save_metadata_json(records: list, output_path: str):
    """Save augmentation metadata as JSON Lines format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_metadata_tsv(records: list, output_path: str):
    """Save augmentation metadata as TSV format."""
    if not records:
        return
    fieldnames = ["id", "crop", "file", "text"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k, "") for k in fieldnames})


def load_text_from_jsonl(image_path: str, jsonl_path: str) -> str:
    """Extract text from JSON Lines file for a given image (by crop_file)."""
    if not os.path.exists(jsonl_path):
        return ""
    
    crop_file = os.path.basename(image_path)
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                if record.get("crop_file") == crop_file:
                    return record.get("text", "")
    except Exception:
        pass
    return ""


def augment_from_jsonl(
    folder: str,
    out_dir: str,
    jsonl_file: str = "boxes.jsonl",
    start_idx: int = 0,
    variants: int = 5,
    fraction: float = 1.0,
    save_metadata: bool = True,
    **kwargs,
):
    """
    Augment word_*.png images from a folder, reading text from JSON Lines.
    
    Args:
        folder: Path to folder with word_*.png and boxes.jsonl
        out_dir: Output directory
        jsonl_file: Name of JSON Lines file (default: boxes.jsonl)
        start_idx: Start from word_NNN.png index (default: 0)
        variants: Augmented variants per image
        fraction: Fraction of images to augment (0-1)
        save_metadata: Whether to save JSON/TSV metadata files
        **kwargs: Other augment_image parameters
    
    Returns:
        Dictionary mapping image paths to (saved_files, metadata) tuples
    """
    jsonl_path = os.path.join(folder, jsonl_file)
    
    # Find all word_*.png files
    files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().startswith("word_") and f.lower().endswith(".png")
    ])
    
    # Apply start_idx and fraction
    files = files[start_idx:]
    k = max(1, int(len(files) * fraction))
    selected = files[:k]
    
    results = {}
    all_metadata = []
    
    os.makedirs(out_dir, exist_ok=True)
    
    for img_path in selected:
        # Load text from JSON
        text = load_text_from_jsonl(img_path, jsonl_path)
        saved, metadata = augment_image(
            img_path, out_dir, variants=variants, text=text, **kwargs
        )
        results[img_path] = (saved, metadata)
        all_metadata.extend(metadata)
    
    if save_metadata:
        save_metadata_json(all_metadata, os.path.join(out_dir, "_metadata.jsonl"))
        save_metadata_tsv(all_metadata, os.path.join(out_dir, "_metadata.tsv"))
    
    return results


def random_crop(img: Image.Image, min_scale: float = 0.6) -> Image.Image:
    w, h = img.size
    scale = random.uniform(min_scale, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w >= w or new_h >= h:
        return img.copy()
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    return img.crop((left, top, left + new_w, top + new_h))


def random_extend(img: Image.Image, max_scale: float = 1.25, background_color=(255, 255, 255)) -> Image.Image:
    w, h = img.size
    scale = random.uniform(1.0, max_scale)
    new_w, new_h = int(w * scale), int(h * scale)
    canvas = Image.new("RGB", (new_w, new_h), background_color)
    # place original image at a random offset inside the larger canvas
    max_x = new_w - w
    max_y = new_h - h
    offset_x = random.randint(0, max_x) if max_x > 0 else 0
    offset_y = random.randint(0, max_y) if max_y > 0 else 0
    canvas.paste(img, (offset_x, offset_y))
    return canvas


def add_gaussian_noise(img: Image.Image, sigma: float = 10.0) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr += noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_salt_and_pepper_noise(img: Image.Image, amount: float = 0.005) -> Image.Image:
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    num_salt = np.ceil(amount * h * w * 0.5).astype(int)
    num_pepper = np.ceil(amount * h * w * 0.5).astype(int)
    # Salt
    coords = [
        np.random.randint(0, i - 1, num_salt) for i in (h, w)
    ]
    arr[coords[0], coords[1]] = 255
    # Pepper
    coords = [
        np.random.randint(0, i - 1, num_pepper) for i in (h, w)
    ]
    arr[coords[0], coords[1]] = 0
    return Image.fromarray(arr)


def add_gaussian_blur(img: Image.Image, radius: float = 1.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


def augment_image(
    path: str,
    out_dir: str,
    variants: int = 5,
    crop_prob: float = 0.5,
    extend_prob: float = 0.5,
    noise_prob: float = 0.5,
    min_crop_scale: float = 0.8,
    max_extend_scale: float = 1.25,
    noise_sigma: float = 10.0,
    blur_prob: float = 0.0,
    blur_radius: float = 1.0,
    sp_prob: float = 0.0,
    sp_amount: float = 0.005,
    text: str = "",
):
    img = Image.open(path).convert("RGB")
    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    metadata = []
    for i in range(variants):
        out = img.copy()
        transforms = {"crop": False}
        # apply crop
        if random.random() < crop_prob:
            out = random_crop(out, min_scale=min_crop_scale)
            transforms["crop"] = True
        # apply extend
        if random.random() < extend_prob:
            out = random_extend(out, max_scale=max_extend_scale)
        # optionally add small random resize to simulate different framing
        if random.random() < 0.3:
            # random resize between 90% and 110%
            scale = random.uniform(0.9, 1.1)
            nw, nh = int(out.width * scale), int(out.height * scale)
            out = out.resize((max(1, nw), max(1, nh)), Image.LANCZOS)
        # optionally add gaussian noise
        if random.random() < noise_prob:
            out = add_gaussian_noise(out, sigma=noise_sigma)
        # optionally add salt-and-pepper noise
        if sp_prob > 0 and random.random() < sp_prob:
            out = add_salt_and_pepper_noise(out, amount=sp_amount)
        # optionally add blur
        if blur_prob > 0 and random.random() < blur_prob:
            out = add_gaussian_blur(out, radius=blur_radius)
        fname = f"{base}_aug_{i+1}.png"
        fpath = os.path.join(out_dir, fname)
        out.save(fpath)

        saved.append(fpath)
        # record metadata
        metadata.append({
            "id": f"{base}_aug_{i+1}",
            "crop": transforms.get("crop", False),
            "file": fpath,
            "text": text,
        })
    return saved, metadata


def augment_directory(
    in_dir: str,
    out_dir: str,
    variants: int = 5,
    fraction: float = 1.0,
    save_metadata: bool = True,
    start_idx: int = 0,
    **kwargs,
):
    # support passing a single file path or a directory
    if os.path.isfile(in_dir):
        os.makedirs(out_dir, exist_ok=True)
        saved, metadata = augment_image(in_dir, out_dir, variants=variants, **kwargs)
        if save_metadata:
            save_metadata_json(metadata, os.path.join(out_dir, "_metadata.jsonl"))
            save_metadata_tsv(metadata, os.path.join(out_dir, "_metadata.tsv"))
        return {in_dir: (saved, metadata)}

    # Check if this looks like a ttData folder with word_*.png and boxes.jsonl
    jsonl_path = os.path.join(in_dir, "boxes.jsonl")
    if os.path.exists(jsonl_path):
        # Use JSON-aware augmentation (supports both .png and .jpg)
        files = sorted([
            os.path.join(in_dir, f)
            for f in os.listdir(in_dir)
            if f.lower().startswith("word_") and f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        files = files[start_idx:]
        k = max(1, int(len(files) * fraction))
        selected = files[:k]
        
        results = {}
        all_metadata = []
        os.makedirs(out_dir, exist_ok=True)
        
        for img_path in selected:
            text = load_text_from_jsonl(img_path, jsonl_path)
            saved, metadata = augment_image(
                img_path,
                out_dir,
                variants=variants,
                text=text,
                **kwargs,
            )
            results[img_path] = (saved, metadata)
            all_metadata.extend(metadata)
        
        if save_metadata:
            save_metadata_json(all_metadata, os.path.join(out_dir, "_metadata.jsonl"))
            save_metadata_tsv(all_metadata, os.path.join(out_dir, "_metadata.tsv"))
        
        return results

    # Standard folder augmentation (image files)
    files = [
        os.path.join(in_dir, f)
        for f in os.listdir(in_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ]
    random.shuffle(files)
    k = max(1, int(len(files) * fraction))
    selected = files[:k]
    results = {}
    all_metadata = []
    for p in selected:
        saved, metadata = augment_image(p, out_dir, variants=variants, **kwargs)
        results[p] = (saved, metadata)
        all_metadata.extend(metadata)
    if save_metadata:
        save_metadata_json(all_metadata, os.path.join(out_dir, "_metadata.jsonl"))
        save_metadata_tsv(all_metadata, os.path.join(out_dir, "_metadata.tsv"))
    return results


if __name__ == "__main__":
    # simple local test helper
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", type=int, default=5)
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()
    augment_directory(args.input, args.output, variants=args.variants, fraction=args.fraction)
