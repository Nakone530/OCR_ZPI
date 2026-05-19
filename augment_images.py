import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ocr.augment import augment_directory


def main():
    parser = argparse.ArgumentParser(description="Augment images in a folder")
    parser.add_argument("--input-dir", required=True, help="Path to input image file or folder with images")
    parser.add_argument("--output-dir", required=False, help="Folder to save augmented images (defaults next to input)")
    parser.add_argument("--variants", type=int, default=5, help="Augmented variants per image")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of images to augment (0-1)")
    parser.add_argument("--level", choices=["low", "medium", "high"], help="Preset augmentation level (low/medium/high)")
    parser.add_argument("--start-idx", type=int, default=0, help="Start from word_NNN.png index when using ttData folder")
    parser.add_argument("--recursive", action="store_true", help="Recursively process subfolders")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker threads when using --recursive")
    parser.add_argument("--inplace", action="store_true", help="Save augmented files into the same folder as the input (in-place)")
    parser.add_argument("--mirror", action="store_true", help="Also copy augmented images into an `augmented` subfolder beside the original files")
    parser.add_argument("--crop-prob", type=float, default=0.5)
    parser.add_argument("--extend-prob", type=float, default=0.5)
    parser.add_argument("--noise-prob", type=float, default=0.5)
    parser.add_argument("--noise-sigma", type=float, default=10.0)
    parser.add_argument("--min-crop-scale", type=float, default=0.8, help="Minimum crop scale (0-1, higher = less aggressive)")
    parser.add_argument("--sp-prob", type=float, default=0.0, help="Salt-and-pepper noise probability")
    parser.add_argument("--sp-amount", type=float, default=0.005, help="Salt-and-pepper amount")
    parser.add_argument("--blur-prob", type=float, default=0.0, help="Gaussian blur probability")
    parser.add_argument("--blur-radius", type=float, default=1.0, help="Gaussian blur radius")
    args = parser.parse_args()
    # presets for three quick levels
    presets = {
        "low": {
            "variants": 2,
            "fraction": 0.3,
            "noise_prob": 0.2,
            "noise_sigma": 6.0,
            "sp_prob": 0.0,
            "sp_amount": 0.003,
            "blur_prob": 0.1,
            "blur_radius": 0.8,
            "min_crop_scale": 0.85,
        },
        "medium": {
            "variants": 4,
            "fraction": 0.6,
            "noise_prob": 0.45,
            "noise_sigma": 10.0,
            "sp_prob": 0.03,
            "sp_amount": 0.007,
            "blur_prob": 0.25,
            "blur_radius": 1.0,
            "min_crop_scale": 0.80,
        },
        "high": {
            "variants": 8,
            "fraction": 1.0,
            "noise_prob": 0.6,
            "noise_sigma": 14.0,
            "sp_prob": 0.08,
            "sp_amount": 0.012,
            "blur_prob": 0.45,
            "blur_radius": 1.6,
            "min_crop_scale": 0.75,
        },
    }

    # If a level is selected, apply its values for any args that were not explicitly provided.
    provided = set(sys.argv[1:])
    if args.level:
        p = presets[args.level]
        # only set values if their flag was not present in command line
        if "--variants" not in provided:
            args.variants = p["variants"]
        if "--fraction" not in provided:
            args.fraction = p["fraction"]
        if "--noise-prob" not in provided:
            args.noise_prob = p["noise_prob"]
        if "--noise-sigma" not in provided:
            args.noise_sigma = p["noise_sigma"]
        if "--sp-prob" not in provided:
            args.sp_prob = p["sp_prob"]
        if "--sp-amount" not in provided:
            args.sp_amount = p["sp_amount"]
        if "--blur-prob" not in provided:
            args.blur_prob = p["blur_prob"]
        if "--blur-radius" not in provided:
            args.blur_radius = p["blur_radius"]
        if "--min-crop-scale" not in provided:
            args.min_crop_scale = p.get("min_crop_scale", 0.8)

    # If output dir wasn't provided, choose a sensible default next to input
    if not args.output_dir:
        # when running recursively, default to placing augmented subfolders inside the input root
        if args.recursive:
            args.output_dir = args.input_dir
        else:
            inp = args.input_dir
            if os.path.isfile(inp):
                base = os.path.splitext(os.path.basename(inp))[0]
                parent = os.path.dirname(inp) or "."
                args.output_dir = os.path.join(parent, f"{base}_augmented")
            else:
                # directory
                args.output_dir = f"{args.input_dir.rstrip(os.sep)}_augmented"
    
    # Check if input path exists before creating output directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input path does not exist: {args.input_dir}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If recursive, find subfolders to process
    tasks = []
    results = {}
    if args.recursive and os.path.isdir(args.input_dir):
        base_root = os.path.abspath(args.input_dir)
        dirs_to_process = []
        for root, dirs, files in os.walk(args.input_dir):
            # skip hidden dirs
            if os.path.basename(root).startswith('.'):
                continue
            # consider directory if it contains word_*.png/jpg or boxes.jsonl
            has_images = any(f.lower().startswith('word_') and f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files)
            has_json = 'boxes.jsonl' in files
            if has_images or has_json:
                dirs_to_process.append(root)

        # process directories in parallel
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            future_to_dir = {}
            for d in dirs_to_process:
                # determine output folder for this subdir: mirror inside args.output_dir if output_dir is a dir
                if args.inplace:
                    out_sub = d
                else:
                    if os.path.isdir(args.output_dir):
                        rel = os.path.relpath(d, args.input_dir)
                        out_sub = os.path.join(args.output_dir, rel)
                    else:
                        out_sub = f"{d.rstrip(os.sep)}_augmented"
                
                # if mirror is requested, append _augmented suffix to output folder
                if args.mirror and not args.inplace:
                    out_sub = f"{out_sub}_augmented"

                future = ex.submit(
                    augment_directory,
                    d,
                    out_sub,
                    variants=args.variants,
                    fraction=args.fraction,
                    start_idx=args.start_idx,
                    crop_prob=args.crop_prob,
                    extend_prob=args.extend_prob,
                    noise_prob=args.noise_prob,
                    noise_sigma=args.noise_sigma,
                    sp_prob=args.sp_prob,
                    sp_amount=args.sp_amount,
                    blur_prob=args.blur_prob,
                    blur_radius=args.blur_radius,
                    min_crop_scale=args.min_crop_scale,
                )
                future_to_dir[future] = (d, out_sub)

            for fut in as_completed(future_to_dir):
                d, out_sub = future_to_dir[fut]
                try:
                    res = fut.result()
                    results.update(res)
                    created = sum(len(v[0]) if isinstance(v, tuple) else len(v) for v in res.values())
                    print(f"Processed {d} -> {out_sub}: created {created} images")
                except Exception as e:
                    print(f"Error processing {d}: {e}")
    else:
        # single-folder or single-file processing
        output_dir = args.output_dir
        if args.mirror:
            output_dir = f"{output_dir}_augmented"
        
        results = augment_directory(
            args.input_dir,
            output_dir,
            variants=args.variants,
            fraction=args.fraction,
            start_idx=args.start_idx,
            crop_prob=args.crop_prob,
            extend_prob=args.extend_prob,
            noise_prob=args.noise_prob,
            noise_sigma=args.noise_sigma,
            sp_prob=args.sp_prob,
            sp_amount=args.sp_amount,
            blur_prob=args.blur_prob,
            blur_radius=args.blur_radius,
            min_crop_scale=args.min_crop_scale,
        )

    total = sum(len(v[0]) if isinstance(v, tuple) else len(v) for v in results.values())
    print(f"Augmented {len(results)} files, created {total} images")


if __name__ == "__main__":
    main()
