"""
Build the training data index for Stage A.

Stage A reads a JSON list of {"image_file": <path>, "text": <caption>} entries. CtrlRegen
does not use captions -- the released weights were trained with every `text` field empty --
so any collection of images works. This script just walks a directory and writes the index.

Example
    python build_data_json.py --image_dir /path/to/images --output data.json
"""
import argparse
import json
import os

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image_dir", required=True, help="Directory to scan (recursively).")
    parser.add_argument("--output", required=True, help="Destination .json file.")
    parser.add_argument("--relative", action="store_true",
                        help="Store paths relative to --image_dir. Pass the same directory as "
                             "--data_root_path when training.")
    args = parser.parse_args()

    root = os.path.abspath(args.image_dir)
    entries = []
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if os.path.splitext(filename)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            path = os.path.join(dirpath, filename)
            entries.append({
                "image_file": os.path.relpath(path, root) if args.relative else path,
                "text": "",  # CtrlRegen does not use captions
            })

    if not entries:
        raise SystemExit(f"No images found under {root}")

    with open(args.output, "w") as handle:
        json.dump(entries, handle, indent=4)
    print(f"Wrote {args.output}: {len(entries):,} images")


if __name__ == "__main__":
    main()
