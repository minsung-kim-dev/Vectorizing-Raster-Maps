# short script to convert coco data to grayscale
import argparse
import shutil
from pathlib import Path

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path, help="COCOfolder")
ap.add_argument("dst", type=Path, help="output folder")
a = ap.parse_args()

shutil.copytree(a.src / "annotations", a.dst / "annotations", dirs_exist_ok=True)
for split in sorted(d for d in a.src.iterdir() if d.is_dir() and d.name != "annotations"):
    out = a.dst / split.name
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(split.glob("*.png"))
    for f in files:
        Image.open(f).convert("L").convert("RGB").save(out / f.name)
    print(f"{split.name}: {len(files)} images")
