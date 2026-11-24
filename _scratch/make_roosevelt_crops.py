"""Grid of adjacent crops around a point, rendered like the training crops, plus one stitched image.

Writes <out>/boxes.csv, <out>/crops/<name>_r<row>c<col>.{png,geojson.gz} (row 1 = north, col 1 = west)
and <out>/<name>_<cols>x<rows>.png.

Usage (from the repo root, in the raster2seq env):
  python _scratch/make_grid_crops.py --lat 40.755265 --lon -73.955766 --cols 2 --rows 3 \
      --name roosevelt --out _scratch/roosevelt_grid
"""
import argparse
import math
import os
import subprocess
import sys

from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--lat", type=float, default=40.755967)
p.add_argument("--lon", type=float, default=-73.955262)
p.add_argument("--cols", type=int, default=2)
p.add_argument("--rows", type=int, default=3)
p.add_argument("--side", type=float, default=125, help="crop side, meters (training crops: 125)")
p.add_argument("--name", default="roosevelt")
p.add_argument("--out", default="_scratch/roosevelt_grid")
p.add_argument("--gpkg", default="data/newyork_omt.gpkg")
a = p.parse_args()

# 1. boxes: same meters -> degrees approximation as data_preprocess/make_boxes.py
m_lat, m_lon = 111_320, 111_320 * math.cos(math.radians(a.lat))
dx, dy = a.side / m_lon, a.side / m_lat
west, north = a.lon - a.cols * dx / 2, a.lat + a.rows * dy / 2
os.makedirs(a.out, exist_ok=True)
boxes = os.path.join(a.out, "boxes.csv")
with open(boxes, "w") as f:
    f.write("name,min_lon,min_lat,max_lon,max_lat,width_m,height_m\n")
    for r in range(a.rows):
        for c in range(a.cols):
            x0, y1 = west + c * dx, north - r * dy
            f.write(f"{a.name}_r{r + 1}c{c + 1},{x0:.7f},{y1 - dy:.7f},{x0 + dx:.7f},{y1:.7f},{a.side:g},{a.side:g}\n")

# 2. render with the training pipeline
crops = os.path.join(a.out, "crops")
subprocess.run([sys.executable, "data_preprocess/osm_crops.py", "--gpkg", a.gpkg, "--boxes", boxes,
                "--zoom", "18", "--out", crops], check=True)

# 3. stitch (crops can differ by a pixel or two, so place each by the actual sizes)
ims = [[Image.open(os.path.join(crops, f"{a.name}_r{r}c{c}.png")).convert("RGB") for c in range(1, a.cols + 1)]
       for r in range(1, a.rows + 1)]
widths, heights = [im.width for im in ims[0]], [row[0].height for row in ims]
out = Image.new("RGB", (sum(widths), sum(heights)), "white")
for r, row in enumerate(ims):
    for c, im in enumerate(row):
        out.paste(im, (sum(widths[:c]), sum(heights[:r])))
path = os.path.join(a.out, f"{a.name}_{a.cols}x{a.rows}.png")
out.save(path)
print(f"wrote {path} {out.size}")
