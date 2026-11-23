"""Stitch predict.py or eval.py outputs for a grid of crops (from make_grid_crops.py) back into one map.

Each output was drawn on the crop resized to fit --image_size and padded evenly (ResizeAndPad): cut the padding
off, resize back to the crop's size and paste it at the crop's spot in the grid. Shapes are moved the same way,
then georeferenced with boxes.csv the way osm_crops.py rendered the crops.

Writes to <out>:
- stitched_<kind>.png for each image kind found
- stitched_pred.json / .geojson: predicted shapes (needs --save_pred), in stitched-image pixels / lon-lat
- stitched_gt.json / .geojson: true shapes from --coco
GeoJSON: lines as LineStrings, the rest as Polygons.

eval.py names outputs by COCO image id, so pass --coco (the evaluated split's annotations); with predict.py
outputs --coco is optional and adds the true shapes. predict.py writes into a subfolder named after the
checkpoint's folder: pass that as --pred_dir. Run eval.py with
--disable_image_transform on a train split, otherwise its outputs are randomly flipped / rotated.

Usage:
  python _scratch/stitch_predictions.py --pred_dir eval_outputs/<job> --coco data/coco_roosevelt/annotations/train.json \\
      --crops data/coco_roosevelt/train --out _scratch/roosevelt_grid/stitched_eval
  python _scratch/stitch_predictions.py --pred_dir predictions/<job>/<checkpoint folder> --coco data/coco_roosevelt/annotations/train.json \\
      --crops data/coco_roosevelt/train --out _scratch/roosevelt_grid/stitched_predict

Coordinate systems used below:
- model-input pixels: the --image_size square the model saw (crop scaled to fit, then padded); predictions are here
- crop pixels: the original crop PNG (e.g. 554 x 556); COCO annotations are here
- stitched pixels: crop pixels + the crop's offset in the stitched image
- world pixels: the Web Mercator map at zoom ZOOM, 256 * 2**ZOOM pixels across the whole world; the renderer drew
  each crop from one rectangle of these (its "window"), at SCALE crop pixels per world pixel
- lon/lat: world pixels converted with the Web Mercator formula
"""
import argparse
import csv
import json
import math
import os
import re
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))
from render_carto import pixel_window  # noqa: E402  (the same function osm_crops.py rendered with)

# image outputs to look for: <id>_<kind>.png (predict.py writes the first and third, eval.py all of them)
KINDS = ["pred_room_map", "gt_room_map", "pred_floorplan", "gt_floorplan", "pred_floorplan_sem", "gt_floorplan_sem"]
ZOOM, SCALE = 18, 2  # osm_crops.py --zoom / --scale: render zoom, and crop pixels per world pixel

p = argparse.ArgumentParser()
p.add_argument("--pred_dir", required=True, help="predict.py / eval.py output folder")
p.add_argument("--crops", required=True, help="folder with the original crop PNGs <name>_r<row>c<col>.png")
p.add_argument("--out", required=True)
p.add_argument("--coco", help="eval.py outputs: the evaluated split's COCO annotations")
p.add_argument("--image_size", type=int, default=256, help="--image_size of predict.py / eval.py")
p.add_argument("--boxes", default="_scratch/roosevelt_grid/boxes.csv", help="crop bounds (make_grid_crops.py)")
p.add_argument("--categories", default="data/coco_osm_fixed/annotations/train.json",
               help="COCO annotations with the label names and which labels are lines")
a = p.parse_args()

# predict.py names outputs by crop name, eval.py by COCO image id: map crop name -> image id to find either
coco = json.load(open(a.coco)) if a.coco else None
ids = {im["file_name"].rsplit(".", 1)[0]: im["id"] for im in coco["images"]} if coco else {}

# label names, and which labels are lines: any label with line annotations (same rule as eval.py;
# a buffered dataset keeps the original geometry type in source_geom_type)
cats = json.load(open(a.categories))
names = {c["id"]: c["name"] for c in cats["categories"]}
line_ids = {x["category_id"] for x in cats["annotations"] if x.get("source_geom_type", x.get("geom_type")) == "line"}

# each crop's lon/lat bounds, as given to osm_crops.py
bounds = {r["name"]: [float(r[k]) for k in ("min_lon", "min_lat", "max_lon", "max_lat")] for r in csv.DictReader(open(a.boxes))}

# ---- 1. the crops: everything needed to move between the coordinate systems, computed once per crop ----
crops = []
for f in os.listdir(a.crops):
    m = re.fullmatch(r"(.+_r(\d+)c(\d+))\.png", f)  # e.g. roosevelt_r2c1.png: row 2 (from the north), column 1 (from the west)
    if m:
        w, h = Image.open(os.path.join(a.crops, f)).size
        crops.append({"name": m[1], "row": int(m[2]), "col": int(m[3]), "w": w, "h": h})

# crops can differ by a pixel or two (rounding when rendering), so use each column's width and each row's height
widths = {c["col"]: c["w"] for c in crops}
heights = {c["row"]: c["h"] for c in crops}
for c in crops:
    # offset in the stitched image: the columns to the left and the rows above
    c["x0"] = sum(widths[k] for k in widths if k < c["col"])
    c["y0"] = sum(heights[k] for k in heights if k < c["row"])
    # how the model saw the crop (datasets/transforms.py ResizeAndPad): scaled by s to fit the square,
    # keeping its proportions, to nw x nh pixels, then centered with `left` / `top` pixels of padding
    c["s"] = s = min(a.image_size / c["w"], a.image_size / c["h"])
    c["nw"], c["nh"] = int(c["w"] * s), int(c["h"] * s)
    c["left"], c["top"] = (a.image_size - c["nw"]) // 2, (a.image_size - c["nh"]) // 2
    # the names its outputs may be saved under: predict.py's crop name, eval.py's image id (jsons: zero-padded)
    c["keys"] = [c["name"]] + ([str(ids[c["name"]])] if c["name"] in ids else [])
    # its window of world pixels (x0, y0, x1, y1): the crop's bounds on the zoom-ZOOM world map, rounded outward
    # to whole pixels; the renderer drew exactly this rectangle, so it (not the bounds) maps crop pixels to places
    c["win"] = pixel_window(bounds[c["name"]], ZOOM)
size = (sum(widths.values()), sum(heights.values()))

# ---- 2. images: undo the resize-and-pad for each crop's output, then paste it into place ----
os.makedirs(a.out, exist_ok=True)
for kind in KINDS:
    out, found = Image.new("RGB", size, "white"), 0
    for c in crops:
        path = next((q for q in (os.path.join(a.pred_dir, f"{k}_{kind}.png") for k in c["keys"]) if os.path.exists(q)), None)
        if path:
            im = Image.open(path).convert("RGB")
            # some outputs are drawn larger than the model input (e.g. 1024 px for a 256 px input): scale the box
            k = im.width / a.image_size
            # the crop's area inside the padding, in this image's pixels
            box = (c["left"] * k, c["top"] * k, (c["left"] + c["nw"]) * k, (c["top"] + c["nh"]) * k)
            # NEAREST keeps the flat colors and thin lines crisp
            out.paste(im.crop(box).resize((c["w"], c["h"]), Image.NEAREST), (c["x0"], c["y0"]))
            found += 1
    if found:  # skip kinds this script didn't write
        out.save(os.path.join(a.out, f"stitched_{kind}.png"))
        print(f"stitched_{kind}.png: {found}/{len(crops)} crops")


# ---- 3. shapes: crop pixels -> stitched pixels (JSON) and lon/lat (GeoJSON) ----
def to_lonlat(u, v, win):
    """Crop pixel (u, v) -> [lon, lat]. The crop shows world pixels win[0].. and win[1].., each drawn as SCALE x SCALE
    crop pixels, so (u, v) is world pixel (win[0] + u / SCALE, win[1] + v / SCALE). Then the inverse of
    render_carto.lonlat_to_px: x is linear in longitude; y follows the Mercator projection (top of the map = north)."""
    n = 256 * 2 ** ZOOM  # world map width / height in pixels at this zoom
    x, y = win[0] + u / SCALE, win[1] + v / SCALE
    return [x / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))]


def save(shapes, tag):
    """shapes: [(crop, points in crop pixels, category_id)] -> stitched_<tag>.json (stitched pixels, for drawing on the
    stitched images) and stitched_<tag>.geojson (lon/lat, EPSG:4326, for GIS)."""
    pixels, features = [], []
    for c, uv, cat in shapes:
        pixels.append({"id": len(pixels), "crop": c["name"], "category_id": cat,
                       "segmentation": [[u + c["x0"], v + c["y0"]] for u, v in uv]})
        ll = [to_lonlat(u, v, c["win"]) for u, v in uv]
        # line labels (and anything with under 3 points) are open lines; the rest are polygons, whose ring
        # GeoJSON requires to be closed by repeating the first point
        line = cat in line_ids or len(ll) < 3
        geom = {"type": "LineString", "coordinates": ll} if line else {"type": "Polygon", "coordinates": [ll + ll[:1]]}
        features.append({"type": "Feature", "geometry": geom,
                         "properties": {"label": names.get(cat, str(cat)), "category_id": cat, "crop": c["name"]}})
    if pixels:
        json.dump(pixels, open(os.path.join(a.out, f"stitched_{tag}.json"), "w"))
        json.dump({"type": "FeatureCollection", "features": features}, open(os.path.join(a.out, f"stitched_{tag}.geojson"), "w"))
    print(f"stitched_{tag}: {len(pixels)} shapes")


# predictions (jsons/<name or zero-padded id>.json, from --save_pred) are in model-input pixels:
# remove the padding, then undo the scaling, to get crop pixels
pred = []
for c in crops:
    path = next((q for q in (os.path.join(a.pred_dir, "jsons", f"{k.zfill(5)}.json") for k in c["keys"]) if os.path.exists(q)), None)
    if path:
        for d in json.load(open(path)):
            uv = [((x - c["left"]) / c["s"], (y - c["top"]) / c["s"]) for x, y in d["segmentation"]]
            pred.append((c, uv, d["category_id"]))
save(pred, "pred")

# true shapes: COCO annotations are already in crop pixels, as a flat [x1, y1, x2, y2, ...] list
if coco:
    by_id = {ids[c["name"]]: c for c in crops}
    save([(by_id[x["image_id"]], list(zip(x["segmentation"][0][0::2], x["segmentation"][0][1::2])), x["category_id"])
          for x in coco["annotations"] if x["image_id"] in by_id], "gt")
