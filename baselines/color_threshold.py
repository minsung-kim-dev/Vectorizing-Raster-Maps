"""Colour-threshold baseline: vectorize map crops by their known OSM Carto colours, no learning.

The crops are drawn with a fixed palette (utils/render_carto.py), so many labels can be found by colour:
  1. Pixels -> mask: each pixel goes to its closest palette colour (background land included),
     if that colour is within --tol.
  2. Polygons: cv2.findContours traces each region's outline, cv2.approxPolyDP keeps the corners.
     Blue areas smaller than --pool_max px^2 become "pool", larger ones "water".
  3. Lines: the road / rail / path mask is thinned to 1-pixel centrelines (skeletonize), cut at
     junctions, and each piece is traced end to end and simplified.

Predictions are written to predictions.json and scored by evaluation/score.py (the shared scorer
every method uses): per-label precision / recall / F1 / pixel IoU, recall per geometry bin, macro
averages and bootstrap intervals. Lines are scored as polygons widened --line_buffer_px px on each
side (default 2, like create_coco_osm.py --line_buffer_px); 0 scores centrelines instead.

Usage:
    pip install opencv-python-headless scikit-image shapely numpy pillow matplotlib
    python baselines/color_threshold.py                                # mini test set (12 Queens crops)
    python baselines/color_threshold.py --data data/coco_osm --split test --plots 10
    python baselines/color_threshold.py --data data/coco_osm_buffered --line_buffer_px 2

Writes to --out (default results/color_threshold/<split>/):
    predictions.json  the baseline's shapes, in the shared predictions format
    per_label.csv, per_bin.csv, per_image.csv, summary.json   from evaluation/score.py
    plots/*.png       map | ground truth | prediction, for the first --plots crops
CPU only; a GPU is not used.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluation.score import load_coco, save_predictions, score  # noqa: E402


# label -> fill colours used by utils/render_carto.py
POLY_COLORS = {
    "building": ["#d9d0c9"], "grass": ["#cdebb0"], "park": ["#c8facc"], "woods": ["#add19e"],
    "playground": ["#dffce2"], "sports_field": ["#aae0cb"], "cemetery": ["#aacbaf"], "school": ["#ffffe5"],
    "residential": ["#e0dfdf"], "commercial": ["#f2dad9"], "retail": ["#ffd6d1"], "industrial": ["#ebdbe8"],
    "bare_ground": ["#f5e9c6", "#eee5dc"], "plaza": ["#dddde8"],
}
WATER_COLOR = "#aad3df"  # water and pool share it
LINE_COLORS = {"highway": ["#e892a2", "#f9b29c"], "street": ["#fcd6a4", "#f7fabf", "#ffffff"],
               "railway": ["#707070", "#787878"], "path": ["#fa8072"]}
LAND_COLOR = "#f2efe9"  # background; competes for pixels but is never predicted


# ---------------------------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------------------------

def rgb(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


PALETTE = sorted({h for hs in [*POLY_COLORS.values(), *LINE_COLORS.values(), [WATER_COLOR, LAND_COLOR]] for h in hs})


def nearest_color(arr, tol):
    """Index into PALETTE of each pixel's closest colour (max per-channel difference), -1 if none within tol.
    Taking the closest colour stops near-white land (e.g. #f7f5f1) from counting as school (#ffffe5)."""
    pal = np.array([rgb(h) for h in PALETTE], np.int16)
    d = np.abs(arr.astype(np.int16)[:, :, None, :] - pal[None, None]).max(-1)
    idx = d.argmin(-1)
    idx[np.take_along_axis(d, idx[..., None], -1)[..., 0] > tol] = -1
    return idx


def color_mask(idx, hexes):
    return np.isin(idx, [PALETTE.index(h) for h in hexes])


def mask_to_polygons(m, min_area=20, eps=1.5):
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # drop specks and anti-aliased fringes
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            pts = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            if len(pts) >= 3:
                out.append((area, pts.astype(float).tolist()))
    return out


def trace(pixels):
    """Order the pixels of a 1-pixel-wide path from one end to the other."""
    nbrs = lambda p: [(p[0] + dx, p[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                      if (dx or dy) and (p[0] + dx, p[1] + dy) in pixels]
    path = [next((p for p in pixels if len(nbrs(p)) == 1), next(iter(pixels)))]
    seen = {path[0]}
    while nxt := [q for q in nbrs(path[-1]) if q not in seen]:
        path.append(nxt[0])
        seen.add(nxt[0])
    return path


def mask_to_lines(m, min_len=8, eps=1.5):
    sk = skeletonize(m)
    n_nbrs = cv2.filter2D(sk.astype(np.float32), -1, np.ones((3, 3), np.float32)) - sk
    sk &= ~(n_nbrs > 2)  # cut at junctions -> simple paths
    n, comp = cv2.connectedComponents(sk.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        ys, xs = np.nonzero(comp == i)
        if len(xs) >= min_len:
            path = np.array(trace(set(zip(xs.tolist(), ys.tolist()))), np.int32).reshape(-1, 1, 2)
            out.append(cv2.approxPolyDP(path, eps, False).reshape(-1, 2).astype(float).tolist())
    return out


def predict(arr, tol=12, pool_max=3000.0, min_area=20.0):
    """Shapes [{"label", "points"}] found by colour in an RGB image array."""
    idx = nearest_color(arr, tol)
    shapes = []
    for label, hexes in POLY_COLORS.items():
        shapes += [{"label": label, "kind": "polygon", "points": pts}
                   for _, pts in mask_to_polygons(color_mask(idx, hexes), min_area)]
    for area, pts in mask_to_polygons(color_mask(idx, [WATER_COLOR]), min_area):
        shapes.append({"label": "pool" if area < pool_max else "water", "kind": "polygon", "points": pts})
    for label, hexes in LINE_COLORS.items():
        shapes += [{"label": label, "kind": "line", "points": pts} for pts in mask_to_lines(color_mask(idx, hexes))]
    return shapes


# ---------------------------------------------------------------------------------------------

def config():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--data", default=str(REPO / "data" / "coco_osm_mini"), help="COCO dataset root")
    a.add_argument("--split", default="test")
    a.add_argument("--out", default=None, help="output folder (default results/color_threshold/<split>)")
    a.add_argument("--limit", type=int, default=0, help="only the first N images (0 = all)")
    a.add_argument("--plots", type=int, default=5, help="save side-by-side plots for the first N images")
    a.add_argument("--tol", type=int, default=12, help="max per-channel colour difference (0-255)")
    a.add_argument("--pool_max", type=float, default=3000.0, help="blue areas below this (px^2) are pools")
    a.add_argument("--min_area", type=float, default=20.0, help="drop predicted polygons below this (px^2)")
    a.add_argument("--iou", type=float, default=0.5, help="IoU needed for a shape to count as matched")
    a.add_argument("--line_buffer_px", type=float, default=2.0,
                   help="score lines as polygons widened this many px on each side, like create_coco_osm.py's "
                        "--line_buffer_px; 0 = score lines as centrelines")
    a.add_argument("--line_width", type=float, default=6.0,
                   help="with --line_buffer_px 0: band width (px) centrelines are widened to for scoring")
    return a.parse_args()


def main(args):
    out = Path(args.out or REPO / "results" / "color_threshold" / args.split)
    out.mkdir(parents=True, exist_ok=True)
    images, _ = load_coco(args.data, args.split)
    if args.limit:
        images = images[:args.limit]
    preds = {}
    for k, info in enumerate(images):
        arr = np.array(Image.open(Path(args.data) / args.split / info["file_name"]).convert("RGB"))
        preds[info["file_name"]] = predict(arr, args.tol, args.pool_max, args.min_area)
        print(f"predicted [{k + 1}/{len(images)}] {info['file_name']}: {len(preds[info['file_name']])} shapes")
    save_predictions(out / "predictions.json", "colour threshold", preds)
    score(args.data, args.split, preds, out, "colour threshold", args.line_buffer_px, args.line_width,
          args.iou, args.plots, args.limit)


if __name__ == "__main__":
    main(config())
