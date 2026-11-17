"""Colour-threshold baseline: vectorize map crops by their known OSM Carto colours, no learning.

The crops are drawn with a fixed palette (utils/render_carto.py), so many labels can be found by colour:
  0. Labels: map text (dark glyphs and their white halo) is painted over with the nearest colour
     around it, so letters don't become railway shapes and halos don't become streets.
  1. Pixels -> mask: each pixel goes to its closest palette colour (background land included),
     if that colour is within --tol.
  2. Polygons: cv2.findContours traces each region's outline, cv2.approxPolyDP keeps the corners.
     Blue areas smaller than --pool_max px^2 become "pool", larger ones "water". Landuse areas
     (residential, school, ...) are rejoined across the roads drawn over them, as in OSM.
  3. Lines: dotted footways and dashed railways are widened until the dashes touch, then every
     road / rail / path mask is thinned to 1-pixel centrelines (skeletonize), cut at junctions,
     short spurs are dropped, and pieces that carry on straight through a junction are joined again,
     so a street crossing other streets comes out as one line, like the OSM way.

This file only predicts shapes. Scoring is done with Raster2Seq's own evaluator, so the baseline is
graded exactly like the model: run eval_baseline.py in the Raster2Seq fork, which imports predict()
from this file, e.g.
    python eval_baseline.py --dataset_root data/coco_osm_flushing5000 --vrm_repo ../Vectorizing-Raster-Maps \
        --num_bins 32 --line_radius 10 --per_label

Usage (predictions only):
    pip install opencv-python-headless scikit-image scipy numpy pillow
    python baselines/color_threshold.py --data data/coco_osm_flushing5000 --split test --limit 20
Writes <out>/predictions.json: {file_name: [{"label", "kind", "points"}, ...]} in full-crop pixel coordinates.
CPU only; a GPU is not used.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

REPO = Path(__file__).resolve().parents[1]


# label -> fill colours used by utils/render_carto.py
POLY_COLORS = {
    # #c8facc is landcover grass with subclass park, which create_coco_osm.py labels "grass";
    # the "park" label (park layer) is only drawn as a faint green outline, so it is not predicted
    "building": ["#d9d0c9"],
    "grass": ["#cdebb0", "#c8facc"],
    "woods": ["#add19e"],
    "playground": ["#dffce2"],
    "sports_field": ["#aae0cb"],
    "cemetery": ["#aacbaf"],
    "school": ["#ffffe5"],
    "residential": ["#e0dfdf"],
    "commercial": ["#f2dad9"],
    "retail": ["#ffd6d1"],
    "industrial": ["#ebdbe8"],
    "bare_ground": ["#f5e9c6", "#eee5dc"],
    "plaza": ["#dddde8"],
}
WATER_COLOR = "#aad3df"  # water and pool share it
LINE_COLORS = {
    "highway": ["#e892a2", "#f9b29c"],
    "street": ["#fcd6a4", "#f7fabf", "#ffffff"],
    "railway": ["#707070", "#787878"],
    "path": ["#fa8072"],
}
LAND_COLOR = "#f2efe9"  # background; competes for pixels but is never predicted


# ---------------------------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------------------------


def rgb(h):
    return [int(h[i : i + 2], 16) for i in (1, 3, 5)]


PALETTE = sorted(
    {
        h
        for hs in [
            *POLY_COLORS.values(),
            *LINE_COLORS.values(),
            [WATER_COLOR, LAND_COLOR],
        ]
        for h in hs
    }
)
TEXT_COLORS = [
    "#4d80b3",
    "#7981b0",
    "#d08f55",
]  # water, station and house-number labels (others are near black)
LANDUSE = {
    "residential",
    "commercial",
    "retail",
    "industrial",
    "school",
    "cemetery",
}  # split by roads when drawn
SMALL_KERNEL = {
    "playground": 7
}  # #dffce2 is also the blend of park green and white road edges


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def remove_text(arr, halo=3):
    """Paint over map labels: dark glyphs plus their white halo take the colour of the nearest pixel
    outside them, so labels don't turn into railway (grey letters) or street (white halo) shapes.
    """
    core = arr.max(-1) < 0x50  # black / #222222 glyphs
    for h in TEXT_COLORS:
        core |= np.abs(arr.astype(np.int16) - rgb(h)).max(-1) <= 12
    region = cv2.dilate(core.astype(np.uint8), disk(halo)).astype(bool)
    if (
        not region.any() or region.all()
    ):  # all text: no pixel outside it to take a colour from
        return arr
    _, (iy, ix) = ndimage.distance_transform_edt(region, return_indices=True)
    return arr[iy, ix]


def nearest_color(arr, tol):
    """Index into PALETTE of each pixel's closest colour (max per-channel difference), -1 if none within tol.
    Taking the closest colour stops near-white land (e.g. #f7f5f1) from counting as school (#ffffe5).
    """
    # one palette colour at a time: comparing all at once needs an H x W x colours x 3 array
    # (~170 MB for a 1100 px crop, ~4x that for --scale 2 crops), which can run out of memory
    a = arr.astype(np.int16)
    best = np.full(arr.shape[:2], np.iinfo(np.int16).max, np.int16)
    idx = np.full(arr.shape[:2], -1, np.intp)
    for i, h in enumerate(PALETTE):
        d = np.abs(a - np.array(rgb(h), np.int16)).max(-1)
        closer = d < best  # strict: ties keep the earlier colour, as argmin did
        best[closer] = d[closer]
        idx[closer] = i
    idx[best > tol] = -1
    return idx


def color_mask(idx, hexes):
    return np.isin(idx, [PALETTE.index(h) for h in hexes])


def mask_to_polygons(m, min_area=20, eps=1.5, open_px=3):
    m = cv2.morphologyEx(
        m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8)
    )  # drop specks and fringes
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
    nbrs = lambda p: [
        (p[0] + dx, p[1] + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx or dy) and (p[0] + dx, p[1] + dy) in pixels
    ]
    path = [next((p for p in pixels if len(nbrs(p)) == 1), next(iter(pixels)))]
    seen = {path[0]}
    while nxt := [q for q in nbrs(path[-1]) if q not in seen]:
        path.append(nxt[0])
        seen.add(nxt[0])
    return path


def join_at_junctions(pieces, jlab, max_turn=40, look=10):
    """Join skeleton pieces that carry on straight through a junction, so a street that crosses
    other streets comes out as one line (as it is in OSM), not one piece per block.
    pieces: ordered (x, y) pixel lists; jlab: junction cluster label image (0 = none).
    """
    h, w = jlab.shape
    ends = {}  # junction id -> [(piece, end 0/1, outward unit direction)]
    for i, p in enumerate(pieces):
        for e, (tip, inner) in enumerate(
            [(p[0], p[min(look, len(p) - 1)]), (p[-1], p[max(-look - 1, -len(p))])]
        ):
            x, y = tip
            near = jlab[max(y - 2, 0) : y + 3, max(x - 2, 0) : x + 3]
            j = near.max()
            if j and len(p) > 1:
                d = np.subtract(tip, inner).astype(float)
                ends.setdefault(j, []).append((i, e, d / (np.linalg.norm(d) or 1)))
    link = {}  # (piece, end) -> (piece, end)
    cos_min = np.cos(np.radians(180 - max_turn))
    for items in ends.values():
        pairs = sorted(
            (
                (np.dot(a[2], b[2]), a, b)
                for k, a in enumerate(items)
                for b in items[k + 1 :]
                if a[0] != b[0]
            ),
            key=lambda t: t[0],
        )
        for (
            c,
            a,
            b,
        ) in pairs:  # most opposite outward directions = straightest continuation first
            if c > cos_min:
                break
            if (a[0], a[1]) not in link and (b[0], b[1]) not in link:
                link[(a[0], a[1])], link[(b[0], b[1])] = (b[0], b[1]), (a[0], a[1])
    out, used = [], set()
    starts = [
        i for i in range(len(pieces)) if (i, 0) not in link or (i, 1) not in link
    ] + list(range(len(pieces)))
    for i in starts:
        if i in used:
            continue
        e = 0 if (i, 0) not in link else 1  # walk away from the free end
        line = []
        while i not in used:
            used.add(i)
            p = pieces[i] if e == 0 else pieces[i][::-1]
            line += p
            nxt = link.get((i, 1 - e))
            if nxt is None:
                break
            i, e = nxt
        out.append(line)
    return out


def mask_to_lines(m, min_len=8, eps=1.5, close_px=0, spur_len=20):
    if (
        close_px
    ):  # bridge dashes (footways, railway): widen until they touch, then thin back to a centreline
        m = cv2.dilate(m.astype(np.uint8), disk(close_px)).astype(bool)
    sk = skeletonize(m)
    n_nbrs = cv2.filter2D(sk.astype(np.float32), -1, np.ones((3, 3), np.float32)) - sk
    junction = sk & (n_nbrs > 2)
    sk &= ~junction  # cut at junctions -> simple paths
    _, jlab = cv2.connectedComponents(
        cv2.dilate(junction.astype(np.uint8), disk(1)), connectivity=8
    )
    n, comp = cv2.connectedComponents(sk.astype(np.uint8), connectivity=8)
    pieces = []
    for i in range(1, n):
        ys, xs = np.nonzero(comp == i)
        if len(xs) >= 2:
            pieces.append(trace(set(zip(xs.tolist(), ys.tolist()))))
    # drop spurs: short pieces hanging off a junction with a free end (label leftovers, corner nubs)
    at_j = lambda x, y: jlab[max(y - 2, 0) : y + 3, max(x - 2, 0) : x + 3].max() > 0
    pieces = [p for p in pieces if len(p) >= spur_len or at_j(*p[0]) == at_j(*p[-1])]
    out = []
    for line in join_at_junctions(pieces, jlab):
        if len(line) >= min_len:
            path = np.array(line, np.int32).reshape(-1, 1, 2)
            out.append(
                cv2.approxPolyDP(path, eps, False).reshape(-1, 2).astype(float).tolist()
            )
    return out


def remove_specks(m, min_px=200):
    n, comp, stats, _ = cv2.connectedComponentsWithStats(
        cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, disk(2)), connectivity=4
    )
    big = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= min_px)
    # leave out label 0 (the background) by its id: [1:] assumed it was always big enough to be first,
    # and dropped a real block instead when the mask covers nearly the whole crop
    return np.isin(comp, big[big != 0])


def predict(arr, tol=12, pool_max=3000.0, min_area=20.0, join_px=12):
    """Shapes [{"label", "points"}] found by colour in an RGB image array."""
    idx = nearest_color(remove_text(arr), tol)
    road = color_mask(idx, [h for hs in LINE_COLORS.values() for h in hs])
    shapes = []
    for label, hexes in POLY_COLORS.items():
        m = color_mask(idx, hexes)
        if (
            label in LANDUSE and join_px and m.any()
        ):  # landuse is one polygon under the roads: rejoin the blocks
            m = remove_specks(
                m
            )  # only real blocks, so anti-aliased specks along roads don't get joined up
            m |= (
                cv2.morphologyEx(
                    m.astype(np.uint8), cv2.MORPH_CLOSE, disk(join_px)
                ).astype(bool)
                & road
            )
        shapes += [
            {"label": label, "kind": "polygon", "points": pts}
            for _, pts in mask_to_polygons(
                m, min_area, open_px=SMALL_KERNEL.get(label, 3)
            )
        ]
    for area, pts in mask_to_polygons(color_mask(idx, [WATER_COLOR]), min_area):
        shapes.append(
            {
                "label": "pool" if area < pool_max else "water",
                "kind": "polygon",
                "points": pts,
            }
        )
    for label, hexes in LINE_COLORS.items():
        close = {"path": 6, "railway": 2}.get(
            label, 0
        )  # footway dashes are ~4 px with ~11 px gaps
        shapes += [
            {"label": label, "kind": "line", "points": pts}
            for pts in mask_to_lines(
                color_mask(idx, hexes), min_len=8 if close == 0 else 15, close_px=close
            )
        ]
    return shapes


# ---------------------------------------------------------------------------------------------


def config():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument(
        "--data",
        required=True,
        help="COCO dataset root (built by data_preprocess/create_coco_osm.py)",
    )
    a.add_argument("--split", default="test")
    a.add_argument(
        "--out",
        default=None,
        help="output folder (default results/color_threshold/<split>)",
    )
    a.add_argument(
        "--limit", type=int, default=0, help="only the first N images (0 = all)"
    )
    a.add_argument(
        "--tol", type=int, default=12, help="max per-channel colour difference (0-255)"
    )
    a.add_argument(
        "--pool_max",
        type=float,
        default=3000.0,
        help="blue areas below this (px^2) are pools",
    )
    a.add_argument(
        "--min_area",
        type=float,
        default=20.0,
        help="drop predicted polygons below this (px^2)",
    )
    return a.parse_args()


def main(args):
    out = Path(args.out or REPO / "results" / "color_threshold" / args.split)
    out.mkdir(parents=True, exist_ok=True)
    images = json.load(open(Path(args.data) / "annotations" / f"{args.split}.json"))[
        "images"
    ]
    if args.limit:
        images = images[: args.limit]
    preds = {}
    for k, info in enumerate(images):
        arr = np.array(
            Image.open(Path(args.data) / args.split / info["file_name"]).convert("RGB")
        )
        preds[info["file_name"]] = predict(arr, args.tol, args.pool_max, args.min_area)
        print(
            f"predicted [{k + 1}/{len(images)}] {info['file_name']}: {len(preds[info['file_name']])} shapes"
        )
    json.dump(preds, open(out / "predictions.json", "w"))
    print(
        f"saved {out / 'predictions.json'}  (score it with Raster2Seq's eval_baseline.py)"
    )


if __name__ == "__main__":
    main(config())
