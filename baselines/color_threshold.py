"""Colour-threshold baseline: vectorize map crops by their known OSM Carto colours, no learning.

The crops are drawn with a fixed palette (utils/render_carto.py), so many labels can be found by colour:
  1. Pixels -> mask: each pixel goes to its closest palette colour (background land included),
     if that colour is within --tol.
  2. Polygons: cv2.findContours traces each region's outline, cv2.approxPolyDP keeps the corners.
     Blue areas smaller than --pool_max px^2 become "pool", larger ones "water".
  3. Lines: the road / rail / path mask is thinned to 1-pixel centrelines (skeletonize), cut at
     junctions, and each piece is traced end to end and simplified.

Scored against the COCO annotations made by data_preprocess/create_coco_osm.py, at the PNG's own
resolution (no Raster2Seq install needed), two ways:
  - Shapes: one-to-one matching within each label, IoU >= --iou (greedy, best pairs first); lines are
    widened to --line_width px bands first. Same rule as notebooks/simple_baselines.ipynb, so a
    Raster2Seq prediction can be scored the same way later.
  - Pixels: per-label IoU of the drawn masks (predicted shapes vs. true shapes, both rasterized).

Usage:
    pip install opencv-python-headless scikit-image shapely numpy pillow matplotlib
    python baselines/color_threshold.py                                # mini test set (12 Queens crops)
    python baselines/color_threshold.py --data data/coco_osm --split test --plots 10

Writes to --out (default results/color_threshold/<split>/):
    per_label.csv   label, true / predicted / matched shapes, precision, recall, F1, pixel IoU
    per_image.csv   the same totals per crop
    plots/*.png     map | ground truth | prediction, for the first --plots crops
CPU only; a GPU is not used.
"""

import argparse
import collections
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import LineString, Polygon
from skimage.morphology import skeletonize

REPO = Path(__file__).resolve().parents[1]

# Labels stored as open lines in create_coco_osm.py's VOCAB (everything else is a closed polygon)
LINE_LABELS = {"highway", "street", "path", "railway", "pier", "waterway", "runway", "unknown_line"}

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
        shapes += [{"label": label, "points": pts}
                   for _, pts in mask_to_polygons(color_mask(idx, hexes), min_area)]
    for area, pts in mask_to_polygons(color_mask(idx, [WATER_COLOR]), min_area):
        shapes.append({"label": "pool" if area < pool_max else "water", "points": pts})
    for label, hexes in LINE_COLORS.items():
        shapes += [{"label": label, "points": pts} for pts in mask_to_lines(color_mask(idx, hexes))]
    return shapes


# ---------------------------------------------------------------------------------------------
# Ground truth and scoring
# ---------------------------------------------------------------------------------------------

def load_coco(data, split):
    with open(Path(data) / "annotations" / f"{split}.json") as fh:
        coco = json.load(fh)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    by_image = collections.defaultdict(list)
    for a in coco["annotations"]:
        pts = np.asarray(a["segmentation"][0], float).reshape(-1, 2).tolist()
        by_image[a["image_id"]].append({"label": names[a["category_id"]], "points": pts})
    return coco["images"], by_image


def geometry(s, line_width):
    if s["label"] in LINE_LABELS:
        return LineString(s["points"]).buffer(line_width / 2)
    return Polygon(s["points"]).buffer(0)


def match(preds, gts, line_width=6.0, thr=0.5):
    """Matched shapes per label: one-to-one within a label, IoU >= thr, best pairs first."""
    pg = [geometry(s, line_width) for s in preds]
    gg = [geometry(s, line_width) for s in gts]
    gb = np.array([g.bounds for g in gg]).reshape(-1, 4)
    pairs = []
    for i, (p, a) in enumerate(zip(preds, pg)):
        if a.is_empty:
            continue
        x0, y0, x1, y1 = a.bounds  # only compare shapes whose boxes overlap
        near = np.flatnonzero((gb[:, 0] <= x1) & (gb[:, 2] >= x0) & (gb[:, 1] <= y1) & (gb[:, 3] >= y0))
        for j in near:
            if gts[j]["label"] != p["label"]:
                continue
            union = a.union(gg[j]).area
            if union > 0:
                pairs.append((a.intersection(gg[j]).area / union, i, int(j)))
    used_p, used_g, hits = set(), set(), collections.Counter()
    for iou, i, j in sorted(pairs, reverse=True):
        if iou < thr:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            hits[preds[i]["label"]] += 1
    return hits


def rasterize(shapes, shape_hw, line_width):
    """label -> boolean mask of where its shapes are drawn."""
    masks = {}
    for s in shapes:
        m = masks.setdefault(s["label"], np.zeros(shape_hw, np.uint8))
        pts = np.round(np.asarray(s["points"])).astype(np.int32).reshape(-1, 1, 2)
        if s["label"] in LINE_LABELS:
            cv2.polylines(m, [pts], False, 1, thickness=max(1, int(round(line_width))))
        else:
            cv2.fillPoly(m, [pts], 1)
    return {k: v.astype(bool) for k, v in masks.items()}


# ---------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------

def plot(img, gt, pred, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    labels = sorted({s["label"] for s in gt + pred})
    colors = {l: plt.cm.tab20(i % 20) for i, l in enumerate(labels)}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, (title, shapes) in zip(axes, [("rendered map", None), (f"ground truth ({len(gt)})", gt),
                                          (f"colour threshold ({len(pred)})", pred)]):
        ax.imshow(img, alpha=1 if shapes is None else 0.35)
        for s in shapes or []:
            is_poly = s["label"] not in LINE_LABELS
            ax.add_patch(MplPolygon(s["points"], closed=is_poly, fill=is_poly, facecolor=colors[s["label"]],
                                    edgecolor=colors[s["label"]], alpha=0.6, linewidth=0.6 if is_poly else 1.2))
        ax.set(title=title, xlim=(0, img.width), ylim=(img.height, 0), aspect="equal")
        ax.axis("off")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=colors[l]) for l in labels], labels,
               loc="lower center", ncol=min(len(labels), 8), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------

def ratio(a, b):
    return a / b if b else 0.0


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


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
    a.add_argument("--line_width", type=float, default=6.0, help="band width (px) lines are widened to")
    return a.parse_args()


def main(args):
    out = Path(args.out or REPO / "results" / "color_threshold" / args.split)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    images, gts = load_coco(args.data, args.split)
    if args.limit:
        images = images[:args.limit]

    n_pred, n_gt, hits = collections.Counter(), collections.Counter(), collections.Counter()
    inter, union = collections.Counter(), collections.Counter()
    rows = []
    for k, info in enumerate(images):
        img = Image.open(Path(args.data) / args.split / info["file_name"]).convert("RGB")
        arr = np.array(img)
        gt, pred = gts[info["id"]], predict(arr, args.tol, args.pool_max, args.min_area)

        h = match(pred, gt, args.line_width, args.iou)
        np_, ng = collections.Counter(s["label"] for s in pred), collections.Counter(s["label"] for s in gt)
        n_pred += np_; n_gt += ng; hits += h

        gm, pm = rasterize(gt, arr.shape[:2], args.line_width), rasterize(pred, arr.shape[:2], args.line_width)
        for label in set(gm) | set(pm):
            g = gm.get(label, np.zeros(arr.shape[:2], bool))
            p = pm.get(label, np.zeros(arr.shape[:2], bool))
            inter[label] += int((g & p).sum())
            union[label] += int((g | p).sum())

        P, G, H = sum(np_.values()), sum(ng.values()), sum(h.values())
        rows.append({"image": info["file_name"], "true": G, "predicted": P, "matched": H,
                     "precision": round(ratio(H, P), 3), "recall": round(ratio(H, G), 3)})
        print(f"[{k + 1}/{len(images)}] {info['file_name']}: {G} true, {P} predicted, {H} matched")
        if k < args.plots:
            plot(img, gt, pred, out / "plots" / info["file_name"])

    labels = sorted(set(n_pred) | set(n_gt), key=lambda l: (-n_gt[l], l))
    table = []
    for label in labels:
        p, r = ratio(hits[label], n_pred[label]), ratio(hits[label], n_gt[label])
        table.append({"label": label, "kind": "line" if label in LINE_LABELS else "polygon",
                      "true": n_gt[label], "predicted": n_pred[label], "matched": hits[label],
                      "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1(p, r), 3),
                      "pixel_iou": round(ratio(inter[label], union[label]), 3)})
    P, G, H = sum(n_pred.values()), sum(n_gt.values()), sum(hits.values())
    p, r = ratio(H, P), ratio(H, G)
    table.append({"label": "ALL", "kind": "", "true": G, "predicted": P, "matched": H,
                  "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1(p, r), 3),
                  "pixel_iou": round(ratio(sum(inter.values()), sum(union.values())), 3)})

    for name, data in [("per_label.csv", table), ("per_image.csv", rows)]:
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    print(f"\n{'label':<13} {'kind':<8} {'true':>5} {'pred':>5} {'match':>6} {'prec':>6} {'recall':>7} "
          f"{'f1':>6} {'pix IoU':>8}")
    for t in table:
        print(f"{t['label']:<13} {t['kind']:<8} {t['true']:>5} {t['predicted']:>5} {t['matched']:>6} "
              f"{t['precision']:>6.2f} {t['recall']:>7.2f} {t['f1']:>6.2f} {t['pixel_iou']:>8.2f}")
    print(f"\nwrote {out}/per_label.csv, per_image.csv and {min(args.plots, len(images))} plots")


if __name__ == "__main__":
    main(config())
