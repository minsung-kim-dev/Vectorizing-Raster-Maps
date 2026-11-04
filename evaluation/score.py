"""Shared scorer: score any method's predicted shapes against a COCO split, the same way for every method.

Predictions file (JSON), one entry per image, coordinates in pixels of the PNG as stored in the dataset:
    {
      "method": "colour threshold",
      "images": {
        "queens_01156.png": [
          {"label": "building", "kind": "polygon", "points": [[x, y], [x, y], ...]},
          {"label": "street",   "kind": "line",    "points": [[x, y], [x, y], ...]}
        ]
      }
    }
"label" is a category name from the dataset (create_coco_osm.py's VOCAB). "kind" is "polygon" (closed,
first point not repeated) or "line" (open, start -> finish). Images missing from the file count as
empty predictions. Raster2Seq output must be scaled back from the model's input size (e.g. 512 px,
padded) to the PNG's own width/height before it is written here.

What it reports (all in --out):
    per_label.csv   true / predicted / matched shapes, precision, recall, F1, pixel IoU per label
    per_bin.csv     recall per geometry bin of the true shapes (size, vertices, concavity, angles,
                    overlap, crop edge, line vs polygon)
    per_image.csv   totals per image
    summary.json    overall and macro-averaged scores, with 95% bootstrap intervals over images

Matching: one-to-one within a label, IoU >= --iou, best pairs first (as in notebooks/simple_baselines.ipynb).
Lines are scored as polygons widened --line_buffer_px on each side, like the training data from
create_coco_osm.py --line_buffer_px (default 2). --line_buffer_px 0 scores centrelines widened to
--line_width bands instead.

Usage:
    python evaluation/score.py --pred results/color_threshold/test/predictions.json
    python evaluation/score.py --pred ground_truth          # sanity check: should score 1.0 everywhere
"""

import argparse
import collections
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon

REPO = Path(__file__).resolve().parents[1]
MODEL_INPUT = 512  # Raster2Seq's --image_size; geometry bins are measured at this scale


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------

def load_coco(data, split):
    """(images, {image_id: [shape]}) with shape = {"label", "kind", "points", "buffered"}."""
    with open(Path(data) / "annotations" / f"{split}.json") as fh:
        coco = json.load(fh)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    by_image = collections.defaultdict(list)
    for a in coco["annotations"]:
        pts = np.asarray(a["segmentation"][0], float).reshape(-1, 2).tolist()
        kind = "line" if a.get("geom_type") == "line" else "polygon"  # buffered lines are polygons
        by_image[a["image_id"]].append({"label": names[a["category_id"]], "kind": kind, "points": pts,
                                        "buffered": a.get("source_geom_type") == "line"})
    return coco["images"], by_image


def load_predictions(path):
    with open(path) as fh:
        d = json.load(fh)
    return d.get("method", Path(path).stem), d["images"]


def save_predictions(path, method, images):
    with open(path, "w") as fh:
        json.dump({"method": method, "images": images}, fh)


# ---------------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------------

def buffer_lines(shapes, radius):
    """Turn every line into a polygon by widening it `radius` px on each side (outer ring only)."""
    out = []
    for s in shapes:
        if s["kind"] != "line":
            out.append(s)
            continue
        g = LineString(s["points"]).buffer(radius).simplify(0.5)
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda p: p.area)
        if g.geom_type == "Polygon" and not g.is_empty:
            out.append({**s, "kind": "polygon", "points": [list(p) for p in g.exterior.coords[:-1]]})
    return out


def geometry(s, line_width):
    if s["kind"] == "line":
        return LineString(s["points"]).buffer(line_width / 2)
    return Polygon(s["points"]).buffer(0)


def match(preds, gts, line_width=6.0, thr=0.5):
    """Matched (pred index, true index) pairs: one-to-one within a label, IoU >= thr, best pairs first."""
    pg = [geometry(s, line_width) for s in preds]
    gg = [geometry(s, line_width) for s in gts]
    gb = np.array([g.bounds if not g.is_empty else (np.inf, np.inf, -np.inf, -np.inf) for g in gg]).reshape(-1, 4)
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
    used_p, used_g, matched = set(), set(), []
    for iou, i, j in sorted(pairs, reverse=True):
        if iou < thr:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            matched.append((i, j))
    return matched


def rasterize(shapes, shape_hw, line_width):
    """label -> boolean mask of where its shapes are drawn."""
    masks = {}
    for s in shapes:
        m = masks.setdefault(s["label"], np.zeros(shape_hw, np.uint8))
        pts = np.round(np.asarray(s["points"])).astype(np.int32).reshape(-1, 1, 2)
        if s["kind"] == "line":
            cv2.polylines(m, [pts], False, 1, thickness=max(1, int(round(line_width))))
        else:
            cv2.fillPoly(m, [pts], 1)
    return {k: v.astype(bool) for k, v in masks.items()}


# ---------------------------------------------------------------------------------------------
# Geometry bins of the true shapes
# ---------------------------------------------------------------------------------------------

def corner_angles(pts):
    p = np.asarray(pts, float)
    v1, v2 = np.roll(p, 1, 0) - p, np.roll(p, -1, 0) - p
    cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def geometry_bins(gts, width, height):
    """One dict of {factor: bin} per true shape, from its original (unbuffered) geometry."""
    scale = MODEL_INPUT / max(width, height)
    polys = [Polygon(s["points"]).buffer(0) if s["kind"] == "polygon" and not s["buffered"] else None for s in gts]
    boxes = np.array([p.bounds if p is not None and not p.is_empty else (np.inf, np.inf, -np.inf, -np.inf)
                      for p in polys]).reshape(-1, 4)
    out = []
    for k, s in enumerate(gts):
        pts = np.asarray(s["points"], float)
        edge = bool((pts[:, 0] <= 1).any() or (pts[:, 1] <= 1).any()
                    or (pts[:, 0] >= width - 2).any() or (pts[:, 1] >= height - 2).any())
        b = {"kind": "line" if s["kind"] == "line" or s["buffered"] else "polygon",
             "crop_edge": "touches edge" if edge else "inside"}
        p = polys[k]
        if p is not None and not p.is_empty and p.area > 0:
            side = np.sqrt(p.area) * scale
            n = len(s["points"])
            x0, y0, x1, y1 = p.bounds
            near = np.flatnonzero((boxes[:, 0] < x1) & (boxes[:, 2] > x0) & (boxes[:, 1] < y1) & (boxes[:, 3] > y0))
            overlaps = any(j != k and p.intersection(polys[j]).area > 0.05 * p.area for j in near)
            b.update({
                "size_at_512px": "small (<15 px)" if side < 15 else "medium (15-60 px)" if side <= 60 else "large (>60 px)",
                "vertices": "3-4" if n <= 4 else "5-9" if n <= 9 else "10+",
                "convexity": "convex" if p.area >= 0.97 * p.convex_hull.area else "concave",
                "angles": "all near 90 deg" if (np.abs(corner_angles(s["points"]) - 90) < 10).all() else "other angles",
                "overlap": "overlaps another shape" if overlaps else "no overlap",
            })
        out.append(b)
    return out


# ---------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------

def plot(img, gt, pred, path, method="prediction"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    labels = sorted({s["label"] for s in gt + pred})
    colors = {l: plt.cm.tab20(i % 20) for i, l in enumerate(labels)}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, (title, shapes) in zip(axes, [("rendered map", None), (f"ground truth ({len(gt)})", gt),
                                          (f"{method} ({len(pred)})", pred)]):
        ax.imshow(img, alpha=1 if shapes is None else 0.35)
        for s in shapes or []:
            is_poly = s["kind"] != "line"
            ax.add_patch(MplPolygon(s["points"], closed=is_poly, fill=is_poly, facecolor=colors[s["label"]],
                                    edgecolor=colors[s["label"]], alpha=0.6, linewidth=0.6 if is_poly else 1.2))
        ax.set(title=title, xlim=(0, img.width), ylim=(img.height, 0), aspect="equal")
        ax.axis("off")
    if labels:
        fig.legend([plt.Rectangle((0, 0), 1, 1, color=colors[l]) for l in labels], labels,
                   loc="lower center", ncol=min(len(labels), 8), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------

def ratio(a, b):
    return a / b if b else 0.0


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def bootstrap(per_image, n=1000, seed=0):
    """95% intervals of overall precision / recall / F1, resampling images with replacement."""
    rng = np.random.default_rng(seed)
    a = np.array([[r["predicted"], r["true"], r["matched"]] for r in per_image], float)
    stats = []
    for _ in range(n):
        P, G, H = a[rng.integers(0, len(a), len(a))].sum(0)
        p, r = ratio(H, P), ratio(H, G)
        stats.append((p, r, f1(p, r)))
    lo, hi = np.percentile(stats, [2.5, 97.5], axis=0)
    return {k: [round(float(lo[i]), 3), round(float(hi[i]), 3)] for i, k in enumerate(["precision", "recall", "f1"])}


def score(data, split, predictions, out, method="prediction", line_buffer_px=2.0, line_width=6.0, iou=0.5,
          plots=0, limit=0):
    """Score {file_name: [shape]} against the split; write CSVs + summary.json to out; return the summary."""
    from PIL import Image

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    images, gts = load_coco(data, split)
    if limit:
        images = images[:limit]
    if any(s["buffered"] for g in gts.values() for s in g):
        print("dataset lines are buffered (source_geom_type = line): make sure --line_buffer_px matches the "
              f"value used in create_coco_osm.py (using {line_buffer_px})")
        if not line_buffer_px:
            print("warning: --line_buffer_px 0 compares predicted centrelines with buffered true lines")
    missing = [i["file_name"] for i in images if i["file_name"] not in predictions]
    if missing:
        print(f"warning: no predictions for {len(missing)} of {len(images)} images (scored as empty)")

    n_pred, n_gt, hits = collections.Counter(), collections.Counter(), collections.Counter()
    inter, union = collections.Counter(), collections.Counter()
    bin_true, bin_hit = collections.Counter(), collections.Counter()
    rows = []
    for k, info in enumerate(images):
        w, h = info["width"], info["height"]
        gt_raw, pred = gts[info["id"]], predictions.get(info["file_name"], [])
        bins = geometry_bins(gt_raw, w, h)
        gt = gt_raw
        if line_buffer_px:
            # buffer_lines keeps polygons as they are and drops nothing for valid lines, so indices line up
            gt, pred = buffer_lines(gt_raw, line_buffer_px), buffer_lines(pred, line_buffer_px)
            if len(gt) != len(gt_raw):  # a degenerate true line was dropped: recompute bins to stay aligned
                keep = [s for s in gt_raw if s["kind"] != "line" or len(buffer_lines([s], line_buffer_px))]
                gt, bins = buffer_lines(keep, line_buffer_px), geometry_bins(keep, w, h)

        pairs = match(pred, gt, line_width, iou)
        matched_gt = {j for _, j in pairs}
        np_ = collections.Counter(s["label"] for s in pred)
        ng = collections.Counter(s["label"] for s in gt)
        hp = collections.Counter(gt[j]["label"] for j in matched_gt)
        n_pred += np_; n_gt += ng; hits += hp
        for j, b in enumerate(bins):
            for factor, value in b.items():
                bin_true[factor, value] += 1
                bin_hit[factor, value] += j in matched_gt

        gm, pm = rasterize(gt, (h, w), line_width), rasterize(pred, (h, w), line_width)
        for label in set(gm) | set(pm):
            g = gm.get(label, np.zeros((h, w), bool))
            p = pm.get(label, np.zeros((h, w), bool))
            inter[label] += int((g & p).sum())
            union[label] += int((g | p).sum())

        P, G, H = sum(np_.values()), sum(ng.values()), len(pairs)
        rows.append({"image": info["file_name"], "true": G, "predicted": P, "matched": H,
                     "precision": round(ratio(H, P), 3), "recall": round(ratio(H, G), 3)})
        print(f"[{k + 1}/{len(images)}] {info['file_name']}: {G} true, {P} predicted, {H} matched")
        if k < plots:
            (out / "plots").mkdir(exist_ok=True)
            img = Image.open(Path(data) / split / info["file_name"]).convert("RGB")
            plot(img, gt, pred, out / "plots" / info["file_name"], method)

    labels = sorted(set(n_pred) | set(n_gt), key=lambda l: (-n_gt[l], l))
    table = []
    for label in labels:
        p, r = ratio(hits[label], n_pred[label]), ratio(hits[label], n_gt[label])
        table.append({"label": label, "true": n_gt[label], "predicted": n_pred[label], "matched": hits[label],
                      "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1(p, r), 3),
                      # blank when neither side covers a single pixel (e.g. a sliver smaller than 1 px)
                      "pixel_iou": round(inter[label] / union[label], 3) if union[label] else ""})
    P, G, H = sum(n_pred.values()), sum(n_gt.values()), sum(hits.values())
    p, r = ratio(H, P), ratio(H, G)
    overall = {"true": G, "predicted": P, "matched": H, "precision": round(p, 3), "recall": round(r, 3),
               "f1": round(f1(p, r), 3), "pixel_iou": round(ratio(sum(inter.values()), sum(union.values())), 3)}
    table.append({"label": "ALL", **overall})
    with_truth = [t for t in table[:-1] if t["true"]]
    macro = {k: round(float(np.mean([t[k] for t in with_truth if t[k] != ""])), 3) if with_truth else 0.0
             for k in ("precision", "recall", "f1", "pixel_iou")}

    bin_rows = [{"factor": f, "bin": v, "true": n, "matched": bin_hit[f, v], "recall": round(ratio(bin_hit[f, v], n), 3)}
                for (f, v), n in sorted(bin_true.items())]
    summary = {"method": method, "data": str(data), "split": split, "images": len(images),
               "line_buffer_px": line_buffer_px, "iou": iou, "overall": overall,
               "macro_over_labels": macro, "overall_95ci": bootstrap(rows) if rows else {}}

    write_csv(out / "per_label.csv", table)
    write_csv(out / "per_image.csv", rows)
    write_csv(out / "per_bin.csv", bin_rows)
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'label':<13} {'true':>5} {'pred':>5} {'match':>6} {'prec':>6} {'recall':>7} {'f1':>6} {'pix IoU':>8}")
    for t in table:
        print(f"{t['label']:<13} {t['true']:>5} {t['predicted']:>5} {t['matched']:>6} "
              f"{t['precision']:>6.2f} {t['recall']:>7.2f} {t['f1']:>6.2f} "
              f"{t['pixel_iou'] if t['pixel_iou'] == '' else format(t['pixel_iou'], '.2f'):>8}")
    print(f"macro over labels: precision {macro['precision']:.2f}, recall {macro['recall']:.2f}, F1 {macro['f1']:.2f}")
    if summary["overall_95ci"]:
        print(f"overall F1 95% interval over images: {summary['overall_95ci']['f1']}")
    print(f"\n{'factor':<15} {'bin':<24} {'true':>5} {'match':>6} {'recall':>7}")
    for b in bin_rows:
        print(f"{b['factor']:<15} {b['bin']:<24} {b['true']:>5} {b['matched']:>6} {b['recall']:>7.2f}")
    print(f"\nwrote {out}/per_label.csv, per_bin.csv, per_image.csv, summary.json")
    return summary


def config():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--pred", required=True,
                   help="predictions JSON (format at the top of this file), or 'ground_truth' for a sanity check")
    a.add_argument("--data", default=str(REPO / "data" / "coco_osm_mini"), help="COCO dataset root")
    a.add_argument("--split", default="test")
    a.add_argument("--out", default=None, help="output folder (default: next to --pred, or results/ground_truth)")
    a.add_argument("--limit", type=int, default=0, help="only the first N images (0 = all)")
    a.add_argument("--plots", type=int, default=0, help="save side-by-side plots for the first N images")
    a.add_argument("--iou", type=float, default=0.5, help="IoU needed for a shape to count as matched")
    a.add_argument("--line_buffer_px", type=float, default=2.0,
                   help="score lines as polygons widened this many px on each side, like create_coco_osm.py's "
                        "--line_buffer_px; 0 = score lines as centrelines")
    a.add_argument("--line_width", type=float, default=6.0,
                   help="with --line_buffer_px 0: band width (px) centrelines are widened to for scoring")
    return a.parse_args()


def main(args):
    if args.pred == "ground_truth":
        images, gts = load_coco(args.data, args.split)
        method, preds = "ground truth", {i["file_name"]: [{k: s[k] for k in ("label", "kind", "points")}
                                                          for s in gts[i["id"]]] for i in images}
        out = args.out or REPO / "results" / "ground_truth" / args.split
    else:
        method, preds = load_predictions(args.pred)
        out = args.out or Path(args.pred).parent
    score(args.data, args.split, preds, out, method, args.line_buffer_px, args.line_width, args.iou,
          args.plots, args.limit)


if __name__ == "__main__":
    main(config())
