"""Convert OSM crops (<name>.geojson.gz + <name>.png) into Raster2Seq's COCO layout.

Output (readable by Raster2Seq datasets/poly_data.py -> ConvertToCocoDict):
    <output>/{train,val,test}/<name>.png
    <output>/annotations/{train,val,test}.json

Each OpenMapTiles (layer, class) is mapped to a label from the fixed vocabulary VOCAB
(e.g. "street", "building", "park"); anything not in VOCAB is labelled "unknown" (polygons)
or "unknown_line" (lines).
Shapes are clipped to the image rectangle (parts outside it are dropped, so nothing is squashed
onto the border) and simplified. Polygons are stored as closed rings (closing point dropped,
corners resorted). LineStrings are stored start -> finish, are NOT resorted, and carry
"geom_type": "line" (polygons: "polygon") so the loader can skip resort_corners for them.
By default, polygons and lines are saved as in create_coco_osm.py. If --line_buffer_px
is provided, lines are buffered into closed polygons in pixel space. Buffered annotations
also include source_geom_type to record whether the original geometry was a line or polygon.
"""

# Example commands (run from the project root with the raster2seq environment active):
#
# Convert one crop without buffering (keep line features as lines):
# python data_preprocess/create_coco_osm.py --data_root crops \
#   --output data/coco_osm_no_buffer --sample_name queens_05319
#
# Convert the same crop with a 2-pixel line buffer radius:
# python data_preprocess/create_coco_osm.py --data_root crops \
#   --output data/coco_osm_buffer --sample_name queens_05319 --line_buffer_px 2

import argparse
import gzip
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import LineString, Polygon
from shapely.ops import clip_by_rect

REPO = Path(__file__).resolve().parents[1]
R2S = REPO.parent / "Raster2Seq"
sys.path[:0] = [str(REPO), str(R2S), str(R2S / "data_preprocess" / "waffle")]

from create_coco_waffle_benchmark import (
    convert_numpy_to_python,
    create_coco_bounding_box,
    prepare_dict,
)
from util.poly_ops import resort_corners
from utils.render_carto import lonlat_to_px, pixel_window

# GeoJSON geometry type -> (geom_type, list of point lists); polygons keep only the outer ring
PARTS = {
    "Polygon": lambda c: ("polygon", [c[0]]),
    "MultiPolygon": lambda c: ("polygon", [p[0] for p in c]),
    "LineString": lambda c: ("line", [c]),
    "MultiLineString": lambda c: ("line", c),
}
# label -> OpenMapTiles keys: (layer, class), (layer, class, geom_type), or (layer, "*") for any class.
# Anything not listed (and not in SKIP) is labelled "unknown" / "unknown_line" (e.g. military areas / ferries).
# Labels distinguish the original polygon and line categories. Buffered lines
# keep their original labels even though their output geometry becomes a polygon.
VOCAB = {
    "building": [("building", "*")],
    "highway": [
        ("transportation", c) for c in ("motorway", "motorway_construction", "trunk")
    ],
    "street": [
        ("transportation", c)
        for c in (
            "primary",
            "secondary",
            "tertiary",
            "minor",
            "service",
            "busway",
            "secondary_construction",
            "tertiary_construction",
            "minor_construction",
            "service_construction",
        )
    ],
    "path": [
        ("transportation", "path", "line"),
        ("transportation", "track"),
        ("transportation", "path_construction"),
    ],
    "plaza": [("transportation", "path", "polygon")],  # pedestrian areas
    "railway": [("transportation", "rail"), ("transportation", "transit")],
    "rail_yard": [("landuse", "railway")],
    "bridge": [("transportation", "bridge")],
    "pier": [("transportation", "pier", "line")],
    "pier_area": [("transportation", "pier", "polygon")],
    "water": [("water", c) for c in ("lake", "pond", "river", "ocean", "dock")],
    "pool": [("water", "swimming_pool")],
    "waterway": [("waterway", "*")],
    "park": [
        ("park", "*"),
        ("landuse", "recreation_ground"),
        ("landuse", "zoo"),
        ("landuse", "theme_park"),
    ],
    "playground": [("landuse", "playground")],
    "sports_field": [("landuse", c) for c in ("pitch", "track", "stadium")],
    "grass": [("landcover", "grass"), ("landcover", "farmland")],
    "woods": [("landcover", "wood")],
    "wetland": [("landcover", "wetland")],
    "bare_ground": [("landcover", "sand"), ("landcover", "rock")],
    "school": [
        ("landuse", c)
        for c in ("school", "kindergarten", "education", "college", "university")
    ],
    "hospital": [("landuse", "hospital")],
    "civic": [("landuse", "library"), ("landuse", "bus_station")],
    "cemetery": [("landuse", "cemetery")],
    "residential": [("landuse", "residential")],
    "commercial": [("landuse", "commercial")],
    "retail": [("landuse", "retail")],
    "industrial": [("landuse", "industrial")],
    "airport": [("aeroway", c) for c in ("aerodrome", "apron", "helipad")],
    "runway": [("aeroway", "runway", "line"), ("aeroway", "taxiway", "line")],
    "runway_area": [("aeroway", "runway", "polygon")],
    "unknown": [],  # fallback for polygons not listed above
    "unknown_line": [],  # fallback for lines not listed above
}
LABELS = {key: label for label, keys in VOCAB.items() for key in keys}
CATEGORIES = {label: i for i, label in enumerate(VOCAB)}
SKIP = {  # dropped entirely: (layer, class) or (layer, "*")
    ("transportation_name", "*"),
    ("water_name", "*"),  # copies of other features, only used to place names
    ("boundary", "*"),
    ("landuse", "neighbourhood"),  # administrative, not drawn as visible features
}


def label_for(layer, cls, geom_type):
    """Vocabulary label for an OpenMapTiles feature (most specific key wins), else "unknown"/"unknown_line"."""
    return (
        LABELS.get((layer, cls, geom_type))
        or LABELS.get((layer, cls))
        or LABELS.get((layer, "*"))
        or ("unknown" if geom_type == "polygon" else "unknown_line")
    )


def extract_geometries(data, to_px):
    """(points, label, geom_type) per part; polygons drop the repeated closing point."""
    out = []
    for f in data["features"]:
        g, props = f["geometry"], f["properties"]
        layer, cls = props["omt_layer"], props.get("class")
        if g["type"] not in PARTS or (layer, cls) in SKIP or (layer, "*") in SKIP:
            continue
        geom_type, parts = PARTS[g["type"]](g["coordinates"])
        label = label_for(layer, cls, geom_type)
        is_poly = geom_type == "polygon"
        for part in parts:
            # every point except the last point
            pts = [to_px(lon, lat) for lon, lat in (part[:-1] if is_poly else part)]
            if len(pts) >= 2 + is_poly:
                out.append((pts, label, geom_type))
    return out


def make_to_px(data, png_width):
    """lon/lat -> pixels of the rendered PNG (crops are rendered at camera_zoom, 2x)."""
    zoom = data["metadata"]["camera_zoom"]
    x0, y0, x1, _ = pixel_window(data["bbox"], zoom)
    scale = png_width / (x1 - x0)

    def to_px(lon, lat):
        """Convert to image pixels"""
        px, py = lonlat_to_px(lon, lat, zoom)
        return (px - x0) * scale, (py - y0) * scale

    return to_px


def clip_to_image(shape, width, height):
    """The pieces of a shapely shape inside the image (0..width-1, 0..height-1), as a list."""
    if not shape.is_valid:
        shape = shape.buffer(0)  # repair self-intersecting outlines before clipping
    clipped = clip_by_rect(shape, 0, 0, width - 1, height - 1)
    return [g for g in getattr(clipped, "geoms", [clipped]) if not g.is_empty]


def buffer_line_geometries(geoms, radius_px):
    """Convert each line into its own polygon with the requested pixel-radius buffer."""
    if radius_px <= 0:
        raise ValueError("line_buffer_px must be greater than zero")

    buffered_geoms = []
    for pts, label, geom_type in geoms:
        if geom_type != "line":
            buffered_geoms.append((pts, label, geom_type, geom_type))
            continue

        line_area = LineString(pts).buffer(radius_px, cap_style=2, join_style=2)
        parts = (
            list(line_area.geoms)
            if line_area.geom_type == "MultiPolygon"
            else [line_area]
        )
        for polygon in parts:
            if polygon.is_empty or polygon.geom_type != "Polygon":
                continue
            ring = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
            if len(ring) >= 3:
                buffered_geoms.append((ring.tolist(), label, "polygon", "line"))

    return buffered_geoms


def build_annotations(
    geoms,
    width,
    height,
    min_area=20.0,
    simplify_tol=2.0,
    min_line_length=4.0,
    img_id=0,
    start_id=0,
):
    """COCO annotations for one image; optional 4th tuple item records the source geometry type."""
    anns = []
    for item in geoms:
        pts, label, geom_type = item[:3]
        is_poly = geom_type == "polygon"
        for piece in clip_to_image(
            (Polygon if is_poly else LineString)(pts), width, height
        ):
            if piece.geom_type != ("Polygon" if is_poly else "LineString"):
                continue  # e.g. a point or edge where a shape only touches the image border
            shape = piece.simplify(simplify_tol)
            if is_poly:
                if shape.geom_type != "Polygon" or shape.area < min_area:
                    continue
                out = resort_corners(np.asarray(shape.exterior.coords[:-1])).reshape(
                    -1, 2
                )  # drop closing point
            else:
                if shape.length < min_line_length:
                    continue
                out = np.asarray(shape.coords)
            annotation = {
                "id": start_id + len(anns),
                "image_id": img_id,
                "iscrowd": 0,
                "category_id": CATEGORIES[label],
                "geom_type": geom_type,
                "segmentation": [out.reshape(-1).tolist()],
                "area": shape.area,
                "bbox": create_coco_bounding_box(
                    out[:, 0], out[:, 1], width, height, bound_pad=2
                ),
            }
            # Only buffered runs add a fourth item recording the original geometry type.
            if len(item) > 3:
                annotation["source_geom_type"] = item[3]
            anns.append(annotation)
    return anns


def config():
    a = argparse.ArgumentParser(
        description="Generate COCO-format data from OSM geojson crops"
    )
    a.add_argument(
        "--data_root",
        required=True,
        type=str,
        help="folder with <name>.geojson.gz and <name>.png",
    )
    a.add_argument("--output", required=True, type=str, help="output dataset root")
    a.add_argument(
        "--test_borough",
        default="queens",
        type=str,
        help="crops whose name starts with <test_borough>_ form the test split; any prefix works, e.g. flushing for make_boxes.py --test-nta flushing",
    )
    a.add_argument(
        "--val_frac",
        default=0.1,
        type=float,
        help="val size as a fraction of all crops, test included (taken from non-test crops)",
    )
    a.add_argument(
        "--min_area",
        default=20.0,
        type=float,
        help="drop polygons smaller than this (px^2)",
    )
    a.add_argument(
        "--simplify_tol",
        default=2.0,
        type=float,
        help="shape simplification tolerance in PNG px (2 px ~ 1 px after Raster2Seq's 512 px resize); 0 = off",
    )
    a.add_argument(
        "--min_line_length",
        default=4.0,
        type=float,
        help="drop line pieces shorter than this (px)",
    )
    a.add_argument("--seed", default=0, type=int)
    a.add_argument(
        "--line_buffer_px",
        default=None,
        type=float,
        help="buffer radius in PNG pixels for line features; omit to keep lines unbuffered",
    )
    a.add_argument(
        "--sample_name",
        default="",
        type=str,
        help="convert only this crop basename (without .geojson.gz), and write it to test",
    )
    a.add_argument(
        "--max_crops",
        default=0,
        type=int,
        help="optional maximum number of paired crops to convert (0 means all)",
    )
    args = a.parse_args()
    if args.line_buffer_px is not None and (
        not math.isfinite(args.line_buffer_px) or args.line_buffer_px <= 0
    ):
        a.error(
            "--line_buffer_px must be greater than 0 and finite; omit it to disable buffering"
        )
    return args


def main(args):
    root, out = Path(args.data_root), Path(args.output)
    pngs = {
        f: f.with_name(f.name.replace(".geojson.gz", ".png"))
        for f in sorted(root.glob("*.geojson.gz"))
    }
    pngs = {f: p for f, p in pngs.items() if p.exists()}

    if args.sample_name:
        expected = f"{args.sample_name}.geojson.gz"
        pngs = {f: p for f, p in pngs.items() if f.name == expected}
        if not pngs:
            raise FileNotFoundError(
                f"No paired GeoJSON/PNG crop found for {expected} in {root}"
            )
        splits = {"test": list(pngs)}
    else:
        if args.max_crops > 0:
            pngs = dict(list(pngs.items())[: args.max_crops])
        is_test = lambda f: f.name.startswith(f"{args.test_borough}_")
        rest = [f for f in pngs if not is_test(f)]
        random.Random(args.seed).shuffle(rest)
        n_val = int(args.val_frac * len(pngs))
        splits = {
            "train": rest[n_val:],
            "val": rest[:n_val],
            "test": [f for f in pngs if is_test(f)],
        }

    # write images + COCO json per split (category ids come from the fixed VOCAB)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    ann_id = 0
    for split, split_files in splits.items():
        (out / split).mkdir(exist_ok=True)
        coco = prepare_dict(CATEGORIES)
        for img_id, f in enumerate(split_files):
            png, img = pngs[f], Image.open(pngs[f])
            w, h = img.size
            with gzip.open(f, "rt") as fh:
                data = json.load(fh)
            # crops are palette PNGs; Raster2Seq reads images with np.array(), which would give palette indices
            img.convert("RGB").save(out / split / png.name)
            coco["images"].append(
                {"file_name": png.name, "id": img_id, "width": w, "height": h}
            )
            geoms = extract_geometries(data, make_to_px(data, w))
            # Keep the original line coordinates unless a buffer radius was supplied.
            if args.line_buffer_px is not None:
                geoms = buffer_line_geometries(geoms, args.line_buffer_px)
            anns = build_annotations(
                geoms,
                w,
                h,
                args.min_area,
                args.simplify_tol,
                args.min_line_length,
                img_id,
                ann_id,
            )
            coco["annotations"] += anns
            ann_id += len(anns)
        with open(out / "annotations" / f"{split}.json", "w") as fh:
            json.dump(coco, fh, default=convert_numpy_to_python)
        print(
            f"{split}: {len(split_files)} images, {len(coco['annotations'])} annotations"
        )

    print("categories:", CATEGORIES)


if __name__ == "__main__":
    main(config())
