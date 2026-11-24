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
#   --output data/coco_osm_no_buffer
#
# Convert the same crop with a 2-pixel line buffer radius:
# python data_preprocess/create_coco_osm.py --data_root crops \
#   --output data/coco_osm_buffer --line_buffer_px 10.8

import argparse
import gzip
import json
import math
import random
import sys
import collections
import shapely
import pandas as pd
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import LineString, Polygon
from shapely.ops import clip_by_rect

REPO = Path(__file__).resolve().parents[1]
R2S = REPO.parent / "Raster2Seq"
sys.path[:0] = [str(REPO), str(R2S), str(R2S / "data_preprocess" / "waffle")]

from create_coco_waffle_benchmark import convert_numpy_to_python, create_coco_bounding_box, prepare_dict 
from util.poly_ops import resort_corners 
from utils.render_carto import lonlat_to_px, pixel_window 

# GeoJSON geometry type -> (geom_type, list of point lists); polygons keep only the outer ring
# Line is a list of coordinate points
# Polygon is list of rings (closed point lists), starting with the outer rings
# MultiPolygon is a list of polygons
# We don't deal with 'holes' so only take the first ring for polygons and multipolygons
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
vocab_reviewed = pd.read_csv(Path(__file__).parent / "label_colors_reviewed.csv")
kept = vocab_reviewed[vocab_reviewed.final_label.notna()].copy()
kept["class"] = np.where(kept["class"].notna(), kept["class"], None)  # building: NaN -> None
kept["subclass"] = np.where(kept["subclass"].notna(), kept["subclass"], None)
assert not kept.duplicated(["layer", "class", "subclass", "geom"]).any(), "one label per (layer, class, geom)"
VOCAB = {label: list(zip(g.layer, g["class"], g.subclass, g.geom)) for label, g in kept.groupby("final_label", sort=False)}
if 'unknown' not in VOCAB.keys():
    VOCAB |= {"unknown": []}
if 'unknown_line' not in VOCAB.keys():
    VOCAB |= {"unknown_line": []}

# inverts VOCAB mapping so then it is a tuple -> label
LABELS = {key: label for label, keys in VOCAB.items() for key in keys}
CATEGORIES = {label: i for i, label in enumerate(VOCAB)}

SKIP = {  # dropped entirely: (layer, class) or (layer, "*")
    ("transportation_name", "*"), ("water_name", "*"),  # copies of other features, only used to place names
    ("boundary", "*"), ("landuse", "neighbourhood"),      # administrative, not drawn as visible features
    # not drawn by utils/render_carto.py, so not visible in the images
    ("aeroway", "*"),                                     # aerodrome, taxiway, apron, runway
    ("transportation", "bridge"),                         # bridge deck outlines (the road on top is kept)
    ("transportation", "transit"),                        # subway, light rail
    ("transportation", "pier"), ("transportation", "busway"),
    ("transportation", "motorway_construction"), ("transportation", "tertiary_construction"),
    ("transportation", "service_construction"), ("transportation", "minor_construction"),
    ("transportation", "path_construction"),
    ("landuse", "zoo"), ("landuse", "education"), ("landuse", "toilets"),
    ("mountain_peak", "cliff"),
}
# shapes not drawn by utils/render_carto.py, by (layer, class, subclass, geometry)
SKIP_SHAPES = {("transportation", "path", "platform", "polygon"), ("transportation", "path", "footway", "polygon")}

def label_for(layer, cls, subclass, geom_type):
    """Vocabulary label for an OpenMapTiles feature, else "unknown"/"unknown_line"."""
    return (LABELS.get((layer, cls, subclass, geom_type)) or LABELS.get((layer, cls, None, geom_type))
            or ("unknown" if geom_type == "polygon" else "unknown_line"))

def merge_lines(out):
    # join line pieces of the same label that connect end to end (the source splits paths into short pieces)
    lines = collections.defaultdict(list)
    for pts, label, geom_type in out:
        if geom_type == "line":
            lines[label].append(LineString(pts))
    out = [x for x in out if x[2] != "line"]
    for label, ls in lines.items():
        for merged in shapely.get_parts(shapely.line_merge(shapely.MultiLineString(ls))):
            out.append((list(merged.coords), label, "line"))
    return out 

def extract_geometries(data, to_px):
    """(points, label, geom_type) per part; polygons drop the repeated closing point."""
    out = []
    for f in data["features"]:
        # separate geometry from properties
        g, props = f["geometry"], f["properties"]
        # separate out the omt layer, class, and subclass
        # fields for defining the vocabulary
        layer, cls, subclass = props["omt_layer"], props.get("class"), props.get("subclass")
        # skip if not line, multiline, polygon, or multipolygon
        if g["type"] not in PARTS or (layer, cls) in SKIP or (layer, "*") in SKIP:
            continue
        # extract the actual geometry based on teh dictionary defined above
        geom_type, parts = PARTS[g["type"]](g["coordinates"])
        if (layer, cls, subclass, geom_type) in SKIP_SHAPES:
            continue
        label = label_for(layer, cls, subclass, geom_type)
        is_poly = geom_type == "polygon"
        for part in parts:
            # every point except the last point (for polygons)
            # lines already don't have the last point
            # converts all lon,lat to pixel values
            pts = [to_px(lon, lat) for lon, lat in (part[:-1] if is_poly else part)]
            if len(pts) >= 2 + is_poly:
                out.append((pts, label, geom_type))

    # identical shapes can survive from the source data (e.g. two copies of a large area): keep one of each
    # the same key is jsut replaced when there's a duplicate in terms of the points
    out = list({(label, geom_type, tuple(np.round(pts, 3).ravel())): (pts, label, geom_type)
                 for pts, label, geom_type in out}.values())
    # merge short lines
    return merge_lines(out)


def make_to_px(data, png_width):
    """lon/lat -> pixels of the rendered PNG (crops are rendered at camera_zoom, 2x)."""
    zoom = data["metadata"]["camera_zoom"]
    # function from render_carto
    # convert coordinates to pixels
    # should be the same as the renderer, so it matches up
    x0, y0, x1, _ = pixel_window(data["bbox"], zoom)
    # get scaling ratio
    scale = png_width / (x1 - x0) # should be 2
    #print(f'Scale is: {scale}')

    # conversion with specific values pre-set
    def to_px(lon, lat):
        """Convert to image pixels"""
        px, py = lonlat_to_px(lon, lat, zoom)
        return (px - x0) * scale, (py - y0) * scale

    return to_px


def clip_to_image(shape, width, height):
    """The pieces of a shapely shape inside the image (0..width-1, 0..height-1), as a list."""
    if not shape.is_valid:
        shape = shape.buffer(0)   # repair self-intersecting outlines before clipping
    # clip to bounding box starting at 0,0
    clipped = clip_by_rect(shape, 0, 0, width - 1, height - 1)
    # returns the geometry attribute otherwise it jsut returns [clipped]
    # multishape vs. single shape
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
        parts = list(line_area.geoms) if line_area.geom_type == "MultiPolygon" else [line_area]
        for polygon in parts:
            if polygon.is_empty or polygon.geom_type != "Polygon":
                continue
            ring = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
            if len(ring) >= 3:
                buffered_geoms.append((ring.tolist(), label, "polygon", "line"))

    return buffered_geoms


def build_annotations(geoms, width, height, min_area=20.0, simplify_tol=2.0, min_line_length=4.0,
                      img_id=0, start_id=0):
    """COCO annotations for one image; optional 4th tuple item records the source geometry type."""
    anns = []
    # loops through all geoms in data
    for item in geoms:
        # extracts pts, label, geom_type
        pts, label, geom_type = item[:3]
        is_poly = geom_type == "polygon" # get geom type
        # clip shape accourding to width and height
        for piece in clip_to_image((Polygon if is_poly else LineString)(pts), width, height):
            if piece.geom_type != ("Polygon" if is_poly else "LineString"):
                continue   # e.g. a point or edge where a shape only touches the image border
            shape = piece.simplify(simplify_tol)
            if is_poly:
                if shape.geom_type != "Polygon" or shape.area < min_area:
                    continue
                # sort points clockwise, and closest to upper left corner
                # also drops closing point
                out = resort_corners(np.asarray(shape.exterior.coords[:-1])).reshape(-1, 2)  
            else:
                # drop small lines if less than minimum line length
                if shape.length < min_line_length:
                    continue
                out = np.asarray(shape.coords)
            # setup coco annotation format
            annotation = {
                "id": start_id + len(anns), "image_id": img_id, "iscrowd": 0,
                # add label
                "category_id": CATEGORIES[label], "geom_type": geom_type,
                # add list of pts, add shape area
                "segmentation": [out.reshape(-1).tolist()], "area": shape.area,
                # COCO bounding box [x, y, w, h] in pixels, padded 2 px
                "bbox": create_coco_bounding_box(out[:, 0], out[:, 1], width, height, bound_pad=2),
            }
            # Only buffered runs add a fourth item recording the original geometry type.
            if len(item) > 3:
                annotation["source_geom_type"] = item[3]
            anns.append(annotation)
    return anns


def config():
    a = argparse.ArgumentParser(description="Generate COCO-format data from OSM geojson crops")
    a.add_argument("--data_root", required=True, type=str, help="folder with <name>.geojson.gz and <name>.png")
    a.add_argument("--output", required=True, type=str, help="output dataset root")
    a.add_argument("--test_prefix", default="flushing", type=str, help="crops whose name starts with <test_prefix>_ form the test split; any prefix works, e.g. flushing for make_boxes.py --test-nta flushing")
    a.add_argument("--val_frac", default=0.1, type=float, help="val size as a fraction of all crops, test included (taken from non-test crops)")
    a.add_argument("--min_area", default=20.0, type=float, help="drop polygons smaller than this (px^2)")
    a.add_argument("--simplify_tol", default=2.0, type=float,
                   help="shape simplification tolerance in PNG px; 0 = off")
    a.add_argument("--min_line_length", default=4.0, type=float, help="drop line pieces shorter than this (px)")
    a.add_argument("--seed", default=0, type=int)
    a.add_argument("--line_buffer_px", default=None, type=float,
                   help="buffer radius in PNG pixels for line features; omit to keep lines unbuffered")
    args = a.parse_args()
    if args.line_buffer_px is not None and (not math.isfinite(args.line_buffer_px) or args.line_buffer_px <= 0):
        a.error("--line_buffer_px must be greater than 0 and finite; omit it to disable buffering")
    return args


def main(args):
    root, out = Path(args.data_root), Path(args.output) # load data root
    # get pngs from root path
    pngs = {f: f.with_name(f.name.replace(".geojson.gz", ".png")) for f in sorted(root.glob("*.geojson.gz"))}
    pngs = {f: p for f, p in pngs.items() if p.exists()}

    # get the test
    is_test = lambda f: f.name.startswith(f"{args.test_prefix}_")
    rest = [f for f in pngs if not is_test(f)]
    random.Random(args.seed).shuffle(rest)
    n_val = int(args.val_frac * len(pngs)) # number of validation images
    splits = {"train": rest[n_val:], "val": rest[:n_val], "test": [f for f in pngs if is_test(f)]}

    # write images + COCO json per split (category ids come from the fixed VOCAB)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    ann_id = 0
    for split, split_files in splits.items():
        (out / split).mkdir(exist_ok=True)
        coco = prepare_dict(CATEGORIES) # get labels
        # iterate through all images
        for img_id, f in enumerate(split_files):
            png, img = pngs[f], Image.open(pngs[f]) # path + opened image
            w, h = img.size # image size dimneions
            with gzip.open(f, "rt") as fh:
                data = json.load(fh)
            # crops are palette PNGs; Raster2Seq reads images with np.array(), which would give palette indices
            img.convert("RGB").save(out / split / png.name) # convert to RGB
            coco["images"].append({"file_name": png.name, "id": img_id, "width": w, "height": h})
            # get the geometry for specific crop in pixel units, along with label and geom type
            geoms = extract_geometries(data, make_to_px(data, w))
            # Keep the original line coordinates unless a buffer radius was supplied.
            if args.line_buffer_px is not None:
                geoms = buffer_line_geometries(geoms, args.line_buffer_px)
            # build full set of coco annotations, based on arguments
            # simplifies, clips, drops small geometries 
            anns = build_annotations(geoms, w, h, args.min_area, args.simplify_tol, args.min_line_length, 
                                        img_id, ann_id)
            
            # identical once clipped to the image: keep one of each
            anns = list({(a["category_id"], tuple(np.round(a["segmentation"][0], 2))): a for a in anns}.values())
            if not anns:  # drop empty crops
                coco["images"].pop()                 # remove the entry just added for it
                (out / split / png.name).unlink()    # and its saved PNG
                continue
            for k, a in enumerate(anns):  # renumber so ids stay consecutive and unique
                a["id"] = ann_id + k
            coco["annotations"] += anns
            ann_id += len(anns)
        with open(out / "annotations" / f"{split}.json", "w") as fh:
            json.dump(coco, fh, default=convert_numpy_to_python)
        print(f"{split}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    print("categories:", CATEGORIES)
    print("# of categories:", len(CATEGORIES))


if __name__ == "__main__":
    main(config())
