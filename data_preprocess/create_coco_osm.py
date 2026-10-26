"""Convert OSM crops (<name>.geojson.gz + <name>.png) into Raster2Seq's COCO layout.

Output (readable by Raster2Seq datasets/poly_data.py -> ConvertToCocoDict):
    <output>/{train,val,test}/<name>.png
    <output>/annotations/{train,val,test}.json

Each OpenMapTiles (layer, class) is mapped to a label from the fixed vocabulary VOCAB
(e.g. "street", "building", "park"); anything not in VOCAB is labelled "unknown" (polygons)
or "unknown_line" (lines).
Polygons are stored as closed rings (closing point dropped, corners resorted).
LineStrings are stored start -> finish, are NOT resorted, and carry "geom_type": "line"
(polygons: "polygon") so the loader can skip resort_corners for them.
"""

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import LineString, Polygon

REPO = Path(__file__).resolve().parents[1]
R2S = REPO.parent / "Raster2Seq"
sys.path[:0] = [str(REPO), str(R2S), str(R2S / "data_preprocess" / "waffle")]

from create_coco_waffle_benchmark import convert_numpy_to_python, create_coco_bounding_box, prepare_dict  # noqa: E402
from util.poly_ops import resort_corners  # noqa: E402
from utils.render_carto import lonlat_to_px, pixel_window  # noqa: E402

# GeoJSON geometry type -> (geom_type, list of point lists); polygons keep only the outer ring
PARTS = {
    "Polygon": lambda c: ("polygon", [c[0]]),
    "MultiPolygon": lambda c: ("polygon", [p[0] for p in c]),
    "LineString": lambda c: ("line", [c]),
    "MultiLineString": lambda c: ("line", c),
}
# label -> OpenMapTiles keys: (layer, class), (layer, class, geom_type), or (layer, "*") for any class.
# Anything not listed (and not in SKIP) is labelled "unknown" / "unknown_line" (e.g. military areas / ferries).
# Each label holds only polygons or only lines, since Raster2Seq treats every shape as closed.
VOCAB = {
    "building":     [("building", "*")],
    "highway":      [("transportation", c) for c in ("motorway", "motorway_construction", "trunk")],
    "street":       [("transportation", c) for c in ("primary", "secondary", "tertiary", "minor", "service", "busway",
                                                     "secondary_construction", "tertiary_construction",
                                                     "minor_construction", "service_construction")],
    "path":         [("transportation", "path", "line"), ("transportation", "track"),
                     ("transportation", "path_construction")],
    "plaza":        [("transportation", "path", "polygon")],  # pedestrian areas
    "railway":      [("transportation", "rail"), ("transportation", "transit")],
    "rail_yard":    [("landuse", "railway")],
    "bridge":       [("transportation", "bridge")],
    "pier":         [("transportation", "pier", "line")],
    "pier_area":    [("transportation", "pier", "polygon")],
    "water":        [("water", c) for c in ("lake", "pond", "river", "ocean", "dock")],
    "pool":         [("water", "swimming_pool")],
    "waterway":     [("waterway", "*")],
    "park":         [("park", "*"), ("landuse", "recreation_ground"), ("landuse", "zoo"), ("landuse", "theme_park")],
    "playground":   [("landuse", "playground")],
    "sports_field": [("landuse", c) for c in ("pitch", "track", "stadium")],
    "grass":        [("landcover", "grass"), ("landcover", "farmland")],
    "woods":        [("landcover", "wood")],
    "wetland":      [("landcover", "wetland")],
    "bare_ground":  [("landcover", "sand"), ("landcover", "rock")],
    "school":       [("landuse", c) for c in ("school", "kindergarten", "education", "college", "university")],
    "hospital":     [("landuse", "hospital")],
    "civic":        [("landuse", "library"), ("landuse", "bus_station")],
    "cemetery":     [("landuse", "cemetery")],
    "residential":  [("landuse", "residential")],
    "commercial":   [("landuse", "commercial")],
    "retail":       [("landuse", "retail")],
    "industrial":   [("landuse", "industrial")],
    "airport":      [("aeroway", c) for c in ("aerodrome", "apron", "helipad")],
    "runway":       [("aeroway", "runway", "line"), ("aeroway", "taxiway", "line")],
    "runway_area":  [("aeroway", "runway", "polygon")],
    "unknown":      [],  # fallback for polygons not listed above
    "unknown_line": [],  # fallback for lines not listed above
}
LABELS = {key: label for label, keys in VOCAB.items() for key in keys}
CATEGORIES = {label: i for i, label in enumerate(VOCAB)}
SKIP = {  # dropped entirely: (layer, class) or (layer, "*")
    ("transportation_name", "*"), ("water_name", "*"),  # copies of other features, only used to place names
    ("boundary", "*"), ("landuse", "neighbourhood"),      # administrative, not drawn as visible features
}


def label_for(layer, cls, geom_type):
    """Vocabulary label for an OpenMapTiles feature (most specific key wins), else "unknown"/"unknown_line"."""
    return (LABELS.get((layer, cls, geom_type)) or LABELS.get((layer, cls)) or LABELS.get((layer, "*"))
            or ("unknown" if geom_type == "polygon" else "unknown_line"))


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


def build_annotations(geoms, width, height, min_area=20.0, simplify_tol=1.0, img_id=0, start_id=0):
    """COCO annotations for one image from extract_geometries output.
    Shapes are simplified with tolerance simplify_tol px."""
    anns = []
    for pts, label, geom_type in geoms:
        is_poly = geom_type == "polygon"
        # uses shapely method simpify
        shape = (Polygon if is_poly else LineString)(pts).simplify(simplify_tol)
        if is_poly:
            if shape.geom_type != "Polygon" or shape.area < min_area:
                continue
            pts = resort_corners(np.asarray(shape.exterior.coords[:-1])).reshape(-1, 2)  # drop closing point
        else:
            pts = np.asarray(shape.coords)
        anns.append({
            "id": start_id + len(anns), "image_id": img_id, "iscrowd": 0,
            "category_id": CATEGORIES[label], "geom_type": geom_type,
            "segmentation": [pts.reshape(-1).tolist()], "area": shape.area,
            "bbox": create_coco_bounding_box(pts[:, 0], pts[:, 1], width, height, bound_pad=2),
        })
    return anns


def config():
    a = argparse.ArgumentParser(description="Generate COCO-format data from OSM geojson crops")
    a.add_argument("--data_root", required=True, type=str, help="folder with <name>.geojson.gz and <name>.png")
    a.add_argument("--output", required=True, type=str, help="output dataset root")
    a.add_argument("--test_borough", default="queens", type=str, help="crops named <test_borough>_* form the test split")
    a.add_argument("--val_frac", default=0.1, type=float, help="fraction of non-test crops used for val")
    a.add_argument("--min_area", default=20.0, type=float, help="drop polygons smaller than this (px^2)")
    a.add_argument("--simplify_tol", default=1.0, type=float, help="shape simplification tolerance (px); 0 = off")
    a.add_argument("--seed", default=0, type=int)
    return a.parse_args()


def main(args):
    root, out = Path(args.data_root), Path(args.output)
    pngs = {f: f.with_name(f.name.replace(".geojson.gz", ".png")) for f in sorted(root.glob("*.geojson.gz"))}
    pngs = {f: p for f, p in pngs.items() if p.exists()}

    is_test = lambda f: f.name.startswith(f"{args.test_borough}_")
    rest = [f for f in pngs if not is_test(f)]
    random.Random(args.seed).shuffle(rest)
    n_val = int(args.val_frac * len(rest))
    splits = {"train": rest[n_val:], "val": rest[:n_val], "test": [f for f in pngs if is_test(f)]}

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
            coco["images"].append({"file_name": png.name, "id": img_id, "width": w, "height": h})
            geoms = extract_geometries(data, make_to_px(data, w))
            anns = build_annotations(geoms, w, h, args.min_area, args.simplify_tol, img_id, ann_id)
            coco["annotations"] += anns
            ann_id += len(anns)
        with open(out / "annotations" / f"{split}.json", "w") as fh:
            json.dump(coco, fh, default=convert_numpy_to_python)
        print(f"{split}: {len(split_files)} images, {len(coco['annotations'])} annotations")

    print("categories:", CATEGORIES)


if __name__ == "__main__":
    main(config())
