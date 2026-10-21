#!/usr/bin/env python3
"""Cut OpenMapTiles vector tiles into GeoJSON crops and render them with render_carto.py.

Make the tiles once with Planetiler:
    java -Xmx2g -jar planetiler.jar --download --osm-path=NewYork.osm.pbf --output=newyork.mbtiles
Then:
    pip install mapbox-vector-tile shapely numpy matplotlib pillow
    python osm_crops.py --mbtiles newyork.mbtiles --boxes boxes.csv --zoom 18

boxes.csv:  name,min_lon,min_lat,max_lon,max_lat
Writes <name>.geojson.gz and <name>.png per box, plus index.csv.
Data comes from the highest tile zoom (14); --zoom only sets the render zoom.
"""
import argparse, csv, gzip, json, os, re, sqlite3, sys, time
from functools import lru_cache
from multiprocessing import Pool

import mapbox_vector_tile
import numpy as np
import shapely
from PIL import Image
from shapely.geometry import mapping, shape

sys.path.insert(0, "data/osm_carto_compare")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_carto as rc
from render_carto import lonlat_to_px, pixel_window

DB = None  # one read-only connection per worker process
ALL_NAMES = False  # keep every name:xx translation (--all-names)
NAMES = {"name", "name_en", "name_int"}  # names render_carto.py can use
SKIP_LAYERS = {"housenumber"}  # left out of the GeoJSON
POI_CLASSES = {"railway"}  # the only poi features kept (stations, drawn by render_carto.py)
# every colour written in render_carto.py; palette PNGs always keep these exactly
STYLE_COLOURS = np.array(sorted({tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
                                 for h in re.findall(r"#[0-9a-fA-F]{6}\b", open(rc.__file__).read())}))


def px_to_lonlat(x, y, z):
    """Inverse of render_carto.lonlat_to_px; works on numbers and numpy arrays."""
    n = 256 * 2 ** z
    return x / n * 360 - 180, np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))


def tile_to_lonlat(xy, z, tx, ty):
    """Points in 0..1 units of tile (tx, ty) -> [lon, lat] array, rounded to 7 decimals (~1 cm)."""
    lon, lat = px_to_lonlat((tx + xy[:, 0]) * 256, (ty + xy[:, 1]) * 256, z)
    return np.round(np.column_stack([lon, lat]), 7)


def open_mbtiles(path):
    if not os.path.isfile(path):
        sys.exit(f"[error] {path} not found")
    db = sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)  # read-only: never creates a file
    try:
        db.execute("SELECT 1 FROM tiles LIMIT 1")
    except sqlite3.DatabaseError:
        sys.exit(f"[error] {path} is not a finished .mbtiles file")
    return db


@lru_cache(maxsize=64)
def load_tile(z, x, y):
    """Decode a tile once: (layers, properties, geometries in 0..1 tile units, their bounds)."""
    row = DB.execute("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                     (z, x, 2 ** z - 1 - y)).fetchone()  # MBTiles rows count from the bottom
    if row is None:
        return None
    data = gzip.decompress(row[0]) if row[0][:2] == b"\x1f\x8b" else row[0]
    layers, props, geoms = [], [], []
    for layer, d in mapbox_vector_tile.decode(data, default_options={"y_coord_down": True}).items():
        if layer in SKIP_LAYERS:
            continue
        scale = d.get("extent", 4096)
        for f in d["features"]:
            if layer == "poi" and f["properties"].get("class") not in POI_CLASSES:
                continue
            g = shapely.transform(shape(f["geometry"]), lambda xy: xy / scale)
            if g.geom_type == "Point" and not (0 <= g.x < 1 and 0 <= g.y < 1):
                continue  # duplicate of a neighbouring tile's point
            layers.append(layer)
            props.append({k: v for k, v in f["properties"].items()
                          if ALL_NAMES or not k.startswith("name") or k in NAMES})
            geoms.append(g)
    geoms = np.array(geoms, dtype=object)
    return layers, props, geoms, shapely.bounds(geoms).reshape(-1, 4)


def extract(z, bbox, zoom, buffer_px):
    """Features under the image (plus buffer_px on each side), clipped, in lon/lat."""
    x0, y0, x1, y1 = pixel_window(bbox, zoom)
    k = 2 ** (zoom - z) * 256  # render pixels per tile
    r0 = ((x0 - buffer_px) / k, (y0 - buffer_px) / k, (x1 + buffer_px) / k, (y1 + buffer_px) / k)
    features = []
    for tx in range(int(r0[0]), int(r0[2]) + 1):
        for ty in range(int(r0[1]), int(r0[3]) + 1):
            tile = load_tile(z, tx, ty)
            if tile is None:
                continue
            layers, props, geoms, b = tile
            r = (r0[0] - tx, r0[1] - ty, r0[2] - tx, r0[3] - ty)  # box in this tile's units
            hit = np.nonzero((b[:, 0] <= r[2]) & (b[:, 2] >= r[0]) & (b[:, 1] <= r[3]) & (b[:, 3] >= r[1]))[0]
            clipped = shapely.clip_by_rect(geoms[hit], *r)
            for i, g in zip(hit, shapely.transform(clipped, lambda xy: tile_to_lonlat(xy, z, tx, ty))):
                if not g.is_empty and g.geom_type != "GeometryCollection":
                    features.append({"type": "Feature", "geometry": mapping(g),
                                     "properties": {"omt_layer": layers[i], **props[i]}})
    return features


LUT = np.zeros(2 ** 24, np.uint8)  # colour -> palette index, reused for every image
SEEN = np.zeros(2 ** 24, bool)  # scratch space for finding an image's distinct colours


def shrink_png(path, colours):
    """Rewrite a render as a palette PNG with `colours` colours (about 60% smaller): all of
    render_carto's colours stay exact, the rest of the palette comes from the image itself."""
    im = Image.open(path).convert("RGB")
    n = colours - len(STYLE_COLOURS)
    extra = im.quantize(n, method=Image.Quantize.MEDIANCUT).getpalette()[:3 * n]
    palette = np.vstack([STYLE_COLOURS, np.array(extra).reshape(-1, 3)]).astype(np.int32)
    rgb = np.asarray(im).astype(np.int32)
    keys = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]  # one number per colour
    SEEN[keys] = True
    present = np.flatnonzero(SEEN)  # the image's distinct colours
    SEEN[present] = False
    cols = np.stack([present >> 16, (present >> 8) & 255, present & 255], axis=1)
    LUT[present] = ((cols[:, None] - palette[None]) ** 2).sum(-1).argmin(1)  # nearest palette colour
    out = Image.fromarray(LUT[keys], "P")
    out.putpalette(palette.astype(np.uint8).ravel().tolist())
    out.save(path)


def make_crop(job):
    name, bbox, a, z = job
    gj, png = (os.path.join(a.out, name + ext) for ext in (".geojson.gz", ".png"))
    try:
        t = time.time()
        features = extract(z, bbox, a.zoom, a.buffer_px)
        with gzip.open(gj, "wt", encoding="utf-8") as fh:
            json.dump({"type": "FeatureCollection", "bbox": bbox,
                       "metadata": {"name": name, "source_zoom": z, "camera_zoom": a.zoom,
                                    "pixel_window": list(pixel_window(bbox, a.zoom)),
                                    "attribution": "© OpenMapTiles © OpenStreetMap contributors"},
                       "features": features}, fh, ensure_ascii=False, separators=(",", ":"))
        size = ""
        if not a.no_images:
            size = "x".join(map(str, rc.render(gj, png, a.zoom, a.scale)["size"]))
            if a.colours:
                shrink_png(png, a.colours)
        row = dict(name=name, features=len(features), size=size, zoom=a.zoom, source_zoom=z,
                   min_lon=bbox[0], min_lat=bbox[1], max_lon=bbox[2], max_lat=bbox[3])
        return row, f"{name}: {len(features)} features {size} ({time.time() - t:.1f}s)"
    except Exception as e:  # one bad crop shouldn't stop the run
        return None, f"{name}: ERROR {e!r}"


def init_worker(path, all_names):
    global DB, ALL_NAMES
    DB, ALL_NAMES = open_mbtiles(path), all_names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mbtiles", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--zoom", type=int, default=18, help="render zoom (default 18)")
    ap.add_argument("--scale", type=int, default=1, help="2 = high-DPI images")
    ap.add_argument("--buffer-px", type=int, default=32, help="data kept beyond the image edge")
    ap.add_argument("--out", default="crops")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--no-images", action="store_true", help="only write GeoJSON")
    ap.add_argument("--overwrite", action="store_true", help="redo existing crops")
    ap.add_argument("--all-names", action="store_true", help="keep names in every language")
    ap.add_argument("--colours", type=int, default=256,
                    help="PNG palette size (default 256, max 256); 0 = full-colour PNGs")
    a = ap.parse_args()
    if a.colours and not len(STYLE_COLOURS) < a.colours <= 256:
        sys.exit(f"[error] --colours must be 0 or {len(STYLE_COLOURS) + 1}-256")
    os.makedirs(a.out, exist_ok=True)

    z = min(open_mbtiles(a.mbtiles).execute("SELECT MAX(zoom_level) FROM tiles").fetchone()[0], a.zoom)
    jobs = []
    for b in csv.DictReader(open(a.boxes, newline="")):
        if a.overwrite or not os.path.exists(os.path.join(a.out, b["name"] + ".geojson.gz")):
            jobs.append((b["name"], [float(b[k]) for k in ("min_lon", "min_lat", "max_lon", "max_lat")], a, z))
    # neighbours share tiles: sorting keeps them in the same worker, so each tile is decoded once
    jobs.sort(key=lambda j: tuple(int(v // 256) for v in lonlat_to_px(j[1][0], j[1][3], z)))

    index = []
    with Pool(a.workers, initializer=init_worker, initargs=(a.mbtiles, a.all_names)) as pool:
        for n, (row, msg) in enumerate(pool.imap_unordered(make_crop, jobs, chunksize=8), 1):
            print(f"[{n}/{len(jobs)}] {msg}")
            index += [row] if row else []
    if index:
        with open(os.path.join(a.out, "index.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(index[0]))
            w.writeheader()
            w.writerows(index)
    print(f"done: {len(index)} written, {len(jobs) - len(index)} failed -> {a.out}/")


if __name__ == "__main__":
    main()
