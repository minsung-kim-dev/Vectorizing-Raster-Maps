#!/usr/bin/env python3
"""Cut OpenMapTiles vector tiles into GeoJSON crops and render them with utils/render_carto.py
(default) or utils/render_omt.py (--style omt, the OSM OpenMapTiles style).

Make the tiles once with Planetiler:
    java -Xmx2g -jar planetiler.jar --download --osm-path=NewYork.osm.pbf --output=newyork.mbtiles
Then:
    pip install mapbox-vector-tile shapely numpy matplotlib pillow
    python osm_crops.py --mbtiles newyork.mbtiles --boxes boxes.csv --zoom 18 [--style omt]

boxes.csv:  name,min_lon,min_lat,max_lon,max_lat
Writes <name>.geojson.gz and <name>.png per box, plus index.csv.
Data comes from the highest tile zoom (14); --zoom only sets the render zoom.
With --style omt the GeoJSON also keeps house numbers and all POIs, which that style draws
(they are points, so data_preprocess/create_coco_osm.py ignores them).
"""
import argparse, csv, gzip, json, os, re, sqlite3, sys, time
from functools import lru_cache
from multiprocessing import Pool

import mapbox_vector_tile
import numpy as np
import shapely
from PIL import Image
from shapely.geometry import mapping, shape

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))
import render_carto as rc
import render_omt as ro
from render_carto import lonlat_to_px, pixel_window

DB = None  # one read-only connection per worker process
ALL_NAMES = False  # keep every name:xx translation (--all-names)
NAMES = {"name", "name_en", "name_int"}  # names the renderers can use
SKIP_LAYERS = {"housenumber"}  # left out of the GeoJSON (--style carto; omt keeps everything)
POI_CLASSES = {"railway"}  # the only poi features kept with --style carto (stations); omt keeps all
ZOOM=18
# every color each style uses; palette PNGs always keep these exactly
STYLE_COLORS = {
    "carto": np.array(sorted({tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
                              for h in re.findall(r"#[0-9a-fA-F]{6}\b", open(rc.__file__).read())})),
    "omt": np.array(ro.STYLE_COLORS()),
}


def px_to_lonlat(x, y, z):
    """Inverse of render_carto.lonlat_to_px; works on numbers and numpy arrays."""
    # source:
    # https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    # n = 2 ^ zoom
    # lon_deg = xtile / n * 360.0 - 180.0
    # lat_rad = arctan(sinh(π * (1 - 2 * ytile / n)))
    # lat_deg = lat_rad * 180.0 / π`
    n = 256 * 2 ** z
    return x / n * 360 - 180, np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))


def tile_to_lonlat(xy, z, tx, ty):
    """Points in 0..1 units of tile (tx, ty) -> [lon, lat] array, rounded to 7 decimals (~1 cm)."""
    #  xy is a point's position inside its tile, from 0 to 1
    lon, lat = px_to_lonlat((tx + xy[:, 0]) * 256, (ty + xy[:, 1]) * 256, z)
    # round to 7
    return np.round(np.column_stack([lon, lat]), 7)


def open_mbtiles(path):
    """Load MBTiles"""
    if not os.path.isfile(path):
        sys.exit(f"[error] {path} not found")
    db = sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)  # read-only: never creates a file
    try:
        db.execute("SELECT 1 FROM tiles LIMIT 1")
    except sqlite3.DatabaseError:
        sys.exit(f"[error] {path} is not a finished .mbtiles file")
    return db

# tile cache
@lru_cache(maxsize=64)
def load_tile(z, x, y):
    """Decode a tile once: (layers, properties, geometries in 0..1 tile units, their bounds)."""
    # https://github.com/mapbox/mbtiles-spec/blob/master/1.3/spec.md
    # Note that in the TMS tiling scheme, the Y axis is reversed from the "XYZ" coordinate system commonly 
    # used in the URLs to request individual tiles, so the tile commonly referred to as 11/327/791 is 
    # inserted as zoom_level 11, tile_column 327, and tile_row 1256, since 1256 is 2^11 - 1 - 791.

    row = DB.execute("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                     (z, x, 2 ** z - 1 - y)).fetchone()  # MBTiles rows count from the bottom
    if row is None:
        return None
    data = gzip.decompress(row[0]) if row[0][:2] == b"\x1f\x8b" else row[0]
    layers, props, geoms = [], [], []
    # load tile
    for layer, d in mapbox_vector_tile.decode(data, default_options={"y_coord_down": True}).items():
        # for each layer, skip it if it is in ski;
        if layer in SKIP_LAYERS:
            continue
        scale = d.get("extent", 4096)
        for f in d["features"]:
            # skip poi
            if layer == "poi" and POI_CLASSES is not None and f["properties"].get("class") not in POI_CLASSES:
                continue
            # normalizes xy
            g = shapely.transform(shape(f["geometry"]), lambda xy: xy / scale)
            if g.geom_type == "Point" and not (0 <= g.x < 1 and 0 <= g.y < 1):
                continue  # duplicate of a neighbouring tile's point
            # add layer to list of layers
            layers.append(layer)
            # add properties 
            # drop names if ALL_NAMES is false or if it is not in NAMES (e.g., nonstandard)
            props.append({k: v for k, v in f["properties"].items()
                          if ALL_NAMES or not k.startswith("name") or k in NAMES})
            # append geom
            geoms.append(g)
    geoms = np.array(geoms, dtype=object)
    return layers, props, geoms, shapely.bounds(geoms).reshape(-1, 4)


def extract(z, bbox, zoom, buffer_px):
    """Features under the image (plus buffer_px on each side), clipped, in lon/lat."""
    # pixels based on a world map
    x0, y0, x1, y1 = pixel_window(bbox, zoom) # function from render_carto, get points
    # data comes from zoom=14; we are showing images at zoom 18
    k = 2 ** (zoom - z) * 256  # pixels per data tile at the render zoom (4096 for 18 vs 14)
    # generate a bounding box in pixel units/tiles
    # needed to load the tiles
    r0 = ((x0 - buffer_px) / k, (y0 - buffer_px) / k, (x1 + buffer_px) / k, (y1 + buffer_px) / k)
    features = []
    # iterate through all posisble tiles based on x,y positions
    for tx in range(int(r0[0]), int(r0[2]) + 1):
        for ty in range(int(r0[1]), int(r0[3]) + 1):
            # load map tile from database
            tile = load_tile(z, tx, ty)
            if tile is None:
                continue
            # get layers, properties, geometries from tile
            layers, props, geoms, b = tile
            # compute bounding box again in tile units (non-integers)
            r = (r0[0] - tx, r0[1] - ty, r0[2] - tx, r0[3] - ty)  # box in this tile's units
            # compare bounding box per-feature to tile bounding box
            # for efficiency, get indices of shapes that meet these conditions
            hit = np.nonzero((b[:, 0] <= r[2]) & (b[:, 2] >= r[0]) & (b[:, 1] <= r[3]) & (b[:, 3] >= r[1]))[0]
            # then clip, *r unpackes the bounding box
            clipped = shapely.clip_by_rect(geoms[hit], *r)
            # convert tile units to lonlat? 
            for i, g in zip(hit, shapely.transform(clipped, lambda xy: tile_to_lonlat(xy, z, tx, ty))):
                # drop non-polygons
                if not g.is_empty and g.geom_type != "GeometryCollection":
                    features.append({"type": "Feature", "geometry": mapping(g),
                                     "properties": {"omt_layer": layers[i], **props[i]}})
    return features


LUT = np.zeros(2 ** 24, np.uint8)  # color -> palette index, reused for every image
SEEN = np.zeros(2 ** 24, bool)  # scratch space for finding an image's distinct colors

def shrink_png(path, colors, STYLE_COLORS):
    """Rewrite a render as a palette PNG with `colors` colors (about 60% smaller): all of
    the style's colors stay exact, the rest of the palette comes from the image itself."""
    # get image as RGB
    im = Image.open(path).convert("RGB")
    # how many palette slots are free after the style colors take theirs
    n = colors - len(STYLE_COLORS)
    # get list of extra 'n' colors (colors are represented by r g b)
    # pick the n extra colors that best represent the image (split by middle)
    extra = im.quantize(n, method=Image.Quantize.MEDIANCUT).getpalette()[:3 * n]
    # add extra
    palette = np.vstack([STYLE_COLORS, np.array(extra).reshape(-1, 3)]).astype(np.int32)
    # get unique rgb colors
    rgb = np.asarray(im).astype(np.int32)
    cols, position = np.unique(rgb.reshape(-1, 3), axis=0, return_position=True)  # distinct [r, g, b] rows
    # based on smallest squared distance
    nearest = ((cols[:, None] - palette[None]) ** 2).sum(-1).argmin(1)
    # produce same image size but with palette colors (slightly smaller)
    out = Image.fromarray(nearest[position].reshape(rgb.shape[:2]).astype(np.uint8), "P")
    out.putpalette(palette.astype(np.uint8).ravel().tolist())
    out.save(path)


def make_crop(job):
    name, bbox, a, z = job # get bbox, for file name
    # set up file names
    gj, png = (os.path.join(a.out, name + ext) for ext in (".geojson.gz", ".png"))
    try:
        t = time.time()
        # extract efatures for the bbox and default zoom level of tile
        features = extract(z, bbox, a.zoom, a.buffer_px)
        with gzip.open(gj, "wt", encoding="utf-8") as fh:
            # great a geojson file
            json.dump({"type": "FeatureCollection", "bbox": bbox,
                       "metadata": {"name": name, "source_zoom": z, "camera_zoom": a.zoom,
                                    "pixel_window": list(pixel_window(bbox, a.zoom)),
                                    "attribution": "© OpenMapTiles © OpenStreetMap contributors"},
                       "features": features}, fh, ensure_ascii=False, separators=(",", ":"))
        size = ""
        # render with style from render_rc
        renderer = ro if a.style == "omt" else rc
        # formatting from render_rc
        # render with the geojson file as the png at a specifi zoom and scale
        # scale increases precision/qualiy of image
        size = "x".join(map(str, renderer.render(gj, png, a.zoom, a.scale)["size"]))
        # use palette colors (to reduce file size)
        if not a.rgb:
            shrink_png(png, STYLE_COLORS[a.style])  # 256-color palette PNG
        row = dict(name=name, features=len(features), size=size, zoom=a.zoom, source_zoom=z,
                   min_lon=bbox[0], min_lat=bbox[1], max_lon=bbox[2], max_lat=bbox[3])
        return row, f"{name}: {len(features)} features {size} ({time.time() - t:.1f}s)"
    except Exception as e:  # one bad crop shouldn't stop the run
        return None, f"{name}: ERROR {e!r}"


def init_worker(path, all_names, style):
    global DB, ALL_NAMES, SKIP_LAYERS, POI_CLASSES, NAMES
    DB, ALL_NAMES = open_mbtiles(path), all_names
    if style == "omt":  # the OpenMapTiles style draws house numbers and POI labels, using name:en etc.
        SKIP_LAYERS, POI_CLASSES = set(), None
        NAMES = NAMES | {"name:en", "name:latin", "name:nonlatin"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mbtiles", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--zoom", type=int, default=18, help="render zoom (default 18)")
    ap.add_argument("--scale", type=int, default=1, help="2 = high-DPI images")
    ap.add_argument("--buffer-px", type=int, default=32, help="data kept beyond the image edge")
    ap.add_argument("--out", default="crops")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--all-names", action="store_true", help="keep names in every language")
    ap.add_argument("--rgb", action="store_true", help="save full RGB PNGs instead of 256-color palette PNGs")
    ap.add_argument("--style", choices=["carto", "omt"], default="carto",
                    help="carto = utils/render_carto.py (default); omt = utils/render_omt.py, the OSM "
                         "OpenMapTiles style (also keeps house numbers and all POIs in the GeoJSON)")
    a = ap.parse_args()

    # initialize
    os.makedirs(a.out, exist_ok=True)

    z = min(open_mbtiles(a.mbtiles).execute("SELECT MAX(zoom_level) FROM tiles").fetchone()[0], a.zoom)
    jobs = []
    for b in csv.DictReader(open(a.boxes, newline="")):
        jobs.append((b["name"], [float(b[k]) for k in ("min_lon", "min_lat", "max_lon", "max_lat")], a, z))
    # neighbors share tiles: sorting keeps them in the same worker, so each tile is decoded once
    # sort by tile containing top-left corner
    jobs.sort(key=lambda j: tuple(int(v // 256) for v in lonlat_to_px(j[1][0], j[1][3], z)))

    index = []
    with Pool(a.workers, initializer=init_worker, initargs=(a.mbtiles, a.all_names, a.style)) as pool:
        # for each row in list of jobs, make crop
        for n, (row, msg) in enumerate(pool.imap_unordered(make_crop, jobs, chunksize=8), 1):
            print(f"[{n}/{len(jobs)}] {msg}")
            index += [row] if row else []
    if index:
        # write summary csv
        with open(os.path.join(a.out, "index.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(index[0]))
            w.writeheader()
            w.writerows(index)
    print(f"done: {len(index)} written, {len(jobs) - len(index)} failed -> {a.out}/")


if __name__ == "__main__":
    main()
