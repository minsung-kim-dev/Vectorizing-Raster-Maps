#!/usr/bin/env python3
"""Cut OpenMapTiles vector tiles into GeoJSON crops and render them with utils/render_carto.py
(default) or utils/render_omt.py (--style omt, the OSM OpenMapTiles style).

Make the tiles once with Planetiler, then convert them to a GeoPackage with GDAL
(GDAL clips each tile's features to the tile, so nothing is duplicated at tile edges):
    java -Xmx2g -jar planetiler.jar --download --osm-path=NewYork.osm.pbf --output=newyork.mbtiles
    ogr2ogr -f GPKG newyork_omt.gpkg newyork.mbtiles -oo ZOOM_LEVEL=14
Then:
    python osm_crops.py --gpkg newyork_omt.gpkg --boxes boxes.csv --zoom 18 [--style omt]

boxes.csv:  name,min_lon,min_lat,max_lon,max_lat
Writes <name>.geojson.gz and <name>.png per box, plus index.csv.
Data comes from zoom-14 tiles; --zoom only sets the render zoom.
With --style omt the GeoJSON also keeps house numbers and all POIs, which that style draws
(they are points, so data_preprocess/create_coco_osm.py ignores them).
"""
import argparse, csv, gzip, json, os, re, sys, time
from multiprocessing import Pool

import numpy as np
import pyogrio
import shapely
import geopandas as gpd
import pandas as pd
from PIL import Image
from shapely.geometry import mapping, box

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))
import render_carto as rc
import render_omt as ro
from render_carto import pixel_window

SOURCE_ZOOM = 14  # zoom of the tiles the GeoPackage was converted from
GPKG, LAYERS = None, []  # set per worker: GeoPackage path, layers to read
ALL_NAMES = False  # keep every name:xx translation (--all-names)
NAMES = {"name", "name_en", "name_int"}  # names the renderers can use
SKIP_LAYERS = {"housenumber"}  # left out of the GeoJSON (--style carto; omt keeps everything)
POI_CLASSES = {"railway"}  # the only poi features kept with --style carto (stations); omt keeps all
BUFFER_M = 20  # margin kept beyond the image edge,

# every color each style uses; palette PNGs always keep these exactly
STYLE_COLORS = {
    "carto": np.array(sorted({tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
                              for h in re.findall(r"#[0-9a-fA-F]{6}\b", open(rc.__file__).read())})),
    "omt": np.array(ro.style_colours()),
}

def extract(bbox):
    """Features under the image (plus buffer_px on each side), clipped, in lon/lat."""
    # the image's corners in zoom-`zoom` pixels, plus the buffer, converted to Web Mercator projection
    x0, y0, x1, y1 = gpd.GeoSeries([box(*bbox)], crs=4326).to_crs(3857).total_bounds
    area = (x0 - BUFFER_M, y0 - BUFFER_M, x1 + BUFFER_M, y1 + BUFFER_M) # buffered area
    features = []
    for layer in LAYERS:
        df = pyogrio.read_dataframe(GPKG, layer=layer, bbox=area)
        df = df.explode(index_parts=False)  # one row per shape (buildings come grouped per tile)
        df["geometry"] = df.geometry.clip_by_rect(*area)  # clip first, so copies are compared as they'll be saved
        df = df[~df.geometry.is_empty]
        cs = df.reindex(columns=["class", "subclass"]).astype(str)
        kind = cs["class"] + "/" + cs["subclass"]
        # repeated shapes of the same type
        df = df[~pd.DataFrame({"kind": kind, "shape": df.geometry.set_precision(0.01, mode="pointwise").normalize().to_wkb()}).duplicated().to_numpy()] 
        if len(df):  # some layers have no shapes in a crop
            g = shapely.make_valid(df.geometry.values); a = shapely.area(g)
            cs = df.reindex(columns=["class", "subclass"]).astype(str)
            kind = (cs["class"] + "/" + cs["subclass"]).to_numpy()
            i, j = shapely.STRtree(g).query(g, predicate="intersects")
            # i!= j -> must be two different shapes
            # checks area intersection relative to first shape
            # only compares cases where j is larger than i
            dup = ((i != j) & (kind[i] == kind[j]) & (shapely.area(shapely.intersection(g[i], g[j])) > 0.8 * a[i])
                   & ((a[i] < a[j]) | ((a[i] == a[j]) & (i > j))))
            # iterate through duplicate pairs nad keep the union
            for k, l in zip(i[dup], j[dup]):  # merge each piece into the shape it lies in, keeping any part sticking out
                # store in the larger shape
                g[l] = shapely.union(g[l], g[k])
            # update the geometry column with this
            df["geometry"] = gpd.GeoSeries(g, index=df.index, crs=df.crs)
            df = df.iloc[np.setdiff1d(np.arange(len(df)), i[dup])]
        
        if layer == "poi" and POI_CLASSES is not None:
            df = df[df["class"].isin(POI_CLASSES)]
        geoms = df.geometry.to_crs(4326)  # already clipped
        geoms = shapely.transform(geoms.values, lambda xy: np.round(xy, 7))

        for g, props in zip(geoms, df.drop(columns=["geometry", "mvt_id"]).to_dict("records")):
            # skip empty clips and mixed geometry collections
            if g.is_empty or g.geom_type == "GeometryCollection":
                continue
            # keep set properties; drop name translations the renderers don't use (unless --all-names)
            props = {k: v for k, v in props.items() if v is not None and v == v
                     and (ALL_NAMES or not k.startswith("name") or k in NAMES)}
            features.append({"type": "Feature", "geometry": mapping(g), "properties": {"omt_layer": layer, **props}})
    return features


def shrink_png(path, style_colors):
    """Rewrite a render as a 256-color palette PNG (about 60% smaller): all of
    the style's colors stay exact, the rest of the palette comes from the image itself."""
    # get image as RGB
    im = Image.open(path).convert("RGB")
    # how many palette slots are free after the style colors take theirs
    n = 256 - len(style_colors)
    # get list of extra 'n' colors (colors are represented by r g b)
    # pick the n extra colors that best represent the image (split by middle)
    extra = im.quantize(n, method=Image.Quantize.MEDIANCUT).getpalette()[:3 * n]
    # add extra
    palette = np.vstack([style_colors, np.array(extra).reshape(-1, 3)]).astype(np.int32)
    # get unique rgb colors
    rgb = np.asarray(im).astype(np.int32)
    cols, position = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True)  # distinct [r, g, b] rows
    # based on smallest squared distance
    nearest = ((cols[:, None] - palette[None]) ** 2).sum(-1).argmin(1)
    # produce same image size but with palette colors (slightly smaller)
    out = Image.fromarray(nearest[position].reshape(rgb.shape[:2]).astype(np.uint8), "P")
    out.putpalette(palette.astype(np.uint8).ravel().tolist())
    out.save(path)


def make_crop(job):
    name, bbox, a = job
    # set up file names
    gj, png = (os.path.join(a.out, name + ext) for ext in (".geojson.gz", ".png"))
    try:
        t = time.time()
        # extract features for the bbox and default zoom level of tile
        features = extract(bbox)
        with gzip.open(gj, "wt", encoding="utf-8") as fh:
            # create a geojson file
            json.dump({"type": "FeatureCollection", "bbox": bbox,
                       "metadata": {"name": name, "source_zoom": SOURCE_ZOOM, "camera_zoom": a.zoom,
                                    "pixel_window": list(pixel_window(bbox, a.zoom)),
                                    "attribution": "© OpenMapTiles © OpenStreetMap contributors"},
                       "features": features}, fh, ensure_ascii=False, separators=(",", ":"))

        # render with style from render_rc
        renderer = ro if a.style == "omt" else rc
        # formatting from render_rc
        # render with the geojson file as the png at a specifi zoom and scale
        # scale increases precision/qualiy of image
        size = "x".join(map(str, renderer.render(gj, png, a.zoom, a.scale)["size"]))
        # use palette colors (to reduce file size)
        if not a.rgb:
            shrink_png(png, STYLE_COLORS[a.style])  # 256-color palette PNG
        row = dict(name=name, features=len(features), size=size, zoom=a.zoom, source_zoom=SOURCE_ZOOM,
                   min_lon=bbox[0], min_lat=bbox[1], max_lon=bbox[2], max_lat=bbox[3])
        return row, f"{name}: {len(features)} features {size} ({time.time() - t:.1f}s)"
    except Exception as e:  # one bad crop shouldn't stop the run
        return None, f"{name}: ERROR {e!r}"


def init_worker(path, all_names, style):
    global GPKG, LAYERS, ALL_NAMES, SKIP_LAYERS, POI_CLASSES, NAMES
    GPKG, ALL_NAMES = path, all_names
    if style == "omt":  # the OpenMapTiles style draws house numbers and POI labels, using name:en etc.
        SKIP_LAYERS, POI_CLASSES = set(), None
        NAMES = NAMES | {"name:en", "name:latin", "name:nonlatin"}
    LAYERS = [layer for layer, _ in pyogrio.list_layers(path) if layer not in SKIP_LAYERS]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpkg", required=True, help="OpenMapTiles GeoPackage (see above)")
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--zoom", type=int, default=18, help="render zoom (default 18)")
    ap.add_argument("--scale", type=int, default=2, help="2 = high-DPI images")
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

    jobs = [(b["name"], [float(b[k]) for k in ("min_lon", "min_lat", "max_lon", "max_lat")], a)
            for b in csv.DictReader(open(a.boxes, newline=""))]

    index = []
    with Pool(a.workers, initializer=init_worker, initargs=(a.gpkg, a.all_names, a.style)) as pool:
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
