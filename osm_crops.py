#!/usr/bin/env python3
"""Cut an OpenStreetMap extract into OpenMapTiles-schema GeoJSON crops and render
each one with render_carto.py (same folder) in its OSM Carto approximation.

For every bounding box it writes
  <name>.geojson.gz  the vector data, in the same format render_carto.py and
                     compare_osm.py read (FeatureCollection with "bbox",
                     "metadata.source_zoom" and features tagged omt_layer /
                     class / subclass)
  <name>.png         render_carto.render() of that file, pixel-aligned to the
                     256px OSM tile grid at the box's zoom
and an index.csv describing every crop. The GeoJSON can be fed straight to
compare_osm.py to score it against tile.openstreetmap.org.

Pipeline
  1. download a .osm.pbf extract once (e.g. from https://download.geofabrik.de/)
  2. convert it once to a GeoPackage (spatially indexed -> fast bbox queries)
  3. per box: query, translate OSM tags to OpenMapTiles layers, clip to the
     pixel window plus a buffer, write GeoJSON, render it

Usage:
    pip install geopandas pyogrio shapely matplotlib pandas
    python osm_crops.py --pbf-url https://download.geofabrik.de/europe/monaco-latest.osm.pbf \\
        --boxes boxes.csv --zoom 16
    python osm_crops.py --pbf my_region.osm.pbf --boxes boxes.csv --zoom 15 --scale 2

boxes.csv (WGS84 lon/lat; the zoom column is optional and overrides --zoom):
    name,min_lon,min_lat,max_lon,max_lat,zoom
    harbor,7.4180,43.7300,7.4300,43.7380,17

Raw OSM has no sea polygons (only coastline lines). For coastal areas pass
--water with water-polygons-split-4326 from osmdata.openstreetmap.de, which is
written to the GeoJSON as water/ocean, as in OpenMapTiles.
"""
import argparse, gzip, json, math, os, re, shutil, subprocess, sys, urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "./data/osm_carto_compare")
import render_carto as rc
from render_carto import lonlat_to_px, pixel_window

SRC_LAYERS = ["points", "lines", "multipolygons"]  # GDAL OSM driver layers we need
LAYER_ORDER = ["water", "landcover", "landuse", "park", "boundary", "waterway", "building",
               "transportation", "transportation_name", "place", "mountain_peak", "poi"]
DIGITS = 7  # coordinate decimals (~1 cm)

# ----------------------------------------------------------------------------
# Projection helpers (inverse of render_carto.lonlat_to_px)
# ----------------------------------------------------------------------------

def px_to_lonlat(x, y, z):
    n = 256 * 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def query_box(bbox, z, buffer_px):
    """Lon/lat box of the render's pixel window grown by buffer_px on each side."""
    x0, y0, x1, y1 = pixel_window(bbox, z)
    min_lon, max_lat = px_to_lonlat(x0 - buffer_px, y0 - buffer_px, z)
    max_lon, min_lat = px_to_lonlat(x1 + buffer_px, y1 + buffer_px, z)
    return (max(min_lon, -180.0), max(min_lat, -85.05112878),
            min(max_lon, 180.0), min(max_lat, 85.05112878))

# ----------------------------------------------------------------------------
# OSM tags -> OpenMapTiles layers (subset of the openmaptiles mapping that
# render_carto.py draws, plus generic POIs)
# ----------------------------------------------------------------------------
LANDCOVER = {  # class -> subclasses (raw tag values)
    "farmland": ["farmland", "farm", "orchard", "vineyard", "plant_nursery"],
    "ice": ["glacier", "ice_shelf"],
    "wood": ["wood", "forest"],
    "rock": ["bare_rock", "scree"],
    "grass": ["fell", "grassland", "heath", "scrub", "tundra", "grass", "meadow", "allotments",
              "park", "village_green", "recreation_ground", "garden", "golf_course"],
    "wetland": ["wetland", "bog", "swamp", "wet_meadow", "marsh", "reedbed", "saltern",
                "tidalflat", "saltmarsh", "mangrove"],
    "sand": ["beach", "sand", "dune"],
}
SUB2LANDCOVER = {s: c for c, subs in LANDCOVER.items() for s in subs}
LANDUSE_CLASSES = {
    "railway", "cemetery", "military", "residential", "commercial", "industrial", "garages",
    "retail", "bus_station", "school", "university", "kindergarten", "college", "library",
    "hospital", "stadium", "pitch", "playground", "track", "theme_park", "zoo", "suburb",
    "quarter", "neighbourhood", "dam", "quarry", "brownfield", "construction", "farmyard",
}
LANDUSE_KEYS = ("landuse", "amenity", "leisure", "tourism", "place", "waterway")
RIVER_WATER = {"river", "canal", "stream", "ditch", "drain", "oxbow", "lock"}
WATERWAY_CLASSES = {"river", "canal", "stream", "ditch", "drain"}

ROAD_CLASS = {  # highway=* -> transportation class
    **{h: h for h in ("motorway", "trunk", "primary", "secondary", "tertiary")},
    **{h + "_link": h for h in ("motorway", "trunk", "primary", "secondary", "tertiary")},
    "residential": "minor", "unclassified": "minor", "living_street": "minor", "road": "minor",
    "service": "service", "track": "track", "raceway": "raceway",
    "pedestrian": "path", "path": "path", "footway": "path", "cycleway": "path",
    "steps": "path", "bridleway": "path", "corridor": "path",
}
RAIL_CLASS = {"rail": "rail", "narrow_gauge": "rail", "preserved": "rail", "funicular": "rail",
              "light_rail": "transit", "subway": "transit", "tram": "transit", "monorail": "transit"}
PLACE_CLASSES = {"continent", "country", "state", "province", "city", "town", "village", "hamlet",
                 "suburb", "quarter", "neighbourhood", "isolated_dwelling", "island"}
POI_KEYS = ("amenity", "shop", "tourism", "leisure", "historic", "office", "craft", "sport")

OTHER_TAGS_RE = re.compile(r'"((?:[^"\\]|\\.)*)"=>"((?:[^"\\]|\\.)*)"')
SKIP_COLS = {"geometry", "osm_id", "osm_way_id", "other_tags", "z_order"}


def unescape(s):
    return s.replace('\\"', '"').replace("\\\\", "\\")


def tags_of(row):
    """All OSM tags of a GDAL OSM-driver row: real columns plus the other_tags hstore."""
    t = {k: str(v) for k, v in row.items()
         if k not in SKIP_COLS and v is not None and not (isinstance(v, float) and math.isnan(v))}
    other = row.get("other_tags")
    if isinstance(other, str):
        for k, v in OTHER_TAGS_RE.findall(other):
            t.setdefault(unescape(k), unescape(v))
    return t


def osm_ref(row, src):
    """(osm_type, osm_id) of a row; multipolygons are either relations or closed ways."""
    for col, kind in (("osm_id", "relation" if src == "multipolygons" else
                       "node" if src == "points" else "way"), ("osm_way_id", "way")):
        v = row.get(col)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return kind, int(float(v))
    return None, None


def as_int(v):
    """Integer from free-form OSM numbers: '38,000', '1 234 m', '12.5', '8;9' (first value)."""
    if v is None:
        return None
    s = str(v).split(";")[0].strip().lower().removesuffix("m").replace(" ", "")
    s = s.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s) else s.replace(",", ".")
    try:
        return int(round(float(s)))
    except ValueError:
        return None


def yes(v):
    return v in ("yes", "1", "true")


def brunnel(t):
    if t.get("bridge") and t["bridge"] != "no":
        return "bridge"
    if t.get("tunnel") and t["tunnel"] != "no":
        return "tunnel"
    if yes(t.get("ford")):
        return "ford"
    return None


def names(t):
    p = {}
    if t.get("name"):
        p["name"] = t["name"]
    if t.get("name:en"):
        p["name_en"] = t["name:en"]
    if p:
        p["name_int"] = p.get("name_en", p.get("name"))
    return p


def clean(p):
    return {k: v for k, v in p.items() if v is not None}


def classify(src, t):
    """OpenMapTiles features for one OSM object: list of (omt_layer, properties, kind).
    kind is 'area', 'line', 'outline' (an area's boundary as lines) or 'point'."""
    out = []
    if src == "multipolygons":
        # landcover
        nat, lu, lei = t.get("natural"), t.get("landuse"), t.get("leisure")
        if nat == "wetland":
            sub = t.get("wetland") if t.get("wetland") in SUB2LANDCOVER else "wetland"
            out.append(("landcover", {"class": "wetland", "subclass": sub}, "area"))
        else:
            for v in (lu, nat, lei):
                if v in SUB2LANDCOVER:
                    out.append(("landcover", {"class": SUB2LANDCOVER[v], "subclass": v}, "area"))
                    break
        # landuse
        for k in LANDUSE_KEYS:
            if t.get(k) in LANDUSE_CLASSES:
                out.append(("landuse", {"class": t[k]}, "area"))
                break
        # water
        wcls = None
        if nat == "water":
            w = t.get("water")
            wcls = "river" if w in RIVER_WATER else "pond" if w == "pond" else "lake"
        elif lu in ("reservoir", "basin"):
            wcls = "lake"
        elif t.get("waterway") == "riverbank":
            wcls = "river"
        elif t.get("waterway") == "dock":
            wcls = "dock"
        elif lei == "swimming_pool":
            wcls = "swimming_pool"
        if wcls:
            out.append(("water", clean({"class": wcls, "intermittent": 1 if yes(t.get("intermittent")) else 0,
                                        "brunnel": "tunnel" if t.get("tunnel") == "culvert" else None,
                                        **names(t)}), "area"))
        # buildings
        if t.get("building") and t["building"] != "no":
            out.append(("building", clean({"render_height": as_int(t.get("height")),
                                           "colour": t.get("building:colour")}), "area"))
        # protected areas
        if t.get("boundary") in ("national_park", "protected_area") or lei == "nature_reserve":
            out.append(("park", {"class": t.get("boundary") if t.get("boundary") in
                                 ("national_park", "protected_area") else "nature_reserve", **names(t)}, "area"))
        # pedestrian squares
        if t.get("highway") in ("pedestrian", "footway") and yes(t.get("area", "yes")):
            out.append(("transportation", {"class": "path", "subclass": t["highway"]}, "area"))
        # administrative boundaries (relations), drawn as lines
        if t.get("boundary") == "administrative" and as_int(t.get("admin_level")) is not None:
            out.append(("boundary", {"admin_level": as_int(t.get("admin_level")),
                                     "maritime": 1 if yes(t.get("maritime")) else 0}, "outline"))

    elif src == "lines":
        hw, rw, ww = t.get("highway"), t.get("railway"), t.get("waterway")
        if hw in ROAD_CLASS:
            cls = ROAD_CLASS[hw]
            p = {"class": cls, "subclass": hw if cls == "path" else None,
                 "ramp": 1 if hw.endswith("_link") else None, "brunnel": brunnel(t),
                 "layer": as_int(t.get("layer")), "oneway": 1 if yes(t.get("oneway")) else None,
                 "service": t.get("service") if cls == "service" else None,
                 "surface": t.get("surface")}
            out.append(("transportation", clean(p), "line"))
            if t.get("name") or t.get("ref"):
                out.append(("transportation_name", clean({"class": cls, "subclass": p["subclass"],
                                                          "ref": t.get("ref"), "brunnel": p["brunnel"],
                                                          **names(t)}), "line"))
        elif rw in RAIL_CLASS:
            out.append(("transportation", clean({"class": RAIL_CLASS[rw], "subclass": rw, "brunnel": brunnel(t),
                                                 "layer": as_int(t.get("layer")),
                                                 "service": t.get("service")}), "line"))
        if ww in WATERWAY_CLASSES:
            out.append(("waterway", clean({"class": ww, "brunnel": brunnel(t),
                                           "intermittent": 1 if yes(t.get("intermittent")) else 0,
                                           **names(t)}), "line"))

    elif src == "points":
        if t.get("place") in PLACE_CLASSES:
            out.append(("place", clean({"class": t["place"], "population": as_int(t.get("population")),
                                        "capital": as_int(t.get("capital")), **names(t)}), "point"))
        if t.get("natural") in ("peak", "volcano", "saddle"):
            out.append(("mountain_peak", clean({"class": t["natural"], "ele": as_int(t.get("ele")),
                                                **names(t)}), "point"))
        if t.get("railway") in ("station", "halt"):
            out.append(("poi", {"class": "railway", "subclass": t["railway"], **names(t)}, "point"))
        else:
            for k in POI_KEYS:
                if t.get(k):
                    out.append(("poi", {"class": t[k], "subclass": t[k], "osm_key": k, **names(t)}, "point"))
                    break
    return out

# ----------------------------------------------------------------------------
# Geometry: clip to the query box, keep the parts of the right dimension
# ----------------------------------------------------------------------------

def rnd(c):
    return round(c, DIGITS) if isinstance(c, float) else [rnd(x) for x in c]


def flat_parts(g):
    import shapely
    for p in shapely.get_parts(g):
        if p.geom_type in ("MultiPolygon", "MultiLineString", "MultiPoint", "GeometryCollection"):
            yield from flat_parts(p)
        elif not p.is_empty:
            yield p


def clip_geom(g, box, kind):
    """Clipped GeoJSON geometry dict, or None if nothing of the right type is left."""
    import shapely
    from shapely.geometry import MultiLineString, MultiPolygon, mapping
    if g is None or g.is_empty:
        return None
    if kind == "point":
        if g.geom_type != "Point":
            return None
        return mapping(g) if box[0] <= g.x <= box[2] and box[1] <= g.y <= box[3] else None
    if kind == "outline":
        g = g.boundary
    elif kind == "area" and not g.is_valid:
        g = shapely.make_valid(g)
    g = shapely.clip_by_rect(g, *box)
    want = "Polygon" if kind == "area" else "LineString"
    parts = [p for p in flat_parts(g) if p.geom_type == want]
    if kind != "area":  # clip_by_rect can return a MultiLineString of short pieces; merge them
        parts = [p for p in parts if p.length > 0]
    if not parts:
        return None
    if len(parts) == 1:
        return mapping(parts[0])
    return mapping(MultiPolygon(parts) if kind == "area" else MultiLineString(parts))


def feature(layer, props, geom, osm_type=None, osm_id=None):
    p = {"omt_layer": layer, **props}
    if osm_id is not None:
        p["osm_type"], p["osm_id"] = osm_type, osm_id
    return {"type": "Feature", "geometry": {"type": geom["type"], "coordinates": rnd(geom["coordinates"])},
            "properties": p}

# ----------------------------------------------------------------------------
# Steps 1 & 2: download + one-time conversion
# ----------------------------------------------------------------------------

def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} already downloaded")
        return
    print(f"[download] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)


def to_gpkg(pbf: Path, gpkg: Path, clip=None) -> None:
    import pyogrio
    if gpkg.exists():
        have = {name for name, _ in pyogrio.list_layers(gpkg)}
        if set(SRC_LAYERS) <= have:
            print(f"[skip] {gpkg.name} already built")
            return
        print(f"[rebuild] {gpkg.name} lacks layers {sorted(set(SRC_LAYERS) - have)}")
        gpkg.unlink()
    print(f"[convert] {pbf.name} -> {gpkg.name} (one-time, can take a while)")
    if shutil.which("ogr2ogr"):
        spat = ["-spat", *map(str, clip)] if clip else []
        subprocess.run(["ogr2ogr", "-f", "GPKG", *spat, str(gpkg), str(pbf), *SRC_LAYERS], check=True)
    else:
        for layer in SRC_LAYERS:
            df = pyogrio.read_dataframe(pbf, layer=layer, bbox=clip)
            pyogrio.write_dataframe(df, gpkg, layer=layer, driver="GPKG")


def prepare_water(shp: Path, out: Path, clip) -> None:
    """Clip the (global) water polygons once into an indexed GeoPackage."""
    import pyogrio
    if out.exists():
        print(f"[skip] {out.name} already built")
        return
    if not clip:
        print("[warn] no --region-bbox given; importing the full global water file")
    print(f"[convert] {shp.name} -> {out.name}")
    df = pyogrio.read_dataframe(shp, bbox=clip)
    pyogrio.write_dataframe(df, out, layer="sea", driver="GPKG")

# ----------------------------------------------------------------------------
# Step 3: one crop -> GeoJSON (+ render)
# ----------------------------------------------------------------------------

def extract(gpkg, bbox, zoom, buffer_px, water_path=None):
    """OpenMapTiles-schema features for a box, clipped to its pixel window + buffer."""
    import pyogrio
    box = query_box(bbox, zoom, buffer_px)
    by_layer = {layer: [] for layer in LAYER_ORDER}
    if water_path:
        sea = pyogrio.read_dataframe(water_path, layer="sea", bbox=box)
        for g in sea.geometry.values:
            geom = clip_geom(g, box, "area")
            if geom:
                by_layer["water"].append(feature("water", {"class": "ocean"}, geom))
    for src in SRC_LAYERS:
        df = pyogrio.read_dataframe(gpkg, layer=src, bbox=box)
        if not len(df):
            continue
        rows = df.drop(columns="geometry").to_dict("records")
        for row, g in zip(rows, df.geometry.values):
            specs = classify(src, tags_of(row))
            if not specs:
                continue
            kind_id = osm_ref(row, src)
            for layer, props, kind in specs:
                geom = clip_geom(g, box, kind)
                if geom:
                    by_layer[layer].append(feature(layer, props, geom, *kind_id))
    return [f for layer in LAYER_ORDER for f in by_layer[layer]]


def write_geojson(path, name, bbox, zoom, features, buffer_px, source):
    data = {"type": "FeatureCollection",
            "bbox": [round(v, DIGITS) for v in bbox],
            "metadata": {"name": name, "source_zoom": zoom, "pixel_window": list(pixel_window(bbox, zoom)),
                         "buffer_px": buffer_px, "schema": "openmaptiles (subset)",
                         "source": source, "attribution": "© OpenStreetMap contributors (ODbL)",
                         "generator": "osm_crops.py"},
            "features": features}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))


def crop_job(job):
    """Worker: one failed crop shouldn't stop a run of thousands."""
    name, bbox, zoom, kw = job
    row = dict(name=name, zoom=zoom, min_lon=bbox[0], min_lat=bbox[1], max_lon=bbox[2], max_lat=bbox[3])
    out = Path(kw["out_dir"])
    gj = out / f"{name}.{kw['gj_ext']}"
    img = out / f"{name}.{kw['fmt']}"
    try:
        win = pixel_window(bbox, zoom)
        w, h = (win[2] - win[0]) * kw["scale"], (win[3] - win[1]) * kw["scale"]
        if max(w, h) > kw["max_px"]:
            raise ValueError(f"{w}x{h} px exceeds --max-px {kw['max_px']}; lower the zoom or raise the cap")
        feats = extract(kw["gpkg"], bbox, zoom, kw["buffer_px"], kw["water_path"])
        write_geojson(gj, name, bbox, zoom, feats, kw["buffer_px"], kw["source"])
        row.update(geojson=gj.name, features=len(feats), px_x0=win[0], px_y0=win[1], px_x1=win[2], px_y1=win[3])
        if kw["images"]:
            info = rc.render(str(gj), str(img), zoom, kw["scale"])
            row.update(file=img.name, width=info["size"][0], height=info["size"][1])
        return dict(row, error="")
    except Exception as e:  # noqa: BLE001
        return dict(row, error=repr(e))

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pbf-url", help="URL of a .osm.pbf extract (e.g. Geofabrik)")
    src.add_argument("--pbf", type=Path, help="local .osm.pbf file")
    ap.add_argument("--boxes", type=Path, required=True, help="CSV with name,min_lon,min_lat,max_lon,max_lat[,zoom]")
    ap.add_argument("--zoom", type=int, default=16,
                    help="OSM zoom to cut and render at (default 16); a 'zoom' CSV column overrides it per box")
    ap.add_argument("--scale", type=int, default=1, help="render_carto scale (2 = high-DPI)")
    ap.add_argument("--buffer-px", type=int, default=32,
                    help="vector data kept beyond the image edge, in pixels at the box's zoom (default 32)")
    ap.add_argument("--out", type=Path, default=Path("crops"))
    ap.add_argument("--workdir", type=Path, default=Path("osm_data"))
    ap.add_argument("--max-px", type=int, default=12000, help="cap on either image dimension")
    ap.add_argument("--format", choices=["png", "jpg"], default="png")
    ap.add_argument("--no-gzip", action="store_true", help="write .geojson instead of .geojson.gz")
    ap.add_argument("--no-images", action="store_true", help="only write the GeoJSON, skip rendering")
    ap.add_argument("--region-bbox", type=float, nargs=4, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                    help="only keep data inside this box when converting (saves time/disk)")
    ap.add_argument("--water", type=Path, help="water polygons shapefile (water-polygons-split-4326 from "
                                               "osmdata.openstreetmap.de) for sea areas")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="parallel processes")
    ap.add_argument("--overwrite", action="store_true", help="redo crops whose outputs already exist")
    a = ap.parse_args()

    a.workdir.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)

    if a.pbf_url:
        pbf = a.workdir / a.pbf_url.rstrip("/").split("/")[-1]
        download(a.pbf_url, pbf)
    else:
        pbf = a.pbf
    suffix = "_clip" if a.region_bbox else ""
    gpkg = a.workdir / (pbf.name.split(".")[0] + suffix + ".gpkg")
    to_gpkg(pbf, gpkg, a.region_bbox)

    water_gpkg = None
    if a.water:
        water_gpkg = a.workdir / f"water{suffix}.gpkg"
        prepare_water(a.water, water_gpkg, a.region_bbox)
    else:
        print("[note] no --water given: sea areas will be drawn as land")

    gj_ext = "geojson" if a.no_gzip else "geojson.gz"
    kw = dict(gpkg=str(gpkg), water_path=str(water_gpkg) if water_gpkg else None, out_dir=str(a.out),
              gj_ext=gj_ext, fmt=a.format, scale=a.scale, buffer_px=a.buffer_px, max_px=a.max_px,
              images=not a.no_images, source=pbf.name)

    boxes = pd.read_csv(a.boxes)
    has_zoom = "zoom" in boxes.columns
    jobs, skipped = [], 0
    for i, row in enumerate(boxes.itertuples(index=False)):
        name = str(getattr(row, "name", f"box_{i:05d}"))
        bbox = (float(row.min_lon), float(row.min_lat), float(row.max_lon), float(row.max_lat))
        zoom = int(row.zoom) if has_zoom and pd.notna(row.zoom) else a.zoom
        if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            print(f"[warn] {name}: invalid bbox {bbox}, skipped")
            continue
        if not 0 <= zoom <= 22:
            print(f"[warn] {name}: zoom {zoom} out of range, skipped")
            continue
        done = (a.out / f"{name}.{gj_ext}").exists() and (a.no_images or (a.out / f"{name}.{a.format}").exists())
        if done and not a.overwrite:
            skipped += 1
            continue
        jobs.append((name, bbox, zoom, kw))
    if skipped:
        print(f"[skip] {skipped} crops already exist (use --overwrite to redo)")

    results = []
    if jobs:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(crop_job, j) for j in jobs]
            for n, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                results.append(r)
                status = f"ERROR {r['error']}" if r["error"] else f"{r['geojson']} ({r['features']} features)"
                if r["error"] or n % 50 == 0 or n == len(jobs):
                    print(f"[{n}/{len(jobs)}] {r['name']}: {status}")

    index_path = a.out / "index.csv"
    new = pd.DataFrame(results)
    if index_path.exists() and not new.empty:  # keep rows from earlier runs
        old = pd.read_csv(index_path)
        new = pd.concat([old[~old["name"].astype(str).isin(new["name"])], new])
    if not new.empty:
        new.to_csv(index_path, index=False)
    errors = sum(1 for r in results if r["error"])
    print(f"done: {len(results) - errors} crops written, {errors} failed, {skipped} skipped -> {a.out}/")


if __name__ == "__main__":
    main()
