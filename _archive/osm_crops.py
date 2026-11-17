1  #!/usr/bin/env python3
"""
osm_crops.py: download a regional OpenStreetMap extract once, then render
PNG/JPEG images of the vector data for a list of bounding boxes.

Pipeline
  1. Download a .osm.pbf extract (e.g. from https://download.geofabrik.de/)
  2. Convert it once to a GeoPackage (spatially indexed -> fast bbox queries)
  3. For each bounding box: query features, style them, save an image

Install
  pip install geopandas pyogrio matplotlib pandas
  (Optional) GDAL command-line tools: if `ogr2ogr` is on PATH, step 2 uses it.

Usage
  python osm_crops.py \
      --pbf-url https://download.geofabrik.de/europe/monaco-latest.osm.pbf \
      --boxes boxes.csv --out crops --width 1024 --format png --style light

  or, with a file you already have:
  python osm_crops.py --pbf my_region.osm.pbf --boxes boxes.csv

boxes.csv (WGS84 lon/lat):
  name,min_lon,min_lat,max_lon,max_lat
  harbor,7.4180,43.7300,7.4300,43.7380
"""
import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import subprocess
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyogrio
from pyproj import Transformer

LAYERS = ["multipolygons", "lines"]
DPI = 100
MIN_LW = 0.3  # thinnest line drawn, in points

STYLES = {
    "light": dict(
        bg="#f2efe9",
        landuse="#e6e2da",
        green="#cde6c0",
        water="#aad3df",
        building="#d9d0c9",
        building_edge="#c4b8ae",
        major="#f6b26b",
        mid="#fde9a9",
        minor="#ffffff",
        path="#d98c7a",
        rail="#7a7a7a",
    ),
    "dark": dict(
        bg="#1b1d22",
        landuse="#22252b",
        green="#1f3326",
        water="#16304a",
        building="#3a3d45",
        building_edge="#4a4e57",
        major="#e0a458",
        mid="#c9c9c9",
        minor="#8a8f99",
        path="#5a5f69",
        rail="#9a9a9a",
    ),
}

# highway tag -> (style group, approximate real-world width in meters)
ROAD_CLASSES = {
    "motorway": ("major", 14),
    "motorway_link": ("major", 8),
    "trunk": ("major", 12),
    "trunk_link": ("major", 7),
    "primary": ("major", 10),
    "primary_link": ("major", 6),
    "secondary": ("mid", 9),
    "secondary_link": ("mid", 5),
    "tertiary": ("mid", 8),
    "tertiary_link": ("mid", 5),
    "residential": ("minor", 6),
    "unclassified": ("minor", 6),
    "living_street": ("minor", 5),
    "service": ("minor", 4),
    "pedestrian": ("path", 4),
    "track": ("path", 3),
    "footway": ("path", 1.5),
    "path": ("path", 1.5),
    "cycleway": ("path", 1.5),
    "steps": ("path", 1.5),
}
GROUP_ORDER = ["path", "minor", "mid", "major"]  # draw bottom -> top

GREEN = dict(
    leisure=["park", "garden", "pitch", "golf_course", "nature_reserve", "playground"],
    landuse=[
        "grass",
        "forest",
        "meadow",
        "recreation_ground",
        "cemetery",
        "village_green",
        "orchard",
        "vineyard",
        "allotments",
    ],
    natural=["wood", "scrub", "grassland", "heath"],
)
WATER = dict(natural=["water", "bay", "wetland"], landuse=["reservoir", "basin"])
LANDUSE = dict(landuse=["residential", "commercial", "industrial", "retail"])


# ---------------------------------------------------------------- step 1 & 2
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
    if gpkg.exists():
        print(f"[skip] {gpkg.name} already built")
        return
    print(f"[convert] {pbf.name} -> {gpkg.name} (one-time, can take a while)")
    if shutil.which("ogr2ogr"):
        spat = ["-spat", *map(str, clip)] if clip else []
        subprocess.run(
            ["ogr2ogr", "-f", "GPKG", *spat, str(gpkg), str(pbf), *LAYERS], check=True
        )
    else:
        for layer in LAYERS:
            df = pyogrio.read_dataframe(pbf, layer=layer, bbox=clip)
            pyogrio.write_dataframe(df, gpkg, layer=layer, driver="GPKG")


# ---------------------------------------------------------------- step 3
def load(gpkg: Path, layer: str, bbox) -> gpd.GeoDataFrame:
    df = pyogrio.read_dataframe(gpkg, layer=layer, bbox=bbox)
    return df.to_crs(3857) if len(df) else df


def match(df, rules: dict) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for field, values in rules.items():
        if field in df.columns:
            mask |= df[field].isin(values)
    return mask


def has(df, field: str) -> pd.Series:
    if field not in df.columns:
        return pd.Series(False, index=df.index)
    return df[field].notna()


def render(
    gpkg,
    name,
    bbox,
    out_dir,
    width_px,
    fmt,
    style,
    water_path=None,
    mpp=None,
    max_px=12000,
):
    min_lon, min_lat, max_lon, max_lat = bbox
    to3857 = Transformer.from_crs(4326, 3857, always_xy=True)
    x0, y0 = to3857.transform(min_lon, min_lat)
    x1, y1 = to3857.transform(max_lon, max_lat)
    # Web Mercator inflates distances by 1/cos(lat)
    merc_scale = 1 / math.cos(math.radians((min_lat + max_lat) / 2))
    if mpp:  # fixed ground scale: image size follows box size
        width_px = round((x1 - x0) / merc_scale / mpp)
    width_px = max(1, min(width_px, max_px))
    height_px = max(1, min(round(width_px * (y1 - y0) / (x1 - x0)), max_px))

    # points per real-world meter
    k = (width_px / DPI * 72) / (x1 - x0) * merc_scale

    fig = plt.figure(figsize=(width_px / DPI, height_px / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    fig.patch.set_facecolor(style["bg"])

    polys = load(gpkg, "multipolygons", bbox)
    lines = load(gpkg, "lines", bbox)

    if water_path:  # sea / harbor polygons (OSM itself only has coastline lines)
        sea = pyogrio.read_dataframe(water_path, layer="sea", bbox=bbox)
        if len(sea):
            sea.to_crs(3857).plot(ax=ax, color=style["water"], linewidth=0)

    if len(polys):
        for rules, color in [(LANDUSE, "landuse"), (GREEN, "green"), (WATER, "water")]:
            sel = polys[match(polys, rules)]
            if len(sel):
                sel.plot(ax=ax, color=style[color], linewidth=0)
        bld = polys[has(polys, "building")]
        if len(bld):
            bld.plot(
                ax=ax,
                color=style["building"],
                edgecolor=style["building_edge"],
                linewidth=0.3 if k > 0.5 else 0,
            )

    if len(lines):
        ww = lines[match(lines, dict(waterway=["river", "stream", "canal"]))]
        if len(ww):
            ww.plot(ax=ax, color=style["water"], linewidth=max(MIN_LW, 6 * k))
        hw = lines["highway"] if "highway" in lines.columns else None
        if hw is not None:
            for group in GROUP_ORDER:
                for tag, (g, w) in ROAD_CLASSES.items():
                    sel = lines[hw == tag] if g == group else []
                    if len(sel):
                        sel.plot(
                            ax=ax, color=style[group], linewidth=max(MIN_LW, w * k)
                        )
        rail = lines[has(lines, "railway")]
        if len(rail):
            rail.plot(ax=ax, color=style["rail"], linewidth=max(MIN_LW, 2 * k))

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    path = out_dir / f"{name}.{fmt}"
    kwargs = {"pil_kwargs": {"quality": 92}} if fmt == "jpg" else {}
    fig.savefig(path, dpi=DPI, facecolor=style["bg"], **kwargs)
    plt.close(fig)
    return path, width_px, height_px


def render_job(job):
    """Worker wrapper: one failed crop shouldn't stop a run of thousands."""
    name, bbox, kw = job
    try:
        path, w, h = render(name=name, bbox=bbox, **kw)
        return dict(
            name=name,
            file=path.name,
            width=w,
            height=h,
            error="",
            min_lon=bbox[0],
            min_lat=bbox[1],
            max_lon=bbox[2],
            max_lat=bbox[3],
        )
    except Exception as e:  # noqa: BLE001
        return dict(
            name=name,
            file="",
            width=0,
            height=0,
            error=repr(e),
            min_lon=bbox[0],
            min_lat=bbox[1],
            max_lon=bbox[2],
            max_lat=bbox[3],
        )


def prepare_water(shp: Path, out: Path, clip) -> None:
    """Clip the (global) water polygons once into an indexed GeoPackage."""
    if out.exists():
        print(f"[skip] {out.name} already built")
        return
    if not clip:
        print("[warn] no --region-bbox given; importing the full global water file")
    print(f"[convert] {shp.name} -> {out.name}")
    df = pyogrio.read_dataframe(shp, bbox=clip)
    pyogrio.write_dataframe(df, out, layer="sea", driver="GPKG")


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pbf-url", help="URL of a .osm.pbf extract (e.g. Geofabrik)")
    src.add_argument("--pbf", type=Path, help="local .osm.pbf file")
    p.add_argument(
        "--boxes",
        type=Path,
        required=True,
        help="CSV with name,min_lon,min_lat,max_lon,max_lat",
    )
    p.add_argument("--out", type=Path, default=Path("crops"))
    p.add_argument("--workdir", type=Path, default=Path("osm_data"))
    size = p.add_mutually_exclusive_group()
    size.add_argument(
        "--width",
        type=int,
        default=1024,
        help="fixed image width in pixels (default; scale varies by box)",
    )
    size.add_argument(
        "--mpp",
        type=float,
        help="fixed meters per pixel instead (image size varies by box)",
    )
    p.add_argument(
        "--max-px", type=int, default=12000, help="cap on either image dimension"
    )
    p.add_argument("--format", choices=["png", "jpg"], default="png")
    p.add_argument("--style", choices=list(STYLES), default="light")
    p.add_argument(
        "--region-bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="only keep data inside this box when converting (saves time/disk)",
    )
    p.add_argument(
        "--water",
        type=Path,
        help="optional water polygons shapefile (water-polygons-split-4326 "
        "from osmdata.openstreetmap.de) to fill sea and harbor areas",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="parallel render processes",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="re-render images that already exist (default: skip them)",
    )
    a = p.parse_args()

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

    kw = dict(
        gpkg=gpkg,
        out_dir=a.out,
        width_px=a.width,
        fmt=a.format,
        style=STYLES[a.style],
        water_path=water_gpkg,
        mpp=a.mpp,
        max_px=a.max_px,
    )

    boxes = pd.read_csv(a.boxes)
    jobs, skipped = [], 0
    for i, row in enumerate(boxes.itertuples(index=False)):
        name = str(getattr(row, "name", f"box_{i:05d}"))
        bbox = (row.min_lon, row.min_lat, row.max_lon, row.max_lat)
        if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            print(f"[warn] {name}: invalid bbox {bbox}, skipped")
            continue
        if not a.overwrite and (a.out / f"{name}.{a.format}").exists():
            skipped += 1
            continue
        jobs.append((name, bbox, kw))
    if skipped:
        print(f"[skip] {skipped} images already exist (use --overwrite to redo)")

    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(render_job, j) for j in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            status = f"ERROR {r['error']}" if r["error"] else r["file"]
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
    print(
        f"done: {len(results) - errors} rendered, {errors} failed, "
        f"{skipped} skipped -> {a.out}/"
    )


if __name__ == "__main__":
    main()
