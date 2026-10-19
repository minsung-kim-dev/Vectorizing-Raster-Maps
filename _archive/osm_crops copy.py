#!/usr/bin/env python3
"""
osm_crops.py: download a regional OpenStreetMap extract once, then render
images of it with the real openstreetmap-carto style (the style used by
tile.openstreetmap.org) for a list of bounding boxes.

Pipeline
  1. Download a .osm.pbf extract (e.g. from https://download.geofabrik.de/)
  2. Import it once into PostGIS with osm2pgsql using Carto's import style,
     add Carto's indexes/functions, and load Carto's external data
     (land/water polygons etc.)
  3. Compile Carto's project.mml into a Mapnik XML style (once)
  4. For each bounding box: render with Mapnik, pixel-aligned to the standard
     256px OSM tile grid at the chosen zoom, so the result can be compared
     pixel for pixel with tile.openstreetmap.org tiles

One-time setup (Debian/Ubuntu shown; see openstreetmap-carto's INSTALL.md)
  sudo apt install postgresql postgis osm2pgsql python3-mapnik \
                   python3-psycopg2 python3-yaml python3-requests nodejs npm
  sudo npm install -g carto
  git clone https://github.com/gravitystorm/openstreetmap-carto.git
  (cd openstreetmap-carto && scripts/get-fonts.sh)
  sudo -u postgres createuser -s $USER
  createdb gis
  psql -d gis -c "CREATE EXTENSION postgis; CREATE EXTENSION hstore;"
  pip install pandas

  Database connection details other than the name (host, user, password,
  port) are taken from the standard libpq variables PGHOST, PGUSER,
  PGPASSWORD, PGPORT; osm2pgsql, psql and Mapnik all honour them.

Usage
  python osm_crops.py \
      --pbf-url https://download.geofabrik.de/europe/monaco-latest.osm.pbf \
      --carto-dir openstreetmap-carto --boxes boxes.csv --zoom 16

  or, with a file you already have:
  python osm_crops.py --pbf my_region.osm.pbf --carto-dir openstreetmap-carto \
      --boxes boxes.csv --zoom 15 --scale 2

boxes.csv (WGS84 lon/lat; the zoom column is optional and overrides --zoom):
  name,min_lon,min_lat,max_lon,max_lat,zoom
  harbor,7.4180,43.7300,7.4300,43.7380,17
"""
import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ORIGIN = math.pi * 6378137.0  # half the Web Mercator world width in metres


# ---------------------------------------------------------------- tile-grid maths
def lonlat_to_px(lon, lat, z):
    """Global pixel coordinates at zoom z (256px tiles), same as render_carto.py."""
    n = 256 * 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat = max(min(lat, 85.05112878), -85.05112878)
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return x, y


def pixel_window(bbox, z):
    """Integer pixel window [x0, y0, x1, y1) covering the bbox at zoom z."""
    x0, y1 = lonlat_to_px(bbox[0], bbox[1], z)
    x1, y0 = lonlat_to_px(bbox[2], bbox[3], z)
    return math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)


def window_to_3857(window, z):
    """Exact EPSG:3857 extent (minx, miny, maxx, maxy) of a pixel window."""
    x0, y0, x1, y1 = window
    res = 2 * ORIGIN / (256 * 2 ** z)  # metres per pixel at zoom z
    return x0 * res - ORIGIN, ORIGIN - y1 * res, x1 * res - ORIGIN, ORIGIN - y0 * res


# ---------------------------------------------------------------- helpers
def run(cmd, **kw):
    print("[run] " + " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True, **kw)


def need(tool, hint):
    if not shutil.which(tool):
        sys.exit(f"[error] '{tool}' not found on PATH. {hint}")


# ---------------------------------------------------------------- step 1
def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} already downloaded")
        return
    print(f"[download] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)


# ---------------------------------------------------------------- step 2
def import_db(pbf: Path, carto: Path, db: str, workdir: Path, clip, extra_args,
              reimport: bool) -> None:
    marker = workdir / f"imported_{db}.txt"
    stamp = f"{pbf.resolve()}\n{clip}\n"
    if marker.exists() and not reimport:
        if marker.read_text() != stamp:
            print(f"[warn] database '{db}' was imported from different data "
                  f"({marker.read_text().splitlines()[0]}); use --reimport to replace it")
        else:
            print(f"[skip] {pbf.name} already imported into '{db}'")
        return

    need("osm2pgsql", "Install osm2pgsql (e.g. apt install osm2pgsql).")
    need("psql", "Install the PostgreSQL client tools.")
    print(f"[import] {pbf.name} -> PostGIS database '{db}' (one-time, can take a while)")

    flex = carto / "openstreetmap-carto-flex.lua"
    if flex.exists():  # openstreetmap-carto v5.9+
        cmd = ["osm2pgsql", "-O", "flex", "-S", flex]
    else:  # older releases use the pgsql output with a tag-transform script
        cmd = ["osm2pgsql", "-G", "--hstore", "--style", carto / "openstreetmap-carto.style",
               "--tag-transform-script", carto / "openstreetmap-carto.lua"]
    if clip:
        cmd += ["--bbox", ",".join(map(str, clip))]
    cmd += ["-d", db, *extra_args, pbf]
    run(cmd)

    for sql in ("indexes.sql", "functions.sql"):
        if (carto / sql).exists():
            run(["psql", "-v", "ON_ERROR_STOP=1", "-d", db, "-f", carto / sql])

    # land/sea polygons, Antarctic ice etc.; Carto loads these into the database
    run([sys.executable, carto / "scripts" / "get-external-data.py", "-d", db], cwd=carto)
    marker.write_text(stamp)


# ---------------------------------------------------------------- step 3
def build_style(carto: Path, db: str, workdir: Path) -> Path:
    mml = carto / "project.mml"
    xml = workdir / f"carto_{db}.xml"
    newest_src = max(p.stat().st_mtime for p in [mml, *carto.glob("style/*.mss")])
    if xml.exists() and xml.stat().st_mtime >= newest_src:
        print(f"[skip] {xml.name} is up to date")
        return xml
    need("carto", "Install the CartoCSS compiler: npm install -g carto")
    print(f"[style] compiling {mml} -> {xml.name}")
    out = subprocess.run(["carto", str(mml)], cwd=carto, check=True,
                         capture_output=True, text=True).stdout
    # point every PostGIS layer at our database (project.mml defaults to "gis")
    out, n = re.subn(r'(<Parameter name="dbname">)(?:<!\[CDATA\[)?[^<\]]*(?:\]\]>)?(</Parameter>)',
                     rf"\g<1><![CDATA[{db}]]>\g<2>", out)
    if n == 0:
        print("[warn] no dbname parameters found in the compiled style")
    xml.write_text(out)
    return xml


# ---------------------------------------------------------------- step 4
_MAP = None  # one Mapnik map per worker process


def init_worker(xml_path, carto_dir):
    global _MAP
    import mapnik
    fonts = Path(carto_dir) / "fonts"
    if fonts.is_dir():
        mapnik.register_fonts(str(fonts))
    _MAP = mapnik.Map(256, 256)
    # base_path makes relative paths in the style (symbols/, patterns/) resolve
    mapnik.load_map_from_string(_MAP, Path(xml_path).read_text(), False, str(carto_dir))
    _MAP.buffer_size = 128  # render beyond the edges so lines/labels aren't clipped


def render(name, bbox, zoom, scale, out_dir, fmt, max_px):
    import mapnik
    window = pixel_window(bbox, zoom)
    w = (window[2] - window[0]) * scale
    h = (window[3] - window[1]) * scale
    if max(w, h) > max_px:
        raise ValueError(f"{w}x{h} px exceeds --max-px {max_px}; lower the zoom or raise the cap")
    _MAP.resize(w, h)
    _MAP.zoom_to_box(mapnik.Box2d(*window_to_3857(window, zoom)))
    im = mapnik.Image(w, h)
    mapnik.render(_MAP, im, float(scale))  # scale_factor keeps Carto's zoom-based styling
    path = Path(out_dir) / f"{name}.{fmt}"
    im.save(str(path), "jpeg92" if fmt == "jpg" else "png")
    return path, w, h, window


def render_job(job):
    """Worker wrapper: one failed crop shouldn't stop a run of thousands."""
    name, bbox, zoom, kw = job
    row = dict(name=name, zoom=zoom, min_lon=bbox[0], min_lat=bbox[1],
               max_lon=bbox[2], max_lat=bbox[3])
    try:
        path, w, h, win = render(name, bbox, zoom, **kw)
        return dict(row, file=path.name, width=w, height=h, error="",
                    px_x0=win[0], px_y0=win[1], px_x1=win[2], px_y1=win[3])
    except Exception as e:  # noqa: BLE001
        return dict(row, file="", width=0, height=0, error=repr(e))


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pbf-url", help="URL of a .osm.pbf extract (e.g. Geofabrik)")
    src.add_argument("--pbf", type=Path, help="local .osm.pbf file")
    p.add_argument("--carto-dir", type=Path, required=True,
                   help="checkout of https://github.com/gravitystorm/openstreetmap-carto")
    p.add_argument("--boxes", type=Path, required=True,
                   help="CSV with name,min_lon,min_lat,max_lon,max_lat[,zoom]")
    p.add_argument("--zoom", type=int, default=16,
                   help="OSM zoom level to render at (default 16); a 'zoom' column in "
                        "the CSV overrides it per box")
    p.add_argument("--scale", type=int, choices=[1, 2, 3, 4], default=1,
                   help="high-DPI factor: 2 = twice the pixels with the same map content")
    p.add_argument("--dbname", default="gis", help="PostGIS database name (default gis)")
    p.add_argument("--out", type=Path, default=Path("crops"))
    p.add_argument("--workdir", type=Path, default=Path("osm_data"))
    p.add_argument("--max-px", type=int, default=12000,
                   help="cap on either image dimension")
    p.add_argument("--format", choices=["png", "jpg"], default="png")
    p.add_argument("--region-bbox", type=float, nargs=4,
                   metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                   help="only import data inside this box (saves time/disk)")
    p.add_argument("--osm2pgsql-args", default="",
                   help='extra osm2pgsql options, e.g. "--cache 4000 --number-processes 4"')
    p.add_argument("--reimport", action="store_true",
                   help="re-run the database import even if it was done before")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="parallel render processes")
    p.add_argument("--overwrite", action="store_true",
                   help="re-render images that already exist (default: skip them)")
    a = p.parse_args()

    carto = a.carto_dir.resolve()
    if not (carto / "project.mml").exists():
        sys.exit(f"[error] {carto} doesn't look like an openstreetmap-carto checkout")
    if not (carto / "fonts").is_dir():
        print("[warn] no fonts/ folder in the Carto checkout; run scripts/get-fonts.sh "
              "there or labels may be missing")
    try:
        import mapnik  # noqa: F401
    except ImportError:
        sys.exit("[error] Python Mapnik bindings not found (e.g. apt install python3-mapnik)")

    a.workdir.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)

    if a.pbf_url:
        pbf = a.workdir / a.pbf_url.rstrip("/").split("/")[-1]
        download(a.pbf_url, pbf)
    else:
        pbf = a.pbf
    import_db(pbf, carto, a.dbname, a.workdir, a.region_bbox,
              a.osm2pgsql_args.split(), a.reimport)
    xml = build_style(carto, a.dbname, a.workdir.resolve())

    kw = dict(scale=a.scale, out_dir=str(a.out), fmt=a.format, max_px=a.max_px)
    boxes = pd.read_csv(a.boxes)
    has_zoom = "zoom" in boxes.columns
    jobs, skipped = [], 0
    for i, row in enumerate(boxes.itertuples(index=False)):
        name = str(getattr(row, "name", f"box_{i:05d}"))
        bbox = (row.min_lon, row.min_lat, row.max_lon, row.max_lat)
        zoom = int(row.zoom) if has_zoom and pd.notna(row.zoom) else a.zoom
        if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            print(f"[warn] {name}: invalid bbox {bbox}, skipped")
            continue
        if not 0 <= zoom <= 22:
            print(f"[warn] {name}: zoom {zoom} out of range, skipped")
            continue
        if not a.overwrite and (a.out / f"{name}.{a.format}").exists():
            skipped += 1
            continue
        jobs.append((name, bbox, zoom, kw))
    if skipped:
        print(f"[skip] {skipped} images already exist (use --overwrite to redo)")

    results = []
    if jobs:
        with ProcessPoolExecutor(max_workers=a.workers, initializer=init_worker,
                                 initargs=(str(xml), str(carto))) as pool:
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
    print(f"done: {len(results) - errors} rendered, {errors} failed, "
          f"{skipped} skipped -> {a.out}/")


if __name__ == "__main__":
    main()
