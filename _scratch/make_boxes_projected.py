#!/usr/bin/env python3
"""
make_boxes_projected.py: data_preprocess/make_boxes.py, but in a projected coordinate system.

Same options, sampling and output as make_boxes.py, except:
- the geometry work (centers, box sizes, land / overlap checks) is done in NY State Plane Long Island
  (EPSG:2263, US survey feet) instead of lon/lat degrees, so box sides are exact lengths; box corners are
  converted back to lon/lat for the CSV
- box centers are sampled in the official borough boundaries (BORO_URL), not hand-drawn outlines

Usage (from the repo root):
  python _scratch/make_boxes_projected.py --n 2400 --min-m 125 --max-m 125 --boroughs queens --test-frac 0.0834 --test-nta flushing --out boxes.csv
"""
import argparse
import math
import os
import random
import sys

import geopandas as gpd
import shapely
from pyproj import Transformer
from shapely.geometry import Point, box

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_preprocess"))
from make_boxes import BORO_URL, load_nta  # noqa: E402

CRS = "EPSG:2263"  # NY State Plane Long Island, US survey feet
FT = 0.3048006096  # meters per US survey foot
TO_LONLAT = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
BOROUGHS = ["bronx", "brooklyn", "manhattan", "queens", "staten_island"]


def sample_boxes(rng, polys, n, min_m, max_m, land, max_overlap=0.5, start=0, avoid=None):
    """n boxes (name, min_lon, min_lat, max_lon, max_lat, w_m, h_m) centered inside polys (area-weighted),
    skipping boxes that are less than half `land`, touch the geometry `avoid`, or share more than
    `max_overlap` of their area with an earlier box. polys, land and avoid are in CRS."""
    names, weights = list(polys), [polys[k].area for k in polys]
    rows, taken = [], []  # taken: the boxes so far
    tries = 0
    while len(rows) < n:
        tries += 1
        assert tries < 1000 * n, "boxes don't fit: lower --n / --test-frac or raise --max-overlap"
        boro = rng.choices(names, weights)[0]
        minx, miny, maxx, maxy = polys[boro].bounds
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if not polys[boro].contains(Point(x, y)):
            continue
        # creates box around center point
        w = math.exp(rng.uniform(math.log(min_m), math.log(max_m)))
        # vary aspect ratio for height
        h = w * rng.uniform(0.75, 1.33)
        # half the box size in feet
        dx, dy = w / FT / 2, h / FT / 2
        b = box(x - dx, y - dy, x + dx, y + dy)

        if land.intersection(b).area < 0.5 * b.area or (avoid is not None and avoid.intersects(b)):
            continue
        # skip if an earlier box covers more than max_overlap of this one
        if taken and shapely.area(shapely.intersection(taken, b)).max() > max_overlap * b.area:
            continue
        taken.append(b)
        # corners back to lon/lat (the box's slight rotation in lon/lat is negligible at this size)
        min_lon, min_lat = TO_LONLAT.transform(x - dx, y - dy)
        max_lon, max_lat = TO_LONLAT.transform(x + dx, y + dy)
        rows.append((f"{boro}_{start + len(rows) + 1:05d}", min_lon, min_lat, max_lon, max_lat, round(w), round(h)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--min-m", type=float, default=150, help="smallest box side, meters")
    p.add_argument("--max-m", type=float, default=2500, help="largest box side, meters")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="boxes.csv")
    p.add_argument("--boroughs", nargs="+", default=BOROUGHS, choices=BOROUGHS)
    p.add_argument("--test-frac", type=float, default=None,
                   help="fraction of boxes in the held-out borough (e.g. 0.1); default: all boroughs area-weighted")
    p.add_argument("--holdout", default="queens", choices=BOROUGHS, help="held-out borough for --test-frac")
    p.add_argument("--test-nta", default='flushing',
                   help="with --test-frac: hold out the NTAs whose name contains this text instead of a borough")
    p.add_argument("--nta-file", default="data/nyc_nta_2020.geojson", help="NYC 2020 NTAs, for land and --test-nta (downloaded if missing)")
    p.add_argument("--osm-file", default="osm_data/NewYork.gpkg", help="OSM GeoPackage, for water")
    p.add_argument("--max-overlap", type=float, default=0.5,
                   help="skip boxes sharing more than this fraction of their area with an earlier box (1 = no limit)")
    a = p.parse_args()

    rng = random.Random(a.seed)
    # official borough boundaries, projected; "The Bronx" -> bronx, "Staten Island" -> staten_island
    boros = gpd.read_file(BORO_URL).to_crs(CRS)
    boros.index = boros.boro_name.str.lower().str.removeprefix("the ").str.replace(" ", "_")
    polys = {k: boros.geometry[k] for k in a.boroughs}
    nta = load_nta(a.nta_file).to_crs(CRS)
    water = gpd.read_file(a.osm_file, layer="multipolygons", where="natural = 'water'").to_crs(CRS)
    land = nta.union_all().difference(water.union_all())  # every NTA, clipped to shoreline
    if a.test_frac is None:
        rows = sample_boxes(rng, polys, a.n, a.min_m, a.max_m, land, a.max_overlap)
    else:
        n_test = round(a.n * a.test_frac)
        match = nta[nta.NTAName.str.contains(a.test_nta, case=False)]
        print("test NTAs:", ", ".join(match.NTAName))
        area = match.union_all()
        rows = sample_boxes(rng, polys, a.n - n_test, a.min_m, a.max_m, land, a.max_overlap, avoid=area)
        test = {a.test_nta.lower().replace(" ", "_"): area}
        rows += sample_boxes(rng, test, n_test, a.min_m, a.max_m, land, a.max_overlap, start=len(rows))

    with open(a.out, "w") as f:
        f.write("name,min_lon,min_lat,max_lon,max_lat,width_m,height_m\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]},{r[6]}\n")
    print(f"wrote {len(rows)} boxes to {a.out}")


if __name__ == "__main__":
    main()
