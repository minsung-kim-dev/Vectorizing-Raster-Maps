#!/usr/bin/env python3
"""
make_boxes.py: generate random NYC bounding boxes of varying sizes for osm_crops.py.

Box centers are sampled inside rough, hand-drawn outlines of the five boroughs, and boxes that are
less than half land are skipped (land = all NYC 2020 NTAs, which follow the shoreline, minus OSM water
such as lakes in parks).

Usage:
  python make_boxes.py --n 3000 --min-m 150 --max-m 2500 --seed 42 --out boxes.csv
  python make_boxes.py --n 15000 --min-m 115 --max-m 115 --test-frac 0.1 --holdout queens --out boxes.csv
      (90% of boxes in the other boroughs, 10% in the held-out borough)
  python make_boxes.py --n 2400 --min-m 125 --max-m 125 --boroughs queens --test-frac 0.0834 --test-nta flushing --out boxes.csv
      (200 boxes named flushing_* inside the "flushing" NTAs, the rest in Queens away from them)
"""
import argparse
import math
import os
import random
import urllib.request


import geopandas as gpd
import shapely
from shapely.geometry import Point, Polygon, box

NTA_URL = ("https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/NYC_Neighborhood_Tabulation_Areas_2020/"
           "FeatureServer/0/query?where=1=1&outFields=*&outSR=4326&f=pgeojson")

BOROUGHS = {  # rough (lon, lat) outlines
    "manhattan": [(-74.016, 40.701), (-73.995, 40.707), (-73.973, 40.711), (-73.971, 40.744),
                  (-73.958, 40.759), (-73.940, 40.784), (-73.930, 40.800), (-73.910, 40.874),
                  (-73.927, 40.878), (-73.947, 40.850), (-73.962, 40.818), (-73.991, 40.781),
                  (-74.008, 40.757), (-74.012, 40.741), (-74.016, 40.717)],
    "brooklyn": [(-74.000, 40.703), (-73.962, 40.736), (-73.925, 40.730), (-73.903, 40.695),
                 (-73.868, 40.695), (-73.856, 40.650), (-73.866, 40.628), (-73.900, 40.610),
                 (-73.935, 40.580), (-74.010, 40.576), (-74.034, 40.605), (-74.040, 40.636),
                 (-74.020, 40.660), (-74.012, 40.682)],
    "queens": [(-73.958, 40.745), (-73.935, 40.778), (-73.905, 40.783), (-73.860, 40.768),
               (-73.830, 40.785), (-73.780, 40.792), (-73.765, 40.770), (-73.701, 40.750),
               (-73.725, 40.655), (-73.760, 40.625), (-73.790, 40.640), (-73.830, 40.650),
               (-73.856, 40.650), (-73.868, 40.695), (-73.903, 40.695), (-73.925, 40.730)],
    "bronx": [(-73.933, 40.808), (-73.900, 40.802), (-73.860, 40.808), (-73.830, 40.808),
              (-73.795, 40.812), (-73.790, 40.840), (-73.810, 40.875), (-73.840, 40.905),
              (-73.900, 40.913), (-73.915, 40.880), (-73.930, 40.835)],
    "staten_island": [(-74.070, 40.645), (-74.180, 40.645), (-74.205, 40.600), (-74.250, 40.505),
                      (-74.200, 40.512), (-74.130, 40.545), (-74.060, 40.597)],
}


def load_nta(path):
    """NYC 2020 NTAs; downloaded to `path` if missing."""
    if not os.path.exists(path):
        urllib.request.urlretrieve(NTA_URL, path)
    return gpd.read_file(path)


def sample_boxes(rng, polys, n, min_m, max_m, land, max_overlap=0.5, start=0, avoid=None):
    """n boxes (name, min_lon, min_lat, max_lon, max_lat, w_m, h_m) centred inside polys (area-weighted),
    skipping boxes that are less than half `land`, touch the geometry `avoid`, or share more than
    `max_overlap` of their area with an earlier box."""
    names, weights = list(polys), [polys[k].area for k in polys]
    rows, taken = [], []  # taken: the boxes so far
    tries = 0
    while len(rows) < n:
        tries += 1
        assert tries < 1000 * n, "boxes don't fit: lower --n / --test-frac or raise --max-overlap"
        boro = rng.choices(names, weights)[0]
        minx, miny, maxx, maxy = polys[boro].bounds
        lon, lat = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if not polys[boro].contains(Point(lon, lat)):
            continue
        # log-uniform size so small and large boxes are both well represented
        # creates box around center point
        w = math.exp(rng.uniform(math.log(min_m), math.log(max_m)))
        h = w * rng.uniform(0.75, 1.33)
        # half the box size in degrees (1 deg latitude ~ 111,320 m; longitude shrinks by cos(lat))
        dlat = h / 111_320 / 2
        dlon = w / (111_320 * math.cos(math.radians(lat))) / 2
        b = box(lon - dlon, lat - dlat, lon + dlon, lat + dlat)
        if land.intersection(b).area < 0.5 * b.area or (avoid is not None and avoid.intersects(b)):
            continue
        # skip if an earlier box covers more than max_overlap of this one
        if taken and shapely.area(shapely.intersection(taken, b)).max() > max_overlap * b.area:
            continue
        taken.append(b)
        rows.append((f"{boro}_{start + len(rows) + 1:05d}", lon - dlon, lat - dlat,
                     lon + dlon, lat + dlat, round(w), round(h)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--min-m", type=float, default=150, help="smallest box side, meters")
    p.add_argument("--max-m", type=float, default=2500, help="largest box side, meters")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="boxes.csv")
    p.add_argument("--boroughs", nargs="+", default=list(BOROUGHS), choices=list(BOROUGHS))
    p.add_argument("--test-frac", type=float, default=None,
                   help="fraction of boxes in the held-out borough (e.g. 0.1); default: all boroughs area-weighted")
    p.add_argument("--holdout", default="queens", choices=list(BOROUGHS), help="held-out borough for --test-frac")
    p.add_argument("--test-nta", default=None,
                   help="with --test-frac: hold out the NTAs whose name contains this text instead of a borough")
    p.add_argument("--nta-file", default="data/nyc_nta_2020.geojson", help="NYC 2020 NTAs, for land and --test-nta (downloaded if missing)")
    p.add_argument("--osm-file", default="osm_data/NewYork.gpkg", help="OSM GeoPackage, for water")
    p.add_argument("--max-overlap", type=float, default=0.5,
                   help="skip boxes sharing more than this fraction of their area with an earlier box (1 = no limit)")
    a = p.parse_args()

    rng = random.Random(a.seed)
    polys = {k: Polygon(v) for k, v in BOROUGHS.items() if k in a.boroughs}
    nta = load_nta(a.nta_file)
    water = gpd.read_file(a.osm_file, layer="multipolygons", where="natural = 'water'")
    land = nta.union_all().difference(water.union_all())  # every NTA, minus lakes etc.
    if a.test_frac is None:
        rows = sample_boxes(rng, polys, a.n, a.min_m, a.max_m, land, a.max_overlap)
    else:
        n_test = round(a.n * a.test_frac)
        if a.test_nta:
            match = nta[nta.NTAName.str.contains(a.test_nta, case=False)]
            print("test NTAs:", ", ".join(match.NTAName))
            area = match.union_all()
            rows = sample_boxes(rng, polys, a.n - n_test, a.min_m, a.max_m, land, a.max_overlap, avoid=area)
            test = {a.test_nta.lower().replace(" ", "_"): area}
        else:
            rows = sample_boxes(rng, {k: v for k, v in polys.items() if k != a.holdout}, a.n - n_test, a.min_m, a.max_m, land, a.max_overlap)
            test = {a.holdout: polys[a.holdout]}
        rows += sample_boxes(rng, test, n_test, a.min_m, a.max_m, land, a.max_overlap, start=len(rows))

    with open(a.out, "w") as f:
        f.write("name,min_lon,min_lat,max_lon,max_lat,width_m,height_m\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]},{r[6]}\n")
    print(f"wrote {len(rows)} boxes to {a.out}")


if __name__ == "__main__":
    main()
