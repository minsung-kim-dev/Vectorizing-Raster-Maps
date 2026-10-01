#!/usr/bin/env python3
"""
make_boxes.py: generate random NYC bounding boxes of varying sizes for osm_crops.py.

Box centers are sampled inside rough, hand-drawn outlines of the five boroughs,
so boxes land mostly on land. The outlines are approximate: boxes near the
shore will include some water, and a few may spill slightly into NJ.

Usage:
  python make_boxes.py --n 3000 --min-m 150 --max-m 2500 --seed 42 --out boxes.csv
"""
import argparse
import math
import random



class Polygon:
    """Tiny polygon helper (no shapely needed)."""
    def __init__(self, pts):
        self.pts = pts
        xs, ys = zip(*pts)
        self.bounds = (min(xs), min(ys), max(xs), max(ys))
        self.area = abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1)
                            in zip(pts, pts[1:] + pts[:1]))) / 2

    def contains(self, x, y):  # ray casting
        inside, pts = False, self.pts
        for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
            if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
                inside = not inside
        return inside

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--min-m", type=float, default=150, help="smallest box side, meters")
    p.add_argument("--max-m", type=float, default=2500, help="largest box side, meters")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="boxes.csv")
    a = p.parse_args()

    rng = random.Random(a.seed)
    polys = {k: Polygon(v) for k, v in BOROUGHS.items()}
    names, weights = list(polys), [polys[k].area for k in polys]  # area-weighted

    rows = []
    while len(rows) < a.n:
        boro = rng.choices(names, weights)[0]
        poly = polys[boro]
        minx, miny, maxx, maxy = poly.bounds
        lon, lat = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if not poly.contains(lon, lat):
            continue
        # log-uniform size so small and large boxes are both well represented
        w = math.exp(rng.uniform(math.log(a.min_m), math.log(a.max_m)))
        h = w * rng.uniform(0.75, 1.33)
        dlat = h / 111_320 / 2
        dlon = w / (111_320 * math.cos(math.radians(lat))) / 2
        rows.append((f"{boro}_{len(rows) + 1:05d}", lon - dlon, lat - dlat,
                     lon + dlon, lat + dlat, round(w), round(h)))

    with open(a.out, "w") as f:
        f.write("name,min_lon,min_lat,max_lon,max_lat,width_m,height_m\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]},{r[6]}\n")
    print(f"wrote {len(rows)} boxes to {a.out}")


if __name__ == "__main__":
    main()
