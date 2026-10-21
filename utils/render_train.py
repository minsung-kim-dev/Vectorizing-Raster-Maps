#!/usr/bin/env python3
"""Batch-render the train10000 GeoJSON files to PNG using render_carto.py.

Usage:
    python render_train.py                      # all files, source zoom, 2x scale
    python render_train.py --limit 10           # quick test on 10 maps
    python render_train.py --scale 1 --workers 4 --out data/train10000_png

Already-rendered PNGs are skipped, so the script can be stopped and resumed.
Per-map render info (zoom, pixel window, size) is written to <out>/renders.jsonl.
"""
import argparse, json, os, sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "data" / "osm_carto_compare"))
import render_carto as rc


def render_one(job):
    src, out, zoom, scale = job
    try:
        info = rc.render(str(src), str(out), zoom, scale)
        return {"map_id": src.name.split(".")[0], "png": out.name, **info}
    except Exception as e:
        return {"map_id": src.name.split(".")[0], "error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=ROOT / "data" / "train10000", type=Path)
    ap.add_argument("--out", default=ROOT / "data" / "train10000_png", type=Path)
    ap.add_argument("--zoom", type=int, help="override each map's source zoom")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, help="only render the first N maps")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    srcs = sorted(a.src.glob("*.geojson.gz"))[: a.limit]
    jobs = [(s, a.out / (s.name.split(".")[0] + ".png"), a.zoom, a.scale) for s in srcs]
    jobs = [j for j in jobs if not j[1].exists()]
    print(f"{len(srcs)} maps, {len(srcs) - len(jobs)} already done, {len(jobs)} to render")

    errors = 0
    with Pool(a.workers) as pool, open(a.out / "renders.jsonl", "a") as log:
        for i, rec in enumerate(pool.imap_unordered(render_one, jobs, chunksize=4), 1):
            log.write(json.dumps(rec) + "\n")
            if "error" in rec:
                errors += 1
                print(f"  FAILED {rec['map_id']}: {rec['error']}")
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} rendered ({errors} errors)")


if __name__ == "__main__":
    main()
