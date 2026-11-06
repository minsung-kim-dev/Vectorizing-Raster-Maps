#!/usr/bin/env bash
# Small COCO dataset (data/coco_osm_mini) from every 375th crop in crops_125m (40 of 15,000). Run in the raster2seq env.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -e data/coco_osm_mini ]] && { echo "data/coco_osm_mini already exists; move or delete it first" >&2; exit 1; }

SRC=$(mktemp -d "${TMPDIR:-/tmp}/make_mini.XXXXXX"); trap 'rm -rf "$SRC"' EXIT
for f in $(find "$PWD/crops_125m/" -name "*.png" | sort | awk 'NR % 375 == 1'); do  # find: too many files for ls
    ln -s "$f" "${f%.png}.geojson.gz" "$SRC/"
done
python data_preprocess/create_coco_osm.py --data_root "$SRC" --output data/coco_osm_mini
(cd data && zip -qr coco_osm_mini.zip coco_osm_mini)
