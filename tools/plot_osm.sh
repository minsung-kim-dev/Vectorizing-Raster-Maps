#!/usr/bin/env bash
# Plot the ground truth of data/coco_osm with Raster2Seq's plot_floor.py
# (adapted from Raster2Seq's data_preprocess/tools/plot_data.sh). Runs on CPU.
#
# Usage (from anywhere, in the raster2seq env):
#   bash tools/plot_osm.sh            # first 5 images of the val split
#   bash tools/plot_osm.sh test       # another split
#   N=20 bash tools/plot_osm.sh       # first 20 images
#   N=all bash tools/plot_osm.sh      # every image in the split
#
# Outputs per image: --plot_gt (shapes drawn from the annotations), --plot_gt_image (the input
# image after resize/pad), --plot_density (shapes overlaid on the input image; despite the name it
# works for any image, not just density maps). --compute_stats prints the max number of points /
# shapes per image (useful for --seq_len) and saves histograms (needs kaleido).
#
# Each plot type runs as its own plot_floor.py call: Raster2Seq's loader resizes annotations in
# place, so loading the same image twice in one process shrinks its shapes again.
# --num_workers 0: on macOS, worker processes can't pickle plot_floor.py's local batch collator.
set -euo pipefail

SPLIT=${1:-val}
REPO=$(cd "$(dirname "$0")/.." && pwd)
R2S=${R2S:-$REPO/../Raster2Seq}
DATA=${DATA:-$REPO/data/coco_osm}
OUT=${OUT:-$REPO/data_plots/osm/$SPLIT}
N=${N:-5}

# vocabulary size + 1 (Raster2Seq reserves the last class), read from create_coco_osm.py
SEM_CLASSES=$(cd "$REPO" && python -c "from data_preprocess.create_coco_osm import CATEGORIES; print(len(CATEGORIES) + 1)")

# N: plot only the first N images, via a temporary dataset root (the split's annotations cut down
# to N images, plus a link to the split's PNG folder). plot_floor.py has no option for this.
if [[ "$N" != "all" ]]; then
    SUB=$(mktemp -d "${TMPDIR:-/tmp}/plot_osm.XXXXXX")  # macOS mktemp ignores TMPDIR without a template
    trap 'rm -rf "$SUB"' EXIT
    mkdir "$SUB/annotations"
    ln -s "$DATA/$SPLIT" "$SUB/$SPLIT"
    python - "$DATA/annotations/$SPLIT.json" "$SUB/annotations/$SPLIT.json" "$N" <<'EOF'
import json, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(src))
d["images"] = sorted(d["images"], key=lambda im: im["id"])[:n]
keep = {im["id"] for im in d["images"]}
d["annotations"] = [a for a in d["annotations"] if a["image_id"] in keep]
json.dump(d, open(dst, "w"))
EOF
    DATA=$SUB
fi

cd "$R2S"  # plot_floor.py imports Raster2Seq's modules relative to its repo
for flag in --plot_gt --plot_gt_image --plot_density --compute_stats; do
    python plot_floor.py --dataset_name=waffle \
                   --dataset_root="$DATA" \
                   --eval_set="$SPLIT" \
                   --output_dir="$OUT" \
                   --semantic_classes="$SEM_CLASSES" \
                   --input_channels 3 \
                   --disable_image_transform \
                   --poly2seq \
                   --seq_len 1024 \
                   --num_bins 64 \
                   --image_size 512 \
                   --image_scale 1 \
                   --num_workers 0 \
                   $flag \
    || [[ $flag == --compute_stats ]]  # stats print first; only their histogram export (needs Chrome) may fail
done
