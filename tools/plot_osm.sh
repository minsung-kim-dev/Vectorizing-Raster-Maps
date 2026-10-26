#!/usr/bin/env bash
# Plot the ground truth of data/coco_osm with Raster2Seq's plot_floor.py
# (adapted from Raster2Seq's data_preprocess/tools/plot_data.sh). Runs on CPU.
#
# Usage (from anywhere, in the raster2seq env):
#   bash tools/plot_osm.sh            # val split
#   bash tools/plot_osm.sh test       # another split
#   DEBUG=1 bash tools/plot_osm.sh    # a single image, for a quick check
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

# vocabulary size + 1 (Raster2Seq reserves the last class), read from create_coco_osm.py
SEM_CLASSES=$(cd "$REPO" && python -c "from data_preprocess.create_coco_osm import CATEGORIES; print(len(CATEGORIES) + 1)")

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
                   ${DEBUG:+--debug}
done
