#!/usr/bin/env bash
# Rescore a run's numbered checkpoints (checkpoint0099.pth, ...) on the validation set with the current eval.py, so
# its validation curve uses today's scoring rules. Writes eval_outputs/<job>/by_epoch/<epoch>.json (eval.py's
# results.txt; epoch counted from 1), for plot_losses.py --rescored. Already-scored epochs are skipped.
#
# Usage (raster2seq env, GPU node, from the repo root):
#   bash tools/raster2seq_modified/evaluation/rescore_checkpoints.sh <run folder> [every]
#   e.g. ... /share/garg/jrg377/osm_stage2_checkpoints/osm_stage2_queens_4000pretrain_nodrop 50
# [every]: only checkpoints whose epoch is a multiple of this (default: all). P evaluations run at a time.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD
RUN=${1:?run folder}
EVERY=${2:-1}
DATA=${DATA:-/share/garg/jrg377/coco_osm_fixed}
SPLIT=${SPLIT:-val}
P=${P:-2}
OUT=eval_outputs/$(basename "$RUN")/by_epoch
SEMANTIC_CLASSES=$(python -c "from data_preprocess.create_coco_osm import CATEGORIES; print(len(CATEGORIES) + 1)")

# the run's model settings: stage 2 (labels); for a stage 1 run use the second MODEL instead
MODEL=(--dataset_name=osm --semantic_classes=$SEMANTIC_CLASSES --per_token_sem_loss --input_channels 3
       --image_size 256 --poly2seq --seq_len 512 --num_bins 32 --converter_version v3
       --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval --add_geom_head --geom_mask --sem_attn)
# MODEL=(--dataset_name=osm --semantic_classes=3 --geom_classes --per_token_sem_loss --input_channels 3
#        --image_size 256 --poly2seq --seq_len 512 --num_bins 32 --converter_version v3
#        --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval)

score() {  # one checkpoint: eval.py writes to a temporary folder; keep results.txt as <epoch>.json
    local ckpt=$1 epoch=$2 tmp
    tmp=$(mktemp -d)
    if python ../Raster2Seq/eval.py --checkpoint "$ckpt" --dataset_root "$DATA" --eval_set "$SPLIT" --output_dir "$tmp" \
        "${MODEL[@]}" --plot_pred '' --plot_density '' --plot_gt '' --num_workers 0 > "$tmp/log.txt" 2>&1; then
        cp "$tmp/results.txt" "$OUT/$epoch.json"; echo "done: epoch $epoch"
    else
        cp "$tmp/log.txt" "$OUT/$epoch.failed.log"; echo "FAILED: epoch $epoch (see $OUT/$epoch.failed.log)"
    fi
    rm -rf "$tmp"
}

mkdir -p "$OUT"
for ckpt in "$RUN"/checkpoint[0-9]*.pth; do
    n=$(basename "$ckpt" .pth); epoch=$((10#${n#checkpoint} + 1))   # checkpoint0099.pth: end of epoch 100
    if (( epoch % EVERY )); then continue; fi
    if [[ -f "$OUT/$epoch.json" ]]; then echo "skip epoch $epoch: already scored"; continue; fi
    score "$ckpt" "$epoch" &
    while (( $(jobs -rp | wc -l) >= P )); do sleep 5; done   # at most P at a time
done
wait
echo "plot: python tools/raster2seq_modified/training/plot_losses.py $RUN --rescored"
