#!/usr/bin/env bash
# Rescore a run's numbered checkpoints (checkpoint0099.pth, ...) on the validation set with the current eval.py
set -euo pipefail
ulimit -n "$(ulimit -Hn)"

cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD
RUN=/share/garg/jrg377/osm_stage1_checkpoints/osm_stage_queens4000_pretrain_tuned
EVERY=${1:-1}
DATA=${DATA:-/share/garg/jrg377/coco_osm_fixed}
SPLIT=${SPLIT:-val}
P=${P:-4}
OUT=eval_outputs/$(basename "$RUN")/by_epoch
SEMANTIC_CLASSES=$(python -c "from data_preprocess.create_coco_osm import CATEGORIES; print(len(CATEGORIES) + 1)")

# the run's model settings: stage 2 (labels); for a stage 1 run use the second MODEL instead
# MODEL=(--dataset_name=osm --semantic_classes=$SEMANTIC_CLASSES --per_token_sem_loss --input_channels 3
#        --image_size 256 --poly2seq --seq_len 512 --num_bins 32 --converter_version v3
#        --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval --add_geom_head --geom_mask --sem_attn)
MODEL=(--dataset_name=osm --semantic_classes=3 --geom_classes --per_token_sem_loss --input_channels 3
       --image_size 256 --poly2seq --seq_len 512 --num_bins 32 --converter_version v3
       --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval)

score() {  # one checkpoint: eval.py writes to a temporary folder; keep results.txt as <epoch>.json
    local ckpt=$1 epoch=$2 tmp
    tmp=$(mktemp -d)
    if python ../Raster2Seq/eval.py --checkpoint "$ckpt" --dataset_root "$DATA" --eval_set "$SPLIT" --output_dir "$tmp" \
        "${MODEL[@]}" --plot_pred '' --plot_density '' --plot_gt '' --batch_size 32 --num_workers 0 > "$tmp/log.txt" 2>&1; then
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
