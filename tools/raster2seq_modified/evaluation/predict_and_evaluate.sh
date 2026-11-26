#!/usr/bin/env bash
# Predict + evaluate the latest checkpoint. Usage (raster2seq env): bash tools/raster2seq_modified/evaluation/predict_and_evaluate_stage2.sh [job] [data] [split]
ulimit -n "$(ulimit -Hn)"
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO"
R2S=${R2S:-$REPO/../Raster2Seq}
DATA=${DATA:-/share/garg/jrg377/coco_osm_fixed}
SPLIT=test
P=${P:-4}                                                # evaluations at a time
SEMANTIC_CLASSES=$(cd "$REPO" && python -c "from data_preprocess.create_coco_osm import CATEGORIES; print(len(CATEGORIES) + 1)")
FOLDER=/share/garg/jrg377/osm_stage1_checkpoints

## job  epoch (counted from 1; "last" = checkpoint.pth)
CHECKPOINTS=(
    "osm_stage_queens4000_pretrain_smallv2  100"
    # "osm_stage_queens4000_pretrain_smallv2  200"
    # "osm_stage_queens4000_pretrain_smallv2  last"
    "osm_stage_queens4000_pretrain_tuned    100"
    # "osm_stage_queens4000_pretrain_tuned    200"
    # "osm_stage_queens4000_pretrain_tuned    last"
)

# initialize basic stage 2 model
MODEL=(--dataset_name=osm --semantic_classes=3 --geom_classes --per_token_sem_loss --input_channels 3
       --image_size 256 --poly2seq --seq_len 512 --num_bins 32
       --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval)

# function to run evaluation
evaluate() {  # one list entry: "job epoch" -> eval_outputs/<job>/<split>/epoch_<epoch>/
    local job epoch ckpt out
    read -r job epoch <<< "$1"
    if [[ $epoch == last ]]; then
        ckpt=$FOLDER/$job/checkpoint.pth
    else
        ckpt=$FOLDER/$job/checkpoint$(printf %04d $((epoch - 1))).pth   # epoch 200 -> checkpoint0199.pth
    fi
    out=eval_outputs/$job/$SPLIT/epoch_$epoch
    rm -rf "$out"; mkdir -p "$out"
    if python ../Raster2Seq/eval.py --checkpoint "$ckpt" --dataset_root "$DATA" --eval_set "$SPLIT" --output_dir "$out" \
        "${MODEL[@]}" --plot_pred '' --plot_density '' --plot_gt ''  --num_workers 0 \
        --filter_gt_by_area --min_poly_area 25 > "$out/log.txt" 2>&1; then
        echo "done: $job epoch $epoch"
    else
        echo "FAILED: $job epoch $epoch (see $out/log.txt)"
    fi
}

# loop through
for entry in "${CHECKPOINTS[@]}"; do
    read -r job epoch attn <<< "$entry"
    [[ $epoch == last ]] && ckpt=$FOLDER/$job/checkpoint.pth || ckpt=$FOLDER/$job/checkpoint$(printf %04d $((epoch - 1))).pth
    if [[ ! -f $ckpt ]]; then echo "skip: no $ckpt"; continue; fi
    echo "start: $job epoch $epoch"
    evaluate "$entry" &
    while (( $(jobs -rp | wc -l) >= ${P:-2} )); do sleep 5; done   # at most P at a time
done
wait
