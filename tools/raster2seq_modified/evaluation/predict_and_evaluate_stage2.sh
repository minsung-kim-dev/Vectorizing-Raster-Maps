#!/usr/bin/env bash
# Predict + evaluate the latest checkpoint. Usage (raster2seq env): bash tools/raster2seq_modified/evaluation/predict_and_evaluate_stage2.sh [job] [data] [split]
ulimit -n "$(ulimit -Hn)"
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
R2S=${R2S:-$REPO/../Raster2Seq}
DATA=${DATA:-/share/garg/jrg377/coco_osm_fixed}
RUNS_DIR=${RUNS_DIR:-/share/garg/jrg377/osm_tuning}      # sweep's OUTPUT_DIR
EVAL_DIR=${EVAL_DIR:-$RUNS_DIR/eval_val}                 # results: $EVAL_DIR/<run>.json
SPLIT=test
P=${P:-4}                                                # evaluations at a time
SEMANTIC_CLASSES=$(cd "$REPO" && python -c "from data_preprocess.create_coco_osm import CATEGORIES; print(len(CATEGORIES) + 1)")
FOLDER=/share/garg/jrg377/osm_stage2_checkpoints

## job  epoch (counted from 1; "last" = checkpoint.pth)  sem_attn
CHECKPOINTS=(
    "osm_stage2_queens_4000pretrain_smallv2  200   yes"
    "osm_stage2_queens_4000pretrain_smallv2  last  yes"
    "osm_stage2_queens_4000pretrain_nodrop   last  yes"
    "osm_stage2_queens_4000pretrain_tuned    100   yes"
    "osm_stage2_queens_4000pretrain_tuned    200   yes"
    "osm_stage2_queens_4000pretrain_tuned    300   yes"
    "osm_stage2_queens_4000pretrain_tuned    last  yes"
)

# initialize basic stage 2 model
MODEL=(--dataset_name=osm --semantic_classes=$SEMANTIC_CLASSES --per_token_sem_loss --input_channels 3
       --image_size 256 --poly2seq --seq_len 512 --num_bins 32 --row_tolerance 0.08 --converter_version v3
       --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval --add_geom_head --geom_mask)

# function to run evaluation
evaluate() {  # one list entry: "job epoch sem_attn" -> eval_outputs/<job>/<split>/epoch_<epoch>/
    local job epoch attn ckpt out
    read -r job epoch attn <<< "$1"
    if [[ $epoch == last ]]; then
        ckpt=$FOLDER/$job/checkpoint.pth
    else
        ckpt=$FOLDER/$job/checkpoint$(printf %04d $((epoch - 1))).pth   # epoch 200 -> checkpoint0199.pth
    fi
    out=eval_outputs/$job/$SPLIT/epoch_$epoch
    local sem_attn=(); [[ $attn == yes ]] && sem_attn=(--sem_attn)   # the model must match how the run was trained
    rm -rf "$out"; mkdir -p "$out"
    if python ../Raster2Seq/eval.py --checkpoint "$ckpt" --dataset_root "$DATA" --eval_set "$SPLIT" --output_dir "$out" \
        "${MODEL[@]}" "${sem_attn[@]}" --plot_pred '' --plot_density '' --plot_gt '' --batch_size 32 --num_workers 0 \
        --filter_gt_by_area --min_poly_area 25 --per_label_eval > "$out/log.txt" 2>&1; then
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
