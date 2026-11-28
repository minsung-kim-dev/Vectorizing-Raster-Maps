#!/usr/bin/env bash
# Predict + evaluate the latest checkpoint. Usage (raster2seq env): bash tools/raster2seq_modified/evaluation/predict_and_evaluate_stage2.sh [job] [data] [split]
ulimit -n "$(ulimit -Hn)"
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO"
SPLIT=test
P=${P:-4}                                                # evaluations at a time
PLOT=yes
SEMANTIC_CLASSES=$(cd "$REPO" && python -c "from data_preprocess.create_coco_osm import CATEGORIES; print(len(CATEGORIES) + 1)")
FOLDER=/share/garg/jrg377/osm_stage2_checkpoints

## job  epoch (counted from 1; "last" = checkpoint.pth)  sem_attn
CHECKPOINTS=(
    # "osm_stage2_queens_4000pretrain_smallv2  200   yes      coco_osm_fixed      256     yes"
    # "osm_stage2_queens_4000pretrain_smallv2  last  yes      coco_osm_fixed      256     yes"
    # "osm_stage2_queens_4000pretrain_nodrop   last  yes      coco_osm_fixed      256     yes"
    # "osm_stage2_queens_4000pretrain_tuned    100   yes      coco_osm_fixed      256     yes"
    # "osm_stage2_queens_4000pretrain_tuned    200   yes      coco_osm_fixed      256     yes"
    # "osm_stage2_queens_4000pretrain_tuned    300   yes      coco_osm_fixed      256     yes"
    # "osm_stage2_queens_4000pretrain_tuned    last  yes      coco_osm_fixed      256     yes"
    # "osm_stage2_queens_4000pretrain_tuned_v2 last  yes      coco_osm_fixed      256     yes"
    "osm_stage2_queens_4000pretrain_tuned_v3 last  yes      coco_osm_fixed      256     yes"
    # "osm_stage2_queens_4000_gray             last  yes      coco_osm_gray       256     yes"
    # "osm_stage2_queens_4000_512px1024seq     last  yes      coco_osm_fixed      512     yes"
    # "osm_stage2_queens_4000pretrain_smallv2  last  yes      coco_osm_gray       256     yes"
    # "osm_stage2_queens_4000pretrain_smallv2  200   yes      coco_osm_fixed      256     no"
    # "osm_stage2_queens_4000pretrain_tuned_v3 last  yes      coco_osm_fixed      256     no"
    # "osm_stage2_queens_4000_batch32          last  yes      coco_osm_fixed      256     yes"
)

# initialize basic stage 2 model
MODEL=(--dataset_name=osm --semantic_classes=$SEMANTIC_CLASSES --per_token_sem_loss --input_channels 3
       --poly2seq --num_bins 32 --row_tolerance 0.08 --converter_version v3
       --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval --add_geom_head)

# function to run evaluation
evaluate() {  # one list entry: "job epoch sem_attn" -> eval_outputs/<job>/<split>/epoch_<epoch>/
    local job epoch attn data size ckpt out mask
    read -r job epoch attn data size mask <<< "$1"
    local res=(--image_size 256 --seq_len 512); [[ $size == 512 ]] && res=(--image_size 512 --seq_len 1024)
    local suffix=""
    [[ $data == *gray* ]] && suffix+=_gray
    [[ $mask == no ]] && suffix+=_nomask

    local plots=(--plot_pred '' --plot_density '' --plot_gt ''); [[ $PLOT == yes ]] && plots=()
    
    if [[ $epoch == last ]]; then
        ckpt=$FOLDER/$job/checkpoint.pth
    else
        ckpt=$FOLDER/$job/checkpoint$(printf %04d $((epoch - 1))).pth   # epoch 200 -> checkpoint0199.pth
    fi
    out=eval_outputs/$job/${SPLIT}/epoch${suffix}_$epoch
    local sem_attn=(); [[ $attn == yes ]] && sem_attn=(--sem_attn)   # the model must match how the run was trained
    local geom_mask=(--geom_mask); [[ $mask == no ]] && geom_mask=()
    rm -rf "$out"; mkdir -p "$out"
    if python ../Raster2Seq/eval.py --checkpoint "$ckpt" --dataset_root "/share/garg/jrg377/$data" --eval_set "$SPLIT" --output_dir "$out" \
        "${MODEL[@]}" "${res[@]}" "${sem_attn[@]}" "${geom_mask[@]}" "${plots[@]}" --batch_size 32 --num_workers 0 \
        --filter_gt_by_area --min_poly_area 25 --per_label_eval --save_pred > "$out/log.txt" 2>&1; then
        echo "done: $job epoch $epoch"
    else
        echo "FAILED: $job epoch $epoch (see $out/log.txt)"
    fi

    if [[ $PLOT == yes ]]; then
        if python ../Raster2Seq/predict.py --checkpoint "$ckpt" --dataset_root "/share/garg/jrg377/$data/$SPLIT" \
            --output_dir "$out/predict" "${MODEL[@]}" "${res[@]}" "${sem_attn[@]}" "${geom_mask[@]}" \
            --min_poly_area 25 --line_classes 0 3 7 16 22 --save_pred > "$out/predict_log.txt" 2>&1; then
            echo "predicted: $job epoch $epoch"
        else
            echo "PREDICT FAILED: $job epoch $epoch (see $out/predict_log.txt)"
        fi
    fi
}

# loop through
for entry in "${CHECKPOINTS[@]}"; do
    read -r job epoch _ <<< "$entry"
    [[ $epoch == last ]] && ckpt=$FOLDER/$job/checkpoint.pth || ckpt=$FOLDER/$job/checkpoint$(printf %04d $((epoch - 1))).pth
    if [[ ! -f $ckpt ]]; then echo "skip: no $ckpt"; continue; fi
    echo "start: $job epoch $epoch"
    evaluate "$entry" &
    while (( $(jobs -rp | wc -l) >= ${P:-2} )); do sleep 5; done   # at most P at a time
done
wait
