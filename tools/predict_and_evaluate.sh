#!/usr/bin/env bash
# Predict + evaluate the latest checkpoint. Usage (raster2seq env): bash tools/predict_eval_osm.sh [job] [data] [split]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
JOB=${1:-osm_stage-full} DATA=${2:-data/coco_osm_mini} SPLIT=${3:-train}
CKPT=save_models/$JOB/checkpoint.pth
MODEL=(--dataset_name=osm --semantic_classes=3 --geom_classes --per_token_sem_loss --input_channels 3
       --image_size 512 --poly2seq --seq_len 1024 --num_bins 32
       --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval)

python ../Raster2Seq/predict.py --checkpoint "$CKPT" --dataset_root "$DATA/$SPLIT" --output_dir predictions "${MODEL[@]}"
# python ../Raster2Seq/eval.py --checkpoint "$CKPT" --dataset_root "$DATA" --eval_set "$SPLIT" --row_tolerance 0.08 --output_dir "eval_outputs/$JOB" "${MODEL[@]}" --plot_pred '' --plot_density '' --plot_gt ''