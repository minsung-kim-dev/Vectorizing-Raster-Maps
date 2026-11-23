#!/usr/bin/env bash
# Predict + evaluate the latest checkpoint. Usage (raster2seq env): bash tools/raster2seq_modified/evaluation/predict_and_evaluate_stage2.sh [job] [data] [split]
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD
JOB=${1:-test_flushing_patches}
# DATA=${2:-/share/garg/jrg377/coco_roosevelt}
DATA=${2:-/share/garg/jrg377/coco_flushing_grid}
SPLIT=${3:-test}
CKPT=${CKPT:-/share/garg/jrg377/osm_stage2_checkpoints/osm_stage2_queens_4000pretrain_smallv2/checkpoint0399.pth}
SEMANTIC_CLASSES=$(python -c "from data_preprocess.create_coco_osm import CATEGORIES; print(len(CATEGORIES) + 1)")

MODEL=(--dataset_name=osm --semantic_classes=$SEMANTIC_CLASSES --per_token_sem_loss --input_channels 3
       --image_size 256 --poly2seq --seq_len 512 --num_bins 32 --row_tolerance 0.08 --converter_version v3
       --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval --add_geom_head --geom_mask --sem_attn )

python ../Raster2Seq/predict.py --checkpoint "$CKPT" --dataset_root "$DATA/$SPLIT" --output_dir "predictions/$JOB" "${MODEL[@]}"  --min_poly_area 25 --line_classes 0 3 7 16 22 --save_pred
# python ../Raster2Seq/predict.py --checkpoint "$CKPT" --dataset_root /share/garg/jrg377/roosevelt_screenshot \
#  --output_dir "predictions/${JOB}_screenshot" "${MODEL[@]}"  --min_poly_area 25 --line_classes 0 3 7 16 22 --save_pred
python ../Raster2Seq/eval.py --checkpoint "$CKPT" --dataset_root "$DATA" --eval_set "$SPLIT" --row_tolerance 0.08 --converter_version v3 --output_dir "eval_outputs/$JOB" "${MODEL[@]}" --filter_gt_by_area --min_poly_area 25 --per_label_eval --save_pred --disable_image_transform