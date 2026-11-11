#!/usr/bin/env bash
# Raster2Seq training pipeline for the OSM crops, modelled on Raster2Seq's tools/train_cc5k.sh
# (stage 1, shapes only) and tools/finetune_cc5k.sh (stage 2, labels).
#
# Usage (in the raster2seq env):  bash tools/finetune_osm.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
R2S=${R2S:-$REPO/../Raster2Seq}
CROPS=${CROPS:-$REPO/crops}
DATA=${DATA:-$REPO/data/coco_osm_flushing}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/save_models}

IMAGE_SIZE=512    
SEQ_LEN=1024      
NUM_BINS=32       # matches Raster2Seq's CubiCasa checkpoint used as the starting point
ROW_TOL=0.08      # shape order: rows 8% of the image height, top to bottom, left to right (needs the fork)
CLS_COEFF=5
COO_COEFF=20
SEM_COEFF=1
CONVERTER=v3
NUM_GPUS=1
BATCH_SIZE=8      # lower if the GPU runs out of memory
EPOCHS=200
# vocabulary size + 1 (Raster2Seq reserves the last class), read from create_coco_osm.py
SEM_CLASSES=$(cd "$REPO" && python -c "from data_preprocess.create_coco_osm import CATEGORIES; print(len(CATEGORIES) + 1)")

# ---- 1. COCO dataset (skipped if it already exists) ----
if [[ ! -d "$DATA/annotations" ]]; then
    (cd "$REPO" && python data_preprocess/create_coco_osm.py --data_root "$CROPS" --output "$DATA")
fi

# ---- 5. Stage 2: shapes + labels, starting from stage 1 (GPU) ----
cd "$R2S"
torchrun --nproc_per_node=${NUM_GPUS} --master_port=14159 main_ddp.py \
    --dataset_name=osm \
    --dataset_root="$DATA" \
    --semantic_classes=${SEM_CLASSES} \
    --job_name=osm_stage2_queens_4000pretrain \
    --output_dir "$OUTPUT_DIR" \
    --batch_size ${BATCH_SIZE} \
    --input_channels=3 \
    --image_size ${IMAGE_SIZE} \
    --poly2seq \
    --seq_len ${SEQ_LEN} \
    --row_tolerance ${ROW_TOL} \
    --num_bins ${NUM_BINS} \
    --ckpt_every_epoch=10 \
    --eval_every_epoch=250 \
    --label_smoothing 0.1 \
    --epochs ${EPOCHS} \
    --lr_drop '100' \
    --cls_loss_coef ${CLS_COEFF} \
    --coords_loss_coef ${COO_COEFF} \
    --room_cls_loss_coef ${SEM_COEFF} \
    --resume "$OUTPUT_DIR/osm_stage2_queens_4000pretrain/checkpoint.pth" \
    --ema4eval \
    --disable_poly_refine \
    --dec_attn_concat_src \
    --per_token_sem_loss \
    --jointly_train \
    --converter_version ${CONVERTER} \
    --use_anchor \
    --add_geom_head \
    --geom_mask \
    --disable_wd_as_line \
    --start_from_checkpoint "$OUTPUT_DIR/osm_stage_queens4000_pretrain/checkpoint.pth"