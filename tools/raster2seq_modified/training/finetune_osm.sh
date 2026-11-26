#!/usr/bin/env bash
# Raster2Seq training pipeline for the OSM crops, modeled on Raster2Seq's tools/train_cc5k.sh
# (stage 1, shapes only) and tools/finetune_cc5k.sh (stage 2, labels).
#
# Usage (in the raster2seq env):  bash tools/raster2seq_modified/training/finetune_osm.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
R2S=${R2S:-$REPO/../Raster2Seq}
CROPS=${CROPS:-$REPO/crops}
DATA=${DATA:-/share/garg/jrg377/coco_osm_gray}
# OUTPUT_DIR=${OUTPUT_DIR:-/share/garg/jrg377/osm_stage2_queens4000_newvocab_checkpoints}
OUTPUT_DIR=${OUTPUT_DIR:-/share/garg/jrg377/osm_stage2_checkpoints}


IMAGE_SIZE=256   
SEQ_LEN=512  
# IMAGE_SIZE=512   
# SEQ_LEN=1024

NUM_BINS=32       # matches Raster2Seq's CubiCasa checkpoint
ROW_TOL=0.08      # shape order: rows 8% of the image height, top to bottom, left to right (needs the fork)
CLS_COEFF=5
COO_COEFF=20
SEM_COEFF=1
CONVERTER=v3
NUM_GPUS=1
BATCH_SIZE=8      # lower if the GPU runs out of memory
EPOCHS=200
# JOB=osm_stage2_queens_4000_512px1024seq
JOB=osm_stage2_queens_4000_gray
# vocabulary size + 1 (Raster2Seq reserves the last class), read from create_coco_osm.py
SEM_CLASSES=$(cd "$REPO" && python -c "from data_preprocess.create_coco_osm import CATEGORIES; print(len(CATEGORIES) + 1)")


# ---- Stage 2 training ----
cd "$R2S"
torchrun --nproc_per_node=${NUM_GPUS} --master_port=14159 main_ddp.py \
    --dataset_name=osm \
    --dataset_root="$DATA" \
    --semantic_classes=${SEM_CLASSES} \
    --job_name="${JOB}" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size ${BATCH_SIZE} \
    --input_channels=3 \
    --image_size ${IMAGE_SIZE} \
    --poly2seq \
    --seq_len ${SEQ_LEN} \
    --row_tolerance ${ROW_TOL} \
    --num_bins ${NUM_BINS} \
    --ckpt_every_epoch=50 \
    --eval_every_epoch=50 \
    --label_smoothing 0.1 \
    --epochs ${EPOCHS} \
    --lr_drop '100' \
    --cls_loss_coef ${CLS_COEFF} \
    --coords_loss_coef ${COO_COEFF} \
    --room_cls_loss_coef ${SEM_COEFF} \
    --resume "$OUTPUT_DIR/$JOB/checkpoint.pth" \
    --ema4eval \
    --disable_poly_refine \
    --dec_attn_concat_src \
    --per_token_sem_loss \
    --jointly_train \
    --converter_version ${CONVERTER} \
    --use_anchor \
    --add_geom_head \
    --geom_mask \
    --sem_attn \
    --disable_wd_as_line \
    --start_from_checkpoint "/share/garg/jrg377/osm_stage1_checkpoints/osm_stage_queens4000_gray/checkpoint0199.pth"

    # --start_from_checkpoint "/share/garg/jrg377/osm_stage_queens4000_pretrain/checkpoint.pth"