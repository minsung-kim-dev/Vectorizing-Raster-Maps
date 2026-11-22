#!/usr/bin/env bash
# Raster2Seq training pipeline for the OSM crops, modeled on Raster2Seq's tools/train_cc5k.sh
# (stage 1, shapes only) and tools/finetune_cc5k.sh (stage 2, labels).
#
# Usage (in the raster2seq env):  bash tools/raster2seq_modified/training/train_osm.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
R2S=${R2S:-$REPO/../Raster2Seq}
DATA=${DATA:-/share/garg/jrg377/coco_osm_newvocab}
OUTPUT_DIR=${OUTPUT_DIR:-/share/garg/jrg377/osm_stage1_checkpoints}

IMAGE_SIZE=256    
SEQ_LEN=512      
NUM_BINS=32       # matches Raster2Seq's CubiCasa checkpoint used as the starting point
ROW_TOL=0.08      # shape order: rows 8% of the image height, top to bottom, left to right (needs the fork)
CLS_COEFF=5
COO_COEFF=20
CONVERTER=v3
NUM_GPUS=1
BATCH_SIZE=8      # lower if the GPU runs out of memory
EPOCHS=200
JOB="osm_stage_queens4000_pretrain_small"


# ---- Stage 1 training ----
cd "$R2S"

mkdir -p wandb_keys && touch "wandb_keys/$(whoami)_wandb.key"  # main_ddp.py otherwise prompts for a wandb key
PRETRAIN=$(python -c "from raster2seq_hub import download_checkpoint; print(download_checkpoint('cubicasa5k'))")
torchrun --nproc_per_node=${NUM_GPUS} --master_port=14159 main_ddp.py \
    --dataset_name=osm \
    --dataset_root="$DATA" \
    --semantic_classes=3 \
    --geom_classes \
    --per_token_sem_loss \
    --jointly_train \
    --disable_wd_as_line \
    --job_name="$JOB" \
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
    --resume "$OUTPUT_DIR/$JOB/checkpoint.pth" \
    --ema4eval \
    --disable_poly_refine \
    --num_workers 6 \
    --dec_attn_concat_src \
    --converter_version ${CONVERTER} \
    --use_anchor  \
    --start_from_checkpoint "$PRETRAIN"