#!/usr/bin/env bash
# Original Raster2Seq Stage 2: fine-tune polygon outlines and OSM labels together.
# Usage: bash tools/raster2seq_original/training/finetune_osm_original.sh
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
R2S=${R2S:-$REPO/../Raster2Seq}
DATA=$(cd "${DATA:-$REPO/data/coco_osm_buffer}" && pwd)
OUTPUT_DIR=$(cd "${OUTPUT_DIR:-$REPO/save_models}" && pwd)

IMAGE_SIZE=256
SEQ_LEN=512
NUM_BINS=32
ROW_TOL=0.08
CLS_COEFF=5
COO_COEFF=20
SEM_COEFF=1
CONVERTER=v3
NUM_GPUS=1
BATCH_SIZE=8
EPOCHS=200
JOB=r2s_original_stage2

STAGE1_CKPT="$OUTPUT_DIR/r2s_original_stage1/checkpoint0199.pth"
RESUME="$OUTPUT_DIR/$JOB/checkpoint.pth"

if [[ ! -f "$RESUME" && ! -f "$STAGE1_CKPT" ]]; then
    echo "Stage 2 needs the completed Stage 1 checkpoint: $STAGE1_CKPT" >&2
    exit 1
fi

# Raster2Seq uses one reserved class in addition to the COCO categories.
SEM_CLASSES=$(python - "$DATA/annotations/train.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as annotations:
    print(len(json.load(annotations)["categories"]) + 1)
PY
)

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_NET=Socket NCCL_NET_PLUGIN=none PYTHONUNBUFFERED=1
cd "$R2S"

torchrun --nproc_per_node="$NUM_GPUS" --master_port=14159 main_ddp.py \
    --dataset_name=osm \
    --dataset_root "$DATA" \
    --semantic_classes="$SEM_CLASSES" \
    --job_name "$JOB" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --input_channels=3 \
    --image_size "$IMAGE_SIZE" \
    --poly2seq \
    --seq_len "$SEQ_LEN" \
    --row_tolerance "$ROW_TOL" \
    --num_bins "$NUM_BINS" \
    --ckpt_every_epoch=50 \
    --eval_every_epoch=250 \
    --label_smoothing 0.1 \
    --epochs "$EPOCHS" \
    --lr_drop 100 \
    --cls_loss_coef "$CLS_COEFF" \
    --coords_loss_coef "$COO_COEFF" \
    --room_cls_loss_coef "$SEM_COEFF" \
    --resume "$RESUME" \
    --ema4eval \
    --disable_poly_refine \
    --dec_attn_concat_src \
    --per_token_sem_loss \
    --jointly_train \
    --converter_version "$CONVERTER" \
    --use_anchor \
    --disable_wd_as_line \
    --start_from_checkpoint "$STAGE1_CKPT"
