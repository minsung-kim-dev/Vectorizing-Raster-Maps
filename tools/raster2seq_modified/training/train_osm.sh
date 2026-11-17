#!/usr/bin/env bash
# Raster2Seq training pipeline for the OSM crops, modelled on Raster2Seq's tools/train_cc5k.sh
# (stage 1, shapes only) and tools/finetune_cc5k.sh (stage 2, labels).
#
# Usage (in the raster2seq env):  bash tools/train_osm.sh
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
CONVERTER=v3
NUM_GPUS=1
BATCH_SIZE=8      # lower if the GPU runs out of memory
# EPOCHS=1000
EPOCHS=200
JOB="osm_stage_queens4000_pretrain"

# vocabulary size + 1 (Raster2Seq reserves the last class), read from create_coco_osm.py
SEM_CLASSES=$(cd "$REPO" && python -c "from data_preprocess.create_coco_osm import CATEGORIES; print(len(CATEGORIES) + 1)")

# ---- 1. COCO dataset (skipped if it already exists) ----
if [[ ! -d "$DATA/annotations" ]]; then
    (cd "$REPO" && python data_preprocess/create_coco_osm.py --data_root "$CROPS" --output "$DATA")
fi

# ---- 2. How many crops fit in SEQ_LEN tokens (corners + one separator/end per shape) ----
python - "$DATA/annotations/val.json" "$SEQ_LEN" <<'EOF'
import collections, json, sys
import numpy as np
anns, seq_len = json.load(open(sys.argv[1]))["annotations"], int(sys.argv[2])
need = collections.Counter()
for a in anns:
    need[a["image_id"]] += len(a["segmentation"][0]) // 2 + 1
n = np.array(list(need.values()))
print(f"val tokens per image: median {np.median(n):.0f}, max {n.max()} | "
      f"{np.mean(n <= seq_len):.0%} fit in SEQ_LEN={seq_len}")
EOF

# ---- Stage 1: shapes only (GPU) ----
cd "$R2S"

mkdir -p wandb_keys && touch "wandb_keys/$(whoami)_wandb.key"  # main_ddp.py otherwise prompts for a wandb key
PRETRAIN=$(python -c "from raster2seq_hub import download_checkpoint; print(download_checkpoint('cubicasa5k'))")
PYTHONPATH=~/Vectorizing-Raster-Maps torchrun --nproc_per_node=${NUM_GPUS} --master_port=14159 main_ddp.py \
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