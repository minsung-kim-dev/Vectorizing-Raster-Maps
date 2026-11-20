#!/usr/bin/env bash
# Raster2Seq training pipeline for the OSM crops, modeled on Raster2Seq's tools/train_cc5k.sh
# (stage 1, shapes only). OSM lines are converted to polygons using a 2 px buffer.
#
# Usage (in the raster2seq env):  bash tools/raster2seq_original/training/train_osm_original.sh
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
R2S=${R2S:-$REPO/../Raster2Seq}
CROPS=${CROPS:-$REPO/crops}
DATA=${DATA:-$REPO/data/coco_osm_flushing_buffer}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/save_models}

IMAGE_SIZE=512
SEQ_LEN=1024
NUM_BINS=32       # Coordinate bins used by the CubiCasa5K checkpoint.
ROW_TOL=0.08      # Group shapes into rows using 8% of the image height.
BATCH_SIZE=8
EPOCHS=200
JOB=osm_buffer_flushing_200_fresh_01

# ---- 1. COCO dataset (skipped if all three split files already exist) ----
# Each crop has a matching .png and .geojson.gz file.
# Buffer lines in the original image coordinates and hold out Flushing for testing.
if [[ ! -f "$DATA/annotations/train.json" ||
      ! -f "$DATA/annotations/val.json" ||
      ! -f "$DATA/annotations/test.json" ]]; then
    python "$REPO/data_preprocess/create_coco_osm.py" \
        --data_root "$CROPS" \
        --output "$DATA" \
        --test_borough flushing \
        --line_buffer_px 2 \
        --val_frac 0.1 \
        --seed 0
fi

# ---- 2. CubiCasa5K pretrained checkpoint ----
# Use the checkpoint revision recorded in the original training command.
if [[ -e "$OUTPUT_DIR/$JOB" ]]; then
    echo "Run directory already exists: $OUTPUT_DIR/$JOB" >&2
    echo "Set OUTPUT_DIR to a fresh directory for a new run." >&2
    exit 1
fi

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_NET=Socket NCCL_NET_PLUGIN=none PYTHONUNBUFFERED=1
cd "$R2S"

mkdir -p wandb_keys
touch "wandb_keys/$(whoami)_wandb.key"

PRETRAIN=$(python - <<'PY'
from huggingface_hub import hf_hub_download

print(hf_hub_download(
    repo_id="haopt/Raster2Seq",
    filename="cubicasa5k/checkpoint.pth",
    revision="58e66d2887a581e4c9750a905872901ea92e4318",
))
PY
)

# ---- 3. Stage 1: shapes only (GPU) ----
# The waffle loader reads the buffered OSM polygons; semantic_classes=-1 disables classification.
torchrun --nproc_per_node=1 --master_port=14159 main_ddp.py \
    --dataset_name=waffle \
    --dataset_root "$DATA" \
    --semantic_classes=-1 \
    --job_name "$JOB" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --input_channels=3 \
    --image_size "$IMAGE_SIZE" \
    --poly2seq \
    --seq_len "$SEQ_LEN" \
    --row_tolerance "$ROW_TOL" \
    --num_bins "$NUM_BINS" \
    --ckpt_every_epoch 10 \
    --eval_every_epoch 250 \
    --epochs "$EPOCHS" \
    --lr 0.0002 \
    --lr_backbone 0.00002 \
    --lr_drop 100 \
    --weight_decay 0.0001 \
    --cls_loss_coef 5 \
    --coords_loss_coef 20 \
    --label_smoothing 0.1 \
    --num_workers 6 \
    --seed 42 \
    --disable_poly_refine \
    --dec_attn_concat_src \
    --converter_version v3 \
    --use_anchor \
    --ema4eval \
    --start_from_checkpoint "$PRETRAIN"
