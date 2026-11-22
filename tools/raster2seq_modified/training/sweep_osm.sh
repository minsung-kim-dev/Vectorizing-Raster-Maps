#!/usr/bin/env bash
# Short stage 1 runs (shapes only, same setup as train_osm.sh) over a few hyperparameter settings,
# each validated every EVAL_EVERY epochs; compare the runs' validation scores in wandb / $OUTPUT_DIR/<job>.
#
# Usage (in the raster2seq env):  bash tools/raster2seq_modified/training/sweep_osm.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
R2S=${R2S:-$REPO/../Raster2Seq}
DATA=${DATA:-$REPO/data/coco_osm_flushing}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/save_models}

NUM_GPUS=1
BATCH_SIZE=8
EPOCHS=30         # short runs: enough to rank the settings, not to finish training
EVAL_EVERY=10     # 3 validation runs per setting

# name  lr  cls_loss_coef  coords_loss_coef  label_smoothing
CONFIGS=(
    "base      2e-4  5   20  0.1"
    "lr1e-4    1e-4  5   20  0.1"
    "lr5e-4    5e-4  5   20  0.1"
    "coords40  2e-4  5   40  0.1"
    "smooth0   2e-4  5   20  0.0"
)

cd "$R2S"
mkdir -p wandb_keys && touch "wandb_keys/$(whoami)_wandb.key"
PRETRAIN=$(python -c "from raster2seq_hub import download_checkpoint; print(download_checkpoint('cubicasa5k'))")

for cfg in "${CONFIGS[@]}"; do
    read -r NAME LR CLS COO SMOOTH <<< "$cfg"
    JOB="osm_stage1_sweep_${NAME}"
    echo "==== $JOB: lr $LR, cls $CLS, coords $COO, smoothing $SMOOTH ===="
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
        --image_size 512 \
        --poly2seq \
        --seq_len 1024 \
        --row_tolerance 0.08 \
        --num_bins 32 \
        --ckpt_every_epoch=${EVAL_EVERY} \
        --eval_every_epoch=${EVAL_EVERY} \
        --label_smoothing ${SMOOTH} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --lr_drop "${EPOCHS}" \
        --cls_loss_coef ${CLS} \
        --coords_loss_coef ${COO} \
        --resume "$OUTPUT_DIR/$JOB/checkpoint.pth" \
        --ema4eval \
        --disable_poly_refine \
        --num_workers 6 \
        --dec_attn_concat_src \
        --converter_version v3 \
        --use_anchor \
        --start_from_checkpoint "$PRETRAIN"
done
