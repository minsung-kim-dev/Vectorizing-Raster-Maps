#!/usr/bin/env bash
# Short stage 1 runs (shapes only, same setup as train_osm.sh) over a few hyperparameter settings,
# each validated every EVAL_EVERY epochs; compare the runs' validation scores in wandb / $OUTPUT_DIR/<job>.
#
# Usage (in the raster2seq env):  bash tools/raster2seq_modified/training/sweep_osm.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
R2S=${R2S:-$REPO/../Raster2Seq}
DATA=${DATA:-/share/garg/jrg377/coco_osm_fixed}
OUTPUT_DIR=${OUTPUT_DIR:-/share/garg/jrg377/osm_tuning}

NUM_GPUS=1
BATCH_SIZE=8
EPOCHS=30         # short runs: enough to rank the settings, not to finish training
EVAL_EVERY=10     # 3 validation runs per setting
SEQ_LEN=512
IMG_SIZE=256

# name  lr  cls_loss_coef  coords_loss_coef label_smoothing  row_tolerance  pretrain
# CONFIGS=(
#     "base           2e-4  5   20  0.1    0.08   cc5k"
#     "lr5e-5         5e-5  5   20  0.1    0.08   cc5k"
#     "lr1e-4         1e-4  5   20  0.1    0.08   cc5k"
#     "lr5e-4         5e-4  5   20  0.1    0.08   cc5k"
#     "coords10       2e-4  5   10  0.1    0.08   cc5k"
#     "coords40       2e-4  5   40  0.1    0.08   cc5k"
#     "smooth0        2e-4  5   20  0.0    0.08   cc5k"
#     "rowtol2x       2e-4  5   20  0.1    0.16   cc5k"
#     "rowtol0.04     2e-4  5   20  0.1    0.04   cc5k"
#     "scratch        2e-4  5   20  0.1    0.08   none"
# )

CONFIGS=(
    "base1              2e-4  5   20  0.1    0.08   cc5k     1"
    "base2              2e-4  5   20  0.1    0.08   cc5k     2"
    "lr5e-4_1           5e-4  5   20  0.1    0.08   cc5k     1"
    "lr5e-4_2           5e-4  5   20  0.1    0.08   cc5k     2"
    "coords40_1         2e-4  5   40  0.1    0.08   cc5k     1"
    "coords40_2         2e-4  5   40  0.1    0.08   cc5k     2"
    "rowtol2x_1         2e-4  5   20  0.1    0.16   cc5k     1"
    "rowtol2x_2         2e-4  5   20  0.1    0.16   cc5k     2"
    "coords60_1         2e-4  5   60  0.1    0.08   cc5k     1"
    "coords60_2         2e-4  5   60  0.1    0.08   cc5k     2"
    "coords40Xrow2x_1   2e-4  5   40  0.1    0.16   cc5k     1"
    "coords40Xrow2x_2   2e-4  5   40  0.1    0.16   cc5k     2"
)

cd "$R2S"
mkdir -p wandb_keys && touch "wandb_keys/$(whoami)_wandb.key"
PRETRAIN=$(python -c "from raster2seq_hub import download_checkpoint; print(download_checkpoint('cubicasa5k'))")

for cfg in "${CONFIGS[@]}"; do
    read -r NAME LR CLS COO SMOOTH TOL PRE SEED <<< "$cfg"
    START=(); [[ $PRE == cc5k ]] && START=(--start_from_checkpoint "$PRETRAIN")
    JOB="osm_stage1_tuning_${NAME}"
    echo "==== $JOB: lr $LR, cls $CLS, coords $COO, smoothing $SMOOTH, tolerance $TOL, pretrain $PRE ===="
    torchrun --nproc_per_node=${NUM_GPUS} --master_port=14159 main_ddp.py \
        --dataset_name=osm \
        --dataset_root="$DATA" \
        --semantic_classes=3 \
        --geom_classes \
        --per_token_sem_loss \
        --jointly_train \
        --disable_wd_as_line \
        --job_name="$JOB" \
        --seed=$SEED \
        --output_dir "$OUTPUT_DIR" \
        --batch_size ${BATCH_SIZE} \
        --input_channels=3 \
        --image_size ${IMG_SIZE} \
        --poly2seq \
        --seq_len ${SEQ_LEN} \
        --row_tolerance ${TOL} \
        --num_bins 32 \
        --ckpt_every_epoch=30 \
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
        "${START[@]}"
done
