#!/usr/bin/env bash
# Stage 2 again from epoch 100 of an earlier stage 2 run, without the learning-rate drop (like Raster2Seq's
# tools/finetune_*.sh, which keep the learning rate constant and rely on the EMA weights).
#
# Copies <FROM_JOB>/checkpoint0099.pth to <JOB>/checkpoint.pth (once) and resumes from it: weights, optimizer,
# EMA and epoch are restored; with --lr_drop '' there is no scheduler, so the learning rate stays at --lr.
# The model / data settings below must match the run that wrote the checkpoint.
#
# Usage (in the raster2seq env):  bash tools/raster2seq_modified/training/finetune_osm_nodrop.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
R2S=${R2S:-$REPO/../Raster2Seq}
DATA=${DATA:-/share/garg/jrg377/coco_osm_fixed}
OUTPUT_DIR=${OUTPUT_DIR:-/share/garg/jrg377/osm_stage2_checkpoints}

FROM_JOB=osm_stage2_queens_4000pretrain_smallv2   # the earlier run
FROM_CKPT=checkpoint0099.pth                       # saved at the end of epoch 100, before its drop
JOB=${FROM_JOB}_nodrop

IMAGE_SIZE=256
SEQ_LEN=512
NUM_BINS=32
ROW_TOL=0.08
CLS_COEFF=5
COO_COEFF=20
SEM_COEFF=1
CONVERTER=v3
NUM_GPUS=1
BATCH_SIZE=12     # same as the earlier run (the optimizer state comes from it)
EPOCHS=400        # total: continues at epoch 100, so 300 more
# vocabulary size + 1 (Raster2Seq reserves the last class), read from create_coco_osm.py
SEM_CLASSES=$(cd "$REPO" && python -c "from data_preprocess.create_coco_osm import CATEGORIES; print(len(CATEGORIES) + 1)")

# start the new run from the old checkpoint, only the first time: after a preemption, checkpoint.pth holds
# this run's latest epoch and must not be replaced
mkdir -p "$OUTPUT_DIR/$JOB"
if [[ ! -f "$OUTPUT_DIR/$JOB/checkpoint.pth" ]]; then
    cp "$OUTPUT_DIR/$FROM_JOB/$FROM_CKPT" "$OUTPUT_DIR/$JOB/checkpoint.pth"
fi

cd "$R2S"
torchrun --nproc_per_node=${NUM_GPUS} --master_port=14159 main_ddp.py \
    --dataset_name=osm \
    --dataset_root="$DATA" \
    --semantic_classes=${SEM_CLASSES} \
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
    --eval_every_epoch=50 \
    --label_smoothing 0.1 \
    --epochs ${EPOCHS} \
    --lr_drop '' \
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
    --disable_wd_as_line
