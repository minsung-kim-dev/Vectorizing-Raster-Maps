#!/usr/bin/env bash
# Short stage 2 runs (shapes + labels, same setup as finetune_osm.sh) from one stage 1 checkpoint, to check the
# effect of the label loss weight and of the semantic attention layer (--sem_attn), each validated every
# EVAL_EVERY epochs; compare the runs' validation scores (room_sem / line_sem vs room / line) in wandb / $OUTPUT_DIR/<job>.
#
# Usage (in the raster2seq env):  bash tools/raster2seq_modified/training/sweep_osm_stage2.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
R2S=${R2S:-$REPO/../Raster2Seq}
DATA=${DATA:-/share/garg/jrg377/coco_osm_fixed}
OUTPUT_DIR=${OUTPUT_DIR:-/share/garg/jrg377/osm_tuning}
STAGE1=${STAGE1:-/share/garg/jrg377/osm_stage1_checkpoints/osm_stage_queens4000_pretrain_tuned/checkpoint.pth}  # every run starts here

NUM_GPUS=1
BATCH_SIZE=8
EPOCHS=30         # short runs: enough to rank the settings, not to finish training
EVAL_EVERY=30     # 3 validation runs per setting
SEQ_LEN=512
IMG_SIZE=256
LR=2e-4
CLS_COEFF=5
# vocabulary size + 1 (Raster2Seq reserves the last class), read from create_coco_osm.py
SEM_CLASSES=$(cd "$REPO" && python -c "from data_preprocess.create_coco_osm import CATEGORIES; print(len(CATEGORIES) + 1)")

# name  room_cls_loss_coef  sem_attn  coords_loss_coef  seed
CONFIGS=(
    "sem1_attn_1      1   yes   40  1"
    "sem1_noattn_1    1   no    40  1"
    "sem1_attn_2      1   yes   40  2"
    "sem1_noattn_2    1   no    40  2"
    "sem2_attn_1      2   yes   40  1"
    "sem2_attn_2      2   yes   40  2"
    "sem4_attn_1      4   yes   40  1"
    "sem4_attn_2      4   yes   40  2"
    "coordcoeff20_1   1   yes   20  1"
    "coordcoeff20_2   1   yes   20  2"
)

cd "$R2S"
mkdir -p wandb_keys && touch "wandb_keys/$(whoami)_wandb.key"

for cfg in "${CONFIGS[@]}"; do
    read -r NAME SEM ATTN COO_COEFF SEED <<< "$cfg"
    SEM_ATTN=(); [[ $ATTN == yes ]] && SEM_ATTN=(--sem_attn)
    JOB="osm_stage2_tuning_${NAME}"
    echo "==== $JOB: label weight $SEM, sem_attn $ATTN, seed $SEED coo coeff $COO_COEFF ===="
    torchrun --nproc_per_node=${NUM_GPUS} --master_port=14159 main_ddp.py \
        --dataset_name=osm \
        --dataset_root="$DATA" \
        --semantic_classes=${SEM_CLASSES} \
        --per_token_sem_loss \
        --jointly_train \
        --disable_wd_as_line \
        --add_geom_head \
        --geom_mask \
        --job_name="$JOB" \
        --seed=$SEED \
        --output_dir "$OUTPUT_DIR" \
        --batch_size ${BATCH_SIZE} \
        --input_channels=3 \
        --image_size ${IMG_SIZE} \
        --poly2seq \
        --seq_len ${SEQ_LEN} \
        --row_tolerance 0.08 \
        --num_bins 32 \
        --ckpt_every_epoch=30 \
        --eval_every_epoch=${EVAL_EVERY} \
        --label_smoothing 0.1 \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --lr_drop "" \
        --cls_loss_coef ${CLS_COEFF} \
        --coords_loss_coef ${COO_COEFF} \
        --room_cls_loss_coef ${SEM} \
        --resume "$OUTPUT_DIR/$JOB/checkpoint.pth" \
        --ema4eval \
        --disable_poly_refine \
        --num_workers 6 \
        --dec_attn_concat_src \
        --converter_version v3 \
        --use_anchor \
        "${SEM_ATTN[@]}" \
        --start_from_checkpoint "$STAGE1"
done
