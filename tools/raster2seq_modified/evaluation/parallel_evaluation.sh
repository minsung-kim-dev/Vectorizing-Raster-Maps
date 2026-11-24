#!/usr/bin/env bash
# Evaluate a checkpoint on a split in K parallel parts, then combine the results.
# Usage (raster2seq env): bash tools/raster2seq_modified/evaluation/parallel_evaluation.sh [job] [data] [split] [K]
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD
# JOB=${1:-osm_stage2_queens_4000pretrain_newvocab}
# JOB=${1:-osm_stage_queens4000_pretrain_small}
# JOB=${1:-osm_stage_queens4000_pretrain}
JOB=${1:-osm_stage2_queens_4000pretrain_nodrop}
DATA=${2:-/share/garg/jrg377/coco_osm_fixed}
SPLIT=${3:-test}
K=${4:-4}
# CKPT=${CKPT:-/share/garg/jrg377/osm_stage2_queens4000_newvocab_checkpoints/osm_stage2_queens_4000pretrain_newvocab/checkpoint.pth}
# CKPT=${CKPT:-/share/garg/jrg377/osm_stage1_checkpoints/osm_stage_queens4000_pretrain_small/checkpoint.pth}
# CKPT=${CKPT:-share/garg/jrg377/osm_stage2_checkpoints/osm_stage2_queens_4000pretrain_smallv2/checkpoint0299.pth}
CKPT=${CKPT:-/share/garg/jrg377/osm_stage2_checkpoints/osm_stage2_queens_4000pretrain_nodrop/checkpoint.pth}
# CKPT=${CKPT:-/share/garg/jrg377/osm_stage_queens4000_pretrain/checkpoint.pth}

OUT=eval_outputs/$JOB/$SPLIT
SEMANTIC_CLASSES=$(python -c "from data_preprocess.create_coco_osm import CATEGORIES; print(len(CATEGORIES) + 1)")

# MODEL=(--dataset_name=osm --semantic_classes=$SEMANTIC_CLASSES --per_token_sem_loss --input_channels 3
#        --image_size 512 --poly2seq --seq_len 1024 --num_bins 32 --row_tolerance 0.08 --converter_version v3
#        --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval --add_geom_head --geom_mask --sem_attn )

MODEL=(--dataset_name=osm --semantic_classes=$SEMANTIC_CLASSES --per_token_sem_loss --input_channels 3
       --image_size 256 --poly2seq --seq_len 512 --num_bins 32 --row_tolerance 0.08 --converter_version v3
       --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval --add_geom_head --geom_mask --sem_attn )

# MODEL=(--dataset_name=osm --semantic_classes=3 --geom_classes --per_token_sem_loss --input_channels 3
#        --converter_version v3
#        --image_size 512 --poly2seq --seq_len 1024 --num_bins 32
#        --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval)

# MODEL=(--dataset_name=osm --semantic_classes=3 --geom_classes --per_token_sem_loss --input_channels 3
#        --converter_version v3
#        --image_size 256 --poly2seq --seq_len 512 --num_bins 32
#        --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval)


SHARDS=$(mktemp -d); trap 'rm -rf "$SHARDS"' EXIT
for k in $(seq 0 $((K - 1))); do
    mkdir -p "$SHARDS/$k/$SPLIT" "$SHARDS/$k/annotations" "$OUT/part$k"
    python -c "
import json, os
d = json.load(open('$DATA/annotations/$SPLIT.json')); d['images'] = d['images'][$k::$K]
ids = {i['id'] for i in d['images']}; d['annotations'] = [a for a in d['annotations'] if a['image_id'] in ids]
json.dump(d, open('$SHARDS/$k/annotations/$SPLIT.json', 'w'))
for i in d['images']: os.symlink(os.path.abspath('$DATA/$SPLIT/' + i['file_name']), '$SHARDS/$k/$SPLIT/' + i['file_name'])"
    python ../Raster2Seq/eval.py --checkpoint "$CKPT" --dataset_root "$SHARDS/$k" --eval_set "$SPLIT" \
        --output_dir "$OUT/part$k" "${MODEL[@]}" --plot_pred '' --plot_density '' --plot_gt ''  --filter_gt_by_area  --min_poly_area 25 --per_label_eval \
        > "$OUT/part$k/log.txt" 2>&1 &
done
wait

python tools/raster2seq_modified/evaluation/run_parallel_results.py "$OUT"