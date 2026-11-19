#!/usr/bin/env bash
# Predict + evaluate the latest checkpoint. Usage (raster2seq env): bash tools/raster2seq_modified/evaluation/predict_and_evaluate_stage2.sh [job] [data] [split]
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD
JOB=${1:-osm_stage2_queens_4000pretrain_small}
DATA=${2:-/share/garg/jrg377/coco_osm_newvocab}
SPLIT=${3:-test}
CKPT=${CKPT:-/share/garg/jrg377/osm_stage2_checkpoints/osm_stage2_queens_4000pretrain_small/checkpoint.pth}
SEMANTIC_CLASSES=$(python -c "from data_preprocess.create_coco_osm import CATEGORIES; print(len(CATEGORIES) + 1)")

# MODEL=(--dataset_name=osm --semantic_classes=34 --per_token_sem_loss --input_channels 3
#        --image_size 512 --poly2seq --seq_len 1024 --num_bins 32
#        --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval
#        --batch_size 24
#        --add_geom_head --geom_mask)

MODEL=(--dataset_name=osm --semantic_classes=$SEMANTIC_CLASSES --per_token_sem_loss --input_channels 3
       --image_size 256 --poly2seq --seq_len 512 --num_bins 32 --row_tolerance 0.08 --converter_version v3
       --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval --add_geom_head --geom_mask --sem_attn )


N=200
SUB=$(mktemp -d); mkdir -p "$SUB/$SPLIT" "$SUB/annotations"
python -c "
import json, os
d = json.load(open('$DATA/annotations/$SPLIT.json')); d['images'] = d['images'][:$N]
ids = {i['id'] for i in d['images']}; d['annotations'] = [a for a in d['annotations'] if a['image_id'] in ids]
json.dump(d, open('$SUB/annotations/$SPLIT.json', 'w'))
for i in d['images']: os.symlink(os.path.abspath('$DATA/$SPLIT/' + i['file_name']), '$SUB/$SPLIT/' + i['file_name'])"

#python ../Raster2Seq/predict.py --checkpoint "$CKPT" --dataset_root "$SUB/$SPLIT" --output_dir predictions "${MODEL[@]}"
python ../Raster2Seq/eval.py --checkpoint "$CKPT" --dataset_root "$SUB" --eval_set "$SPLIT" --row_tolerance 0.08 --converter_version v3 --output_dir "eval_outputs/$JOB/test_200" "${MODEL[@]}" --filter_gt_by_area --min_poly_area 0 --per_label_eval