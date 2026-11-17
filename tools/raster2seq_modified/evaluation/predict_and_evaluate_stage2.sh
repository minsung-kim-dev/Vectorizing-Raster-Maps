#!/usr/bin/env bash
# Predict + evaluate the latest checkpoint. Usage (raster2seq env): bash tools/predict_eval_osm.sh [job] [data] [split]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
JOB=${1:-osm_stage2_queens_4000pretrain}
DATA=${2:-data/coco_osm_flushing} SPLIT=${3:-val}
CKPT=save_models/$JOB/checkpoint.pth
MODEL=(--dataset_name=osm --semantic_classes=34 --per_token_sem_loss --input_channels 3
       --image_size 512 --poly2seq --seq_len 1024 --num_bins 32
       --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval
       --batch_size 24
       --add_geom_head --geom_mask)

N=200
SUB=$(mktemp -d); mkdir -p "$SUB/$SPLIT" "$SUB/annotations"
python -c "
import json, os
d = json.load(open('$DATA/annotations/$SPLIT.json')); d['images'] = d['images'][:$N]
ids = {i['id'] for i in d['images']}; d['annotations'] = [a for a in d['annotations'] if a['image_id'] in ids]
json.dump(d, open('$SUB/annotations/$SPLIT.json', 'w'))
for i in d['images']: os.symlink(os.path.abspath('$DATA/$SPLIT/' + i['file_name']), '$SUB/$SPLIT/' + i['file_name'])"

#python ../Raster2Seq/predict.py --checkpoint "$CKPT" --dataset_root "$SUB/$SPLIT" --output_dir predictions "${MODEL[@]}"
python ../Raster2Seq/eval.py --checkpoint "$CKPT" --dataset_root "$SUB" --eval_set "$SPLIT" --row_tolerance 0.08 --converter_version v3 --output_dir "eval_outputs/$JOB" "${MODEL[@]}" --plot_pred '' --plot_density '' --plot_gt '' --per_label_eval