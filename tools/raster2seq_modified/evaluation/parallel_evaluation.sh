#!/usr/bin/env bash
# Evaluate a checkpoint on a split in K parallel parts, then combine the results.
# Usage (raster2seq env): bash tools/raster2seq_modified/evaluation/parallel_evaluation.sh [job] [data] [split] [K]
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD
JOB=${1:-osm_stage2_queens_4000pretrain} DATA=${2:-data/coco_osm_flushing} SPLIT=${3:-val} K=${4:-4}
CKPT=save_models/$JOB/checkpoint.pth
OUT=eval_outputs/$JOB/$SPLIT
MODEL=(--dataset_name=osm --semantic_classes=34 --per_token_sem_loss --input_channels 3
       --image_size 512 --poly2seq --seq_len 1024 --num_bins 32 --row_tolerance 0.08 --converter_version v3
       --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval --add_geom_head --geom_mask)

# MODEL=(--dataset_name=osm --semantic_classes=3 --geom_classes --per_token_sem_loss --input_channels 3
#        --converter_version v3
#        --image_size 512 --poly2seq --seq_len 1024 --num_bins 32
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
        --output_dir "$OUT/part$k" "${MODEL[@]}" --plot_pred '' --plot_density '' --plot_gt '' \
        > "$OUT/part$k/log.txt" 2>&1 &
done
wait

python tools/raster2seq_modified/evaluation/combine_resutls.py "$OUT"