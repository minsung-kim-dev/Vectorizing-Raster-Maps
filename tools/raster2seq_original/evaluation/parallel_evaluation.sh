#!/usr/bin/env bash
# Evaluate a checkpoint on a split in K parallel parts, then combine the results.
# Usage (raster2seq env): bash tools/raster2seq_original/evaluation/parallel_evaluation.sh [job] [data] [split] [K]
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD

JOB=${1:-r2s_original_stage1}
DATA=${2:-data/coco_osm_buffer}
SPLIT=${3:-test}
K=${4:-4}
CKPT=${CKPT:-${OUTPUT_DIR:-save_models}/$JOB/checkpoint0199.pth}
OUT=${OUT:-eval_outputs/$JOB/$SPLIT}

MODEL=(--dataset_name=osm --per_token_sem_loss --input_channels 3
       --converter_version v3
       --image_size 256 --poly2seq --seq_len 512 --num_bins 32
       --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval --min_poly_area 25 --filter_gt_by_area)

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

# every metric is an average over images: weight each part by its image count, then recompute F1
python - "$OUT" "$SHARDS" "$SPLIT" "$K" <<'EOF'
import json, sys
out, shards, split, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
parts = [(json.load(open(f"{out}/part{i}/results.txt")),
          len(json.load(open(f"{shards}/{i}/annotations/{split}.json"))["images"])) for i in range(k)]
n = sum(c for _, c in parts)
res = {key: sum(r[key] * c for r, c in parts) / n
       for key, v in parts[0][0].items() if isinstance(v, (int, float)) and not key.endswith("_f1")}
for m in ["room", "corner", "angles", "room_sem", "line", "line_sem"]:
    p, r = res.get(f"{m}_prec"), res.get(f"{m}_rec")
    if p is not None and r is not None:
        res[f"{m}_f1"] = 2 * p * r / (p + r) if p + r else 0.0
res["images"] = n
json.dump(res, open(f"{out}/results.txt", "w"), indent=1)
print(json.dumps(res, indent=1))
EOF
