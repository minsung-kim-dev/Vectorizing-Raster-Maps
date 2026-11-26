#!/usr/bin/env bash
# Evaluate the last checkpoint of each stage 2 tuning run (sweep_osm_stage2.sh) on the validation set with eval.py,
# which reports polygon, line and combined (all_*) metrics with 95% bootstrap intervals, then print one table.
# Saves one file per run, $EVAL_DIR/<run>.json (eval.py's results.txt), plus val_results.csv: eval.py writes to a
# temporary folder that is deleted afterwards.
#
# Usage (in the raster2seq env, on a GPU node):  bash tools/raster2seq_modified/evaluation/eval_sweep_osm.sh
ulimit -n "$(ulimit -Hn)"
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
R2S=${R2S:-$REPO/../Raster2Seq}
DATA=${DATA:-/share/garg/jrg377/coco_osm_fixed}
RUNS_DIR=${RUNS_DIR:-/share/garg/jrg377/osm_tuning}      # sweep's OUTPUT_DIR
EVAL_DIR=${EVAL_DIR:-$RUNS_DIR/eval_val}                 # results: $EVAL_DIR/<run>.json
SPLIT=val
P=${P:-4}                                                # evaluations at a time
SEMANTIC_CLASSES=$(cd "$REPO" && python -c "from data_preprocess.create_coco_osm import CATEGORIES; print(len(CATEGORIES) + 1)")

# the sweep's runs (full job names)
RUNS=(osm_stage2_tuning_sem1_attn_1 osm_stage2_tuning_sem1_attn_2 osm_stage2_tuning_sem1_noattn_1 osm_stage2_tuning_sem1_noattn_2
      osm_stage2_tuning_sem2_attn_1 osm_stage2_tuning_sem2_attn_2 osm_stage2_tuning_sem4_attn_1 osm_stage2_tuning_sem4_attn_2
      osm_stage2_tuning_coordcoeff20_1  osm_stage2_tuning_coordcoeff20_2)

# the stage 2 model settings (loss weights only matter for training); --sem_attn is added per run below
MODEL=(--dataset_name=osm --semantic_classes=$SEMANTIC_CLASSES --per_token_sem_loss --input_channels 3
       --image_size 256 --poly2seq --seq_len 512 --num_bins 32 --row_tolerance 0.08 --converter_version v3
       --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval --add_geom_head --geom_mask)

evaluate() {  # one run: eval.py writes to a temporary folder; keep results.txt (and the log if it fails)
    local run=$1 tmp
    local attn=(--sem_attn); [[ $run == *noattn* ]] && attn=()  # the model must match how the run was trained
    tmp=$(mktemp -d)
    if python eval.py --checkpoint "$RUNS_DIR/$run/checkpoint.pth" --dataset_root "$DATA" \
        --eval_set "$SPLIT" --output_dir "$tmp" "${MODEL[@]}" "${attn[@]}" --plot_pred '' --plot_density '' --plot_gt '' \
        --batch_size 32 --num_workers 0 --filter_gt_by_area --min_poly_area 25 > "$tmp/log.txt" 2>&1; then
        cp "$tmp/results.txt" "$EVAL_DIR/$run.json"; echo "done: $run"
    else
        cp "$tmp/log.txt" "$EVAL_DIR/$run.failed.log"; echo "FAILED: $run (see $EVAL_DIR/$run.failed.log)"
    fi
    rm -rf "$tmp"
}

mkdir -p "$EVAL_DIR"
cd "$R2S"
for run in "${RUNS[@]}"; do
    if [[ ! -f "$RUNS_DIR/$run/checkpoint.pth" ]]; then echo "skip $run: no checkpoint"; continue; fi
    if [[ -f "$EVAL_DIR/$run.json" ]]; then echo "skip $run: already evaluated"; continue; fi
    echo "start: $run"
    evaluate "$run" &
    while (( $(jobs -rp | wc -l) >= P )); do sleep 5; done  # at most P at a time
done
wait

# one table: each run's metrics, then the mean / std over seeds of each setting
python - "$EVAL_DIR" <<'EOF'
import glob, json, os, sys
import pandas as pd
rows = []
for f in glob.glob(os.path.join(sys.argv[1], "*.json")):
    r = json.load(open(f))
    rows.append({"run": os.path.basename(f)[:-5], **{k: v for k, v in r.items() if isinstance(v, float)}})
df = pd.DataFrame(rows).set_index("run")
cols = [c for c in ["room_f1", "room_sem_f1", "corner_f1", "angles_f1", "line_f1", "line_sem_f1", "line_corner_f1",
                    "all_f1", "all_corner_f1"] if c in df]
print(df[cols].sort_values("room_f1", ascending=False).round(3).to_string())
df["config"] = df.index.str.replace(r"_\d+$", "", regex=True).str.replace(r"^base\d$", "base", regex=True)
print(df.groupby("config")[cols].agg(["mean", "std", "count"]).sort_values(("room_f1", "mean"), ascending=False).round(3).to_string())
df.to_csv(os.path.join(sys.argv[1], "val_results_stage2.csv"))
EOF