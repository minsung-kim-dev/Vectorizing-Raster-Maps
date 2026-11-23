#!/usr/bin/env bash
# Evaluate the last checkpoint of each sweep_osm.sh run (stage 1) on the validation set with eval.py, which
# reports polygon, line and combined (all_*) metrics with 95% bootstrap intervals, then print one table.
# Saves one file per run, $EVAL_DIR/<run>.json (eval.py's results.txt), plus val_results.csv: eval.py writes to a
# temporary folder that is deleted afterwards.
#
# Usage (in the raster2seq env, on a GPU node):  bash tools/raster2seq_modified/evaluation/eval_sweep_osm.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
R2S=${R2S:-$REPO/../Raster2Seq}
DATA=${DATA:-/share/garg/jrg377/coco_osm_fixed}
RUNS_DIR=${RUNS_DIR:-/share/garg/jrg377/osm_tuning}      # sweep_osm.sh's OUTPUT_DIR
EVAL_DIR=${EVAL_DIR:-$RUNS_DIR/eval_val}                 # results: $EVAL_DIR/<run>.json
SPLIT=val

# the sweep's runs (job names without the osm_stage1_tuning_ prefix)
RUNS=(base base1 base2 lr5e-4 lr5e-4_1 lr5e-4_2 coords40 coords40_1 coords40_2 rowtol2x rowtol2x_1 rowtol2x_2
      coords60_1 coords60_2 coords40Xrow2x_1 coords40Xrow2x_2 rowtol0.04 smooth0 lr1e-4 coords10 lr5e-5)

# the stage 1 model settings, as in sweep_osm.sh (row tolerance and loss weights only matter for training)
MODEL=(--dataset_name=osm --semantic_classes=3 --geom_classes --per_token_sem_loss --input_channels 3
       --image_size 256 --poly2seq --seq_len 512 --num_bins 32 --converter_version v3
       --disable_poly_refine --dec_attn_concat_src --use_anchor --ema4eval)

mkdir -p "$EVAL_DIR"
cd "$R2S"
for run in "${RUNS[@]}"; do
    ckpt="$RUNS_DIR/osm_stage1_tuning_$run/checkpoint.pth"
    if [[ ! -f "$ckpt" ]]; then echo "skip $run: no checkpoint"; continue; fi
    if [[ -f "$EVAL_DIR/$run.json" ]]; then echo "skip $run: already evaluated"; continue; fi
    echo "==== $run"
    tmp=$(mktemp -d)  # eval.py's output folder; only its results.txt is kept
    if python eval.py --checkpoint "$ckpt" --dataset_root "$DATA" --eval_set "$SPLIT" --output_dir "$tmp" "${MODEL[@]}" \
        --plot_pred '' --plot_density '' --plot_gt ''; then
        cp "$tmp/results.txt" "$EVAL_DIR/$run.json"
    else
        echo "FAILED: $run"
    fi
    rm -rf "$tmp"
done

# one table: each run's metrics, then the mean / std over seeds of each setting
python - "$EVAL_DIR" <<'EOF'
import glob, json, os, sys
import pandas as pd
rows = []
for f in glob.glob(os.path.join(sys.argv[1], "*.json")):
    r = json.load(open(f))
    rows.append({"run": os.path.basename(f)[:-5], **{k: v for k, v in r.items() if isinstance(v, float)}})
df = pd.DataFrame(rows).set_index("run")
cols = [c for c in ["room_f1", "corner_f1", "angles_f1", "line_f1", "line_corner_f1", "all_f1", "all_corner_f1"] if c in df]
print(df[cols].sort_values("room_f1", ascending=False).round(3).to_string())
df["config"] = df.index.str.replace(r"_\d+$", "", regex=True).str.replace(r"^base\d$", "base", regex=True)
print(df.groupby("config")[cols].agg(["mean", "std", "count"]).sort_values(("room_f1", "mean"), ascending=False).round(3).to_string())
df.to_csv(os.path.join(sys.argv[1], "val_results.csv"))
EOF
