"""Train vs. validation losses per epoch for one training run, from the run folder's log.txt (main_ddp.py writes
one JSON line per epoch: train_* every epoch, test_* = validation every --eval_every_epoch epochs).

One panel per loss (they have different scales): the weighted total, then each part unscaled. A second figure,
metrics.png, shows the validation precision, recall and F1 per shape type (training computes no such metrics).

Usage (from the repo root):
  python tools/raster2seq_modified/training/plot_losses.py /share/garg/jrg377/osm_tuning/osm_stage1_tuning_coords40
  (writes eval_outputs/<job>/losses.png and metrics.png, with the model's other evaluation results;
  --out to choose another folder)
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

TRAIN, VAL = "#2a78d6", "#eb6834"  # categorical slots 1-2 (blue, orange)
PREC, REC, F1 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (blue, orange, aqua)
LOSSES = {  # log key (after train_ / test_) -> panel title
    "loss": "Total loss (weighted)",
    "loss_ce_unscaled": "Token type (corner / separator / end)",
    "loss_coords_unscaled": "Coordinates",
    "loss_ce_room_unscaled": "Labels",
    "loss_ce_geom_unscaled": "Polygon vs. line",
}
METRICS = {  # log key (after test_, before _prec / _rec) -> panel title
    "room": "Polygons",
    "corner": "Polygon corners",
    "room_sem": "Polygons (+ label)",
    "line": "Lines",
    "line_corner": "Line corners",
    "line_sem": "Lines (+ label)",
}

ap = argparse.ArgumentParser()
ap.add_argument("run", help="run folder (output dir + job name), containing log.txt")
ap.add_argument("--out", help="folder for the figures (default: eval_outputs/<job>)")
a = ap.parse_args()

df = pd.DataFrame([json.loads(line) for line in open(os.path.join(a.run, "log.txt")) if line.strip()])
df = df.drop_duplicates("epoch", keep="last").sort_values("epoch")  # a resumed run can repeat epochs
df["epoch"] += 1  # log counts from 0
panels = [k for k in LOSSES if f"train_{k}" in df]

job = os.path.basename(os.path.normpath(a.run))
out = a.out or os.path.join("eval_outputs", job)
os.makedirs(out, exist_ok=True)


def style(ax, title):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("Epoch")
    ax.grid(color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.4), squeeze=False)
for ax, k in zip(axes[0], panels):
    ax.plot(df.epoch, df[f"train_{k}"], color=TRAIN, lw=2, label="Train")
    if f"test_{k}" in df:
        val = df.dropna(subset=[f"test_{k}"])
        ax.plot(val.epoch, val[f"test_{k}"], color=VAL, lw=2, marker="o", ms=5, label="Validation")
    style(ax, LOSSES[k])
axes[0][0].set_ylabel("Loss")
axes[0][0].legend(frameon=False)
fig.suptitle(job, x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(out, "losses.png"), dpi=150)

# validation precision / recall / F1, one panel per shape type (rows with validation only)
val = df.dropna(subset=["test_room_prec"])
panels = [k for k in METRICS if f"test_{k}_prec" in val]
fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4), squeeze=False, sharey=True)
for ax, k in zip(axes[0], panels):
    p, r = val[f"test_{k}_prec"], val[f"test_{k}_rec"]
    for y, color, name in ((p, PREC, "Precision"), (r, REC, "Recall"), (2 * p * r / (p + r), F1, "F1")):
        ax.plot(val.epoch, y, color=color, lw=2, marker="o", ms=5, label=name)
    style(ax, METRICS[k])
axes[0][0].set_ylabel("Validation score")
axes[0][0].legend(frameon=False)
fig.suptitle(job, x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(out, "metrics.png"), dpi=150)
print("wrote", os.path.join(out, "losses.png"), "and metrics.png")
