"""Collect every run's log.txt from sweep_osm.sh into one table.

Usage:  python tools/raster2seq_modified/training/collect_tuning_logs.py [/share/garg/jrg377/osm_tuning]
Writes <dir>/tuning_logs.csv (every epoch of every run) and prints each run's last validation scores.
"""
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/share/garg/jrg377/osm_tuning")
rows = [{"run": f.parent.name.removeprefix("osm_stage1_tuning_"), **json.loads(line)}
        for f in sorted(root.glob("*/log.txt")) for line in f.read_text().splitlines() if line.strip()]
df = pd.DataFrame(rows)
for m in ["room", "corner", "angles", "line", "line_corner"]:
    df[f"test_{m}_f1"] = 2 * df[f"test_{m}_prec"] * df[f"test_{m}_rec"] / (df[f"test_{m}_prec"] + df[f"test_{m}_rec"])
df.to_csv(root / "tuning_logs.csv", index=False)

# last validation epoch of each run, best room F1 first
val = df.dropna(subset=["test_room_prec"]).groupby("run").last()
cols = ["epoch", "test_room_f1", "test_corner_f1", "test_angles_f1", "test_line_f1", "test_line_corner_f1", "test_loss"]
print(val[cols].sort_values("test_room_f1", ascending=False).round(3).to_string())

# group seeds of the same setting: base1/base2 -> base, lr5e-4_1 -> lr5e-4, coords40Xrow2x_2 -> coords40Xrow2x
val = val.reset_index()
val["config"] = (val.run.str.replace(r"_\d+$", "", regex=True)
                        .str.replace(r"^base\d$", "base", regex=True))
f1 = ["test_room_f1", "test_corner_f1", "test_angles_f1", "test_line_f1", "test_line_corner_f1"]
summary = val.groupby("config")[f1].agg(["mean", "std", "count"])
print(summary.sort_values(("test_room_f1", "mean"), ascending=False).round(3).to_string())