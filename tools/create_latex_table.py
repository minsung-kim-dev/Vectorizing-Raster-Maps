"""LaTeX tables (overall + per label) from eval_outputs/<job>/results.txt.
Usage: python tools/results_table.py --job osm_stage2_queens4000_pretrain"""
import argparse
import json

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--job", required=True, help="folder name under eval_outputs/")
job = ap.parse_args().job
res = json.load(open(f"eval_outputs/{job}/results.txt"))
name = job.replace("_", r"\_")  # captions need escaped underscores

# get column names, map to labels
rows = {"Corners": "corner", "Angles": "angles",
"Polygons": "room", "Lines": "line",  "Polygons (+ label)": "room_sem", "Lines (+ label)": "line_sem",}

overall = pd.DataFrame({k: {"Precision": res[f"{m}_prec"], "Recall": res[f"{m}_rec"]} for k, m in rows.items()}).T
overall["F1"] = 2 * overall.Precision * overall.Recall / (overall.Precision + overall.Recall)
print(overall.T.to_latex(float_format="%.2f", caption=f"Overall results ({name})", label=f"tab:overall_{job}"))

# per label
pl = pd.Series(res["per_label"])
# gather all precision and recall values
per_label = pd.DataFrame({"Precision": pl.filter(regex="^prec_").rename(lambda k: k[5:]),
                          "Recall": pl.filter(regex="^rec_").rename(lambda k: k[4:])})

# calculate F1, fill in missing values as 0
per_label["F1"] = (2 * per_label.Precision * per_label.Recall / (per_label.Precision + per_label.Recall)).fillna(0)
per_label.index = per_label.index.str.replace("_", " ")
print(per_label.sort_values('F1', ascending=False).iloc[:16].T.to_latex(float_format="%.2f", caption=f"Results per label ({name})", label=f"tab:per_label_{job}"))

print(per_label.sort_values('F1', ascending=False).iloc[16:].T.to_latex(float_format="%.2f", caption=f"Results per label ({name})", label=f"tab:per_label_{job}"))