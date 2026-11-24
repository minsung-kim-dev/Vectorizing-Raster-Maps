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
rows = {"Corners": "all_corner", "Angles": "angles",
"All shapes": "all", "Polygons": "room", "Lines": "line"} #"Polygons (+ label)": "room_sem", "Lines (+ label)": "line_sem",

cols = {"Precision": "prec", "Recall": "rec", "F1": "f1"}

def value(m, stat):
    """<m>_<stat>; F1 from precision and recall if it wasn't saved."""
    if f"{m}_{stat}" in res:
        return res[f"{m}_{stat}"]
    p, r = res[f"{m}_prec"], res[f"{m}_rec"]
    return 2 * p * r / (p + r) if p + r else 0.0


def interval(m, stat):
    ci = res.get("ci")
    lo_hi = ci.get(f"{m}_{stat}")
    return f"[{lo_hi[0]:.2f}, {lo_hi[1]:.2f}]" if lo_hi else ""

# one row per statistic, each followed by a row with its 95% intervals
lines, index = [], []
for c, s in cols.items():
    lines.append({k: f"{value(m, s):.2f}" for k, m in rows.items()}); index.append(c)
    lines.append({k: interval(m, s) for k, m in rows.items()}); index.append("")
overall = pd.DataFrame(lines, index=index)
print(overall.to_latex(caption=f"Overall results ({name}); 95\\% bootstrap intervals below each value",
                       label=f"tab:overall_{job}"))
                       
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