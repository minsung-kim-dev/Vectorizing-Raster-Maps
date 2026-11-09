"""Print scored runs side by side, from the files evaluation/score.py already wrote (nothing is rerun).

    python evaluation/compare.py results/color_threshold_125m_v3_full/test results/color_threshold_125m_v3_nobuffer/test
    python evaluation/compare.py <run folders...> --names buffered "no buffer" --labels 12

Each folder needs summary.json, per_label.csv and per_bin.csv (written by score.py / any baseline).
Prints: overall scores with 95% intervals, F1 per label (most common labels first), recall per geometry bin.
"""

import argparse
import csv
import json
from pathlib import Path


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load(folder):
    folder = Path(folder)
    with open(folder / "summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    labels = {r["label"]: r for r in read_csv(folder / "per_label.csv")}
    bins = {(r["factor"], r["bin"]): r for r in read_csv(folder / "per_bin.csv")}
    return summary, labels, bins


def table(title, header, rows, width):
    print(f"\n{title}")
    print(f"{header[0]:<{width}}" + "".join(f"{h:>16}" for h in header[1:]))
    for r in rows:
        print(f"{r[0]:<{width}}" + "".join(f"{v:>16}" for v in r[1:]))


def num(x, fmt=".2f"):
    return "-" if x in ("", None) else format(float(x), fmt)


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("runs", nargs="+", help="result folders (each with summary.json, per_label.csv, per_bin.csv)")
    a.add_argument("--names", nargs="+", help="column names (default: folder names)")
    a.add_argument("--labels", type=int, default=0, help="only the N most common labels (0 = all)")
    args = a.parse_args()
    names = args.names or [Path(r).parent.name if Path(r).name in ("test", "val", "train") else Path(r).name
                           for r in args.runs]
    if len(names) != len(args.runs):
        raise SystemExit("--names needs one name per run folder")
    runs = [load(r) for r in args.runs]

    for n, (s, _, _) in zip(names, runs):
        print(f"{n}: {s['images']} images of {s['data']} ({s['split']}), line_buffer_px {s['line_buffer_px']}, "
              f"IoU {s['iou']}")

    rows = []
    for key in ("precision", "recall", "f1", "pixel_iou"):
        rows.append([f"overall {key}"] + [num(s["overall"][key]) for s, _, _ in runs])
    rows.append(["F1 95% interval"] + ["{:.3f}-{:.3f}".format(*s["overall_95ci"]["f1"]) if s.get("overall_95ci")
                                         else "-" for s, _, _ in runs])
    rows.append(["macro F1"] + [num(s["macro_over_labels"]["f1"]) for s, _, _ in runs])
    rows.append(["true / predicted"] + [f"{s['overall']['true']}/{s['overall']['predicted']}" for s, _, _ in runs])
    table("Overall", ["metric"] + names, rows, 18)

    true = {}
    for _, labels, _ in runs:
        for l, r in labels.items():
            if l != "ALL":
                true[l] = max(true.get(l, 0), int(r["true"]))
    order = sorted(true, key=lambda l: (-true[l], l))[:args.labels or None]
    table("F1 per label (true shapes in brackets)", ["label"] + names,
          [[f"{l} ({true[l]})"] + [num(lab[l]["f1"]) if l in lab else "-" for _, lab, _ in runs] for l in order], 24)

    keys = sorted({k for _, _, b in runs for k in b})
    table("Recall per geometry bin", ["factor: bin"] + names,
          [[f"{f}: {v}"] + [num(b[f, v]["recall"]) if (f, v) in b else "-" for _, _, b in runs] for f, v in keys], 38)


if __name__ == "__main__":
    main()
