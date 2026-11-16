"""Quick checks that the scorer gives the answers we expect on predictions with a known score.

Uses the first 2 test crops of data/coco_osm_mini (in git), so it needs no crops and runs in a few seconds:
    python evaluation/test_score.py
"""

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluation.score import load_coco, score  # noqa: E402

DATA, SPLIT, N = REPO / "data" / "coco_osm_mini", "test", 2


def truth():
    images, gts = load_coco(DATA, SPLIT)
    return {i["file_name"]: [{k: s[k] for k in ("label", "kind", "points")} for s in gts[i["id"]]]
            for i in images[:N]}


def run(name, preds):
    with tempfile.TemporaryDirectory() as out:
        print(f"\n===== {name}")
        return score(DATA, SPLIT, preds, out, name, limit=N)["overall"]


def check(name, ok, got):
    print(f"{'PASS' if ok else 'FAIL'}: {name} -> {got}")
    return ok


def main():
    gt = truth()
    results = []

    s = run("truth vs itself", gt)
    results.append(check("truth vs itself scores 1.0", s["precision"] == s["recall"] == s["pixel_iou"] == 1.0, s))

    s = run("no predictions", {f: [] for f in gt})
    results.append(check("empty predictions score 0", s["recall"] == 0 and s["predicted"] == 0, s))

    half = {f: shapes[::2] for f, shapes in gt.items()}
    s = run("every other shape", half)
    results.append(check("half the shapes: precision 1, recall about 0.5",
                         s["precision"] == 1.0 and 0.45 <= s["recall"] <= 0.55, s))

    shifted = {f: [{**x, "points": [[px + 1, py + 1] for px, py in x["points"]]} for x in shapes]
               for f, shapes in gt.items()}
    s = run("shifted 1 px", shifted)
    results.append(check("a 1 px shift still matches most shapes", s["recall"] >= 0.8, s))

    relabelled = {f: [{**x, "label": "unknown"} for x in shapes] for f, shapes in gt.items()}
    s = run("every label wrong", relabelled)
    results.append(check("right shapes with wrong labels don't match", s["matched"] <= s["true"] * 0.05, s))

    doubled = {f: shapes + shapes for f, shapes in gt.items()}
    s = run("every shape twice", doubled)
    results.append(check("duplicates count once: recall 1, precision 0.5",
                         s["recall"] == 1.0 and s["precision"] == 0.5, s))

    junk = [{"label": "building", "kind": "polygon", "points": [[0, 0], [5, 5]]},  # too few points
            {"label": "street", "kind": "line", "points": [[3, 3]]},
            {"label": "street", "kind": "line", "points": []},
            {"label": "building", "kind": "polygon", "points": [[0, 0], [9, 0], [float("nan"), 9]]}]
    s = run("truth plus unscorable shapes", {f: shapes + junk for f, shapes in gt.items()})
    results.append(check("shapes with too few / non-finite points are dropped, not crashed on",
                         s["precision"] == s["recall"] == 1.0, s))

    print(f"\n{sum(results)} of {len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
