# Vectorizing-Raster-Maps

## Installation

- Clone [Raster2Seq](https://github.com/Cornell-VAILab/Raster2Seq) next to this repo (`../Raster2Seq`), at the commit this project is tested with:
  ```shell
  git clone https://github.com/Cornell-VAILab/Raster2Seq.git ../Raster2Seq
  git -C ../Raster2Seq checkout a6c4e27   # 2026-06-02
  ```
- Create an environment:
  ```shell
  conda create -n raster2seq python=3.10
  conda activate raster2seq
  ```
- Install pytorch and required libraries:
  ```shell
  # adjust the cuda version accordingly (omit --index-url on macOS / CPU)
  pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
  pip install -r requirements.txt
  ```
- For training (GPU only), compile Raster2Seq's CUDA ops as described in its README.

## Data processing

### 1. Get the crops

`crops/` holds one pair of files per map crop:
- `<name>.geojson.gz` – OpenMapTiles vector features for the crop
- `<name>.png` – the rendered map image

Unzip `crops.zip` into `crops/`, or regenerate the crops from OpenStreetMap with `data_preprocess/osm_crops.py` (see the instructions at the top of that file).

### 2. Convert to Raster2Seq's COCO format

```shell
conda activate raster2seq
python data_preprocess/create_coco_osm.py --data_root crops --output data/coco_osm
```

By default, line features remain lines. To convert line features (such as roads,
railways, and waterways) into buffered polygons, pass a positive buffer radius in
the original PNG's pixel units:

```shell
python data_preprocess/create_coco_osm.py --data_root crops \
  --output data/coco_osm_buffer --line_buffer_px 2
```

For example, a radius of 2 pixels extends approximately 2 pixels on each side of
a line. Use a separate output directory to keep buffered and unbuffered datasets.
Buffered lines are saved as polygons with their original category labels. In a
buffered run, annotations also include `source_geom_type` to record whether the
original geometry was a `line` or `polygon`. Without buffering, that field is not
added. Buffered polygons are also subject to `--min_area` and `--simplify_tol`.

This writes the layout Raster2Seq expects:
```
data/coco_osm/
├── train/  val/  test/                 # PNGs
└── annotations/
    └── train.json  val.json  test.json # COCO annotations (polygons and lines in PNG pixels)
```

Splits: crops named `queens_*` form the test set, and 10% of the rest go to val.

Options:
- `--test_borough` – borough used as the test set (default `queens`)
- `--val_frac` – fraction of non-test crops used for val (default `0.1`)
- `--min_area` – drop polygons smaller than this, in px² (default `20`)
- `--simplify_tol` – shape simplification tolerance in px, `0` = off (default `1.0`)
- `--seed` – random seed for the train/val split (default `0`)
- `--line_buffer_px` – buffer radius in original PNG pixels for line features; must be positive (default: no buffering)
- `--sample_name` – convert one paired crop by name, without the file extension; writes it to the test split
- `--max_crops` – convert at most this many paired crops in filename order, before splitting (default `0` = all)

Limiting crops before splitting can leave a validation or test split empty.

For example, to convert and inspect just one crop with buffering:

```shell
python data_preprocess/create_coco_osm.py --data_root crops \
  --output data/coco_osm_sample --sample_name queens_01156 --line_buffer_px 2
```

This single-crop example writes `test/queens_01156.png` and `annotations/test.json`.
When `--sample_name` is supplied, it takes precedence over `--max_crops`.

Labels come from the fixed vocabulary `VOCAB` in `create_coco_osm.py` (e.g. `street`, `building`, `park`). Anything not listed becomes `unknown` / `unknown_line`. When training with labels, set `--semantic_classes` to the number of labels + 1 (Raster2Seq reserves the last class). Rerun the conversion after any change to `VOCAB`, because category IDs follow its order.

`data/coco_osm/` is about 2.6 GB and can be regenerated with the command above, so don't commit it to git.

`_scratch/eda.ipynb` walks through the conversion for a single crop and plots the result against the PNG.

## Evaluation

`evaluation/score.py` is the shared scorer: every method (colour baseline, Claude, Raster2Seq) writes its shapes to a predictions JSON, and this script scores them the same way against a COCO split. It needs only the COCO dataset, not Raster2Seq.

```shell
python evaluation/score.py --pred results/<method>/test/predictions.json --data data/coco_osm --split test
python evaluation/score.py --pred ground_truth      # sanity check: scores the truth against itself, should be 1.0
python evaluation/test_score.py                      # 6 checks with known answers (empty, half, shifted, relabelled, duplicated)
python evaluation/compare.py results/<run A>/test results/<run B>/test   # print scored runs side by side (no rerun)
```

Predictions format, coordinates in pixels of the PNG as stored in the dataset (Raster2Seq output must be scaled back from its 512 px padded input first):

```json
{
  "method": "raster2seq stage 1",
  "images": {
    "queens_01156.png": [
      {"label": "building", "kind": "polygon", "points": [[x, y], [x, y], ...]},
      {"label": "street",   "kind": "line",    "points": [[x, y], [x, y], ...]}
    ]
  }
}
```

`label` is a category name from `VOCAB` in `create_coco_osm.py`; `kind` is `polygon` (closed, first point not repeated) or `line` (open). Images missing from the file are scored as empty.

Outputs, next to the predictions file:
- `per_label.csv`: true / predicted / matched shapes, precision, recall, F1 and pixel IoU per label
- `per_bin.csv`: recall split by the true shapes' geometry (size at the 512 px model input, vertex count, convex or concave, corner angles, overlapping another shape, touching the crop edge, line or polygon)
- `per_image.csv`, and `summary.json` with overall and macro-averaged scores plus a 95% bootstrap interval over images

Shapes match when they share a label and overlap with IoU ≥ 0.5, one-to-one, best pairs first. Lines are scored as buffered polygons, like the training data from Minsung's `--line_buffer_px` (default 2 px each side; pass the value the dataset was built with, or `0` to score centrelines).

## Baselines

### Colour threshold (no learning)

`baselines/color_threshold.py` finds shapes by their known map colours (OpenCV contours for areas, skeletons for roads/rails/paths), writes them to `predictions.json` and scores them with `evaluation/score.py`. It runs on CPU, no GPU needed.

```shell
pip install opencv-python-headless scikit-image scipy shapely numpy pillow matplotlib
python baselines/color_threshold.py                                      # mini test set: 12 Queens crops
python data_preprocess/create_coco_osm.py --data_root crops_125m --output data/coco_osm_125m_buffered --line_buffer_px 2
python baselines/color_threshold.py --data data/coco_osm_125m_buffered --line_buffer_px 2   # full Queens test set
```

Results go to `results/color_threshold/<split>/` (or `--out`). Use `--plots N` for side-by-side plots of the first N crops.

On the 1,500 Queens test crops of `crops_125m` (buffered lines, 2 px): overall F1 0.69 (precision 0.62, recall 0.76), buildings 0.93, pools 0.94, paths 0.41, streets 0.20.
