# Vectorizing-Raster-Maps

## Installation

- Clone [Raster2Seq](https://github.com/Cornell-VAILab/Raster2Seq) next to this repo (`../Raster2Seq`).
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

Labels come from the fixed vocabulary `VOCAB` in `create_coco_osm.py` (e.g. `street`, `building`, `park`). Anything not listed becomes `unknown` / `unknown_line`. When training with labels, set `--semantic_classes` to the number of labels + 1 (Raster2Seq reserves the last class). Rerun the conversion after any change to `VOCAB`, because category IDs follow its order.

`data/coco_osm/` is about 2.6 GB and can be regenerated with the command above, so don't commit it to git.

`_scratch/eda.ipynb` walks through the conversion for a single crop and plots the result against the PNG.
