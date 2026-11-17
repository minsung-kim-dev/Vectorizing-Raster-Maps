# OSM scripts vs. Raster2Seq's CubiCasa scripts

- Stage 1 (shapes): `training/train_osm.sh` vs. `tools/train_cc5k.sh`
- Stage 2 (labels): `training/finetune_osm.sh` vs. `tools/finetune_cc5k.sh`

Needs the fork (jennahgosciak/Raster2Seq, branch `osm-changes`) next to this repo. Only differences are listed.

## Stage 1

| setting | CubiCasa | OSM | why |
|---|---|---|---|
| `--dataset_name` | `cubicasa` | `osm` | fork's OSM loader |
| `--semantic_classes` | -1 | 3 + `--geom_classes` | learns polygon vs. line |
| `--image_size` | 256 | 512 | keeps small shapes |
| `--seq_len` | 512 | 1024 | more shapes per crop |
| `--batch_size` | 56 | 8 | larger images use more memory |
| `--row_tolerance` | – | 0.08 | shapes in reading order (fork) |
| `--epochs` / `--lr_drop` | 500 / none | 200 / 100 | |
| checkpoint / eval every | 50 / 50 | 10 / 250 | eval is run separately |
| starting model | Structured3D | CubiCasa (Hugging Face) | closer to rendered maps |

Also added: `--per_token_sem_loss --jointly_train --disable_wd_as_line`.

## Stage 2

| setting | CubiCasa | OSM |
|---|---|---|
| `--semantic_classes` | 12 | number of labels + 1 (from `create_coco_osm.py`) |
| geometry head | – | `--add_geom_head --geom_mask` (fork) |
| starting model | its stage 1 | its stage 1 |

## Loading the starting model (fork)

- Ignored if `--resume` finds a checkpoint.
- `--seq_len` can differ; `--num_bins` can't (both use 32).
- Weights with a different shape (e.g. label heads) are skipped and trained from scratch.
- Stage 1's label head initialises stage 2's geometry head.
