# `train_osm.sh` vs. Raster2Seq's training scripts

Side-by-side comparison of this repo's `tools/train_osm.sh` with Raster2Seq's CubiCasa scripts
(Raster2Seq commit `a6c4e27`):

- **Stage 1** (shapes only): `train_osm.sh` step 4 vs. Raster2Seq's `tools/train_cc5k.sh`
- **Stage 2** (adds labels): `train_osm.sh` step 5 vs. Raster2Seq's `tools/finetune_cc5k.sh`

Rows marked ✱ differ.

## Stage 1 (shapes only): `train_cc5k.sh` vs. `train_osm.sh` step 4

| setting | `train_cc5k.sh` | `train_osm.sh` | why |
|---|---|---|---|
| launcher | `torchrun --nproc_per_node=1 main_ddp.py` | same | |
| `--master_port` | 14159 | 14159 | |
| `export NCCL_P2P_LEVEL=NVL` ✱ | set | not set | only affects communication between several GPUs; irrelevant with 1 |
| `WANDB_MODE` ✱ | `online` | `disabled` + placeholder key file | so it doesn't stop and ask for a key; switch back if you set up wandb |
| `--dataset_name` ✱ | `cubicasa` | `waffle` | resizes and pads to a square without dropping categories |
| `--dataset_root` ✱ | `data/coco_cubicasa5k_…` | `data/coco_osm` | your data |
| `--semantic_classes` | `-1` | `-1` | shapes only |
| `--job_name` ✱ | `cc5k_res256` | `osm_stage1` | |
| `--output_dir` | `save_models` | `save_models` (in this repo) | |
| `--batch_size` ✱ | **56** | **8** | images are 4× larger and sequences 4× longer, so much more memory per image |
| `--image_size` ✱ | not set → **256** | **512** | crops are ~1,100 px; 256 loses small shapes |
| `--input_channels` | 3 | 3 | |
| `--poly2seq` | ✓ | ✓ | |
| `--seq_len` ✱ | **512** | **2048** | covers ~90% of the crops completely |
| `--num_bins` | 32 | 32 | must match the starting model |
| `--ckpt_every_epoch` / `--eval_every_epoch` | 50 / 50 | 50 / 50 | |
| `--label_smoothing` | 0.1 | 0.1 | |
| `--epochs` | 500 | 500 | |
| `--lr_drop` | `''` | `''` | |
| `--cls_loss_coef` / `--coords_loss_coef` | 5 / 20 | 5 / 20 | |
| `--resume` | `save_models/<job>/checkpoint.pth` | same pattern | continues a run if interrupted |
| `--ema4eval`, `--disable_poly_refine`, `--dec_attn_concat_src`, `--use_anchor` | ✓ | ✓ | |
| `--converter_version` | v3 | v3 | |
| `--start_from_checkpoint` ✱ | `checkpoints/s3dbw_res256_ep1349.pth`, a local file: the Structured3D black-and-white model | `cubicasa5k` from Hugging Face, downloaded with `raster2seq_hub` | CubiCasa floorplans are closer to real images than black-and-white plans |

## Stage 2 (adds labels): `finetune_cc5k.sh` vs. `train_osm.sh` step 5

Everything shared with stage 1 matches as above. The stage-2-specific settings:

| setting | `finetune_cc5k.sh` | `train_osm.sh` |
|---|---|---|
| `--semantic_classes` ✱ | 12 (11 labels + 1) | read from `CATEGORIES` in `data_preprocess/create_coco_osm.py` (number of labels + 1) |
| `--room_cls_loss_coef` | 1 | 1 |
| `--per_token_sem_loss`, `--jointly_train` | ✓ | ✓ |
| `--master_port` ✱ | 14156 | 14159 |
| `--start_from_checkpoint` | its own stage-1 model (`checkpoints/cc5k_res256_ep0499.pth`) | its own stage-1 model (`save_models/osm_stage1/checkpoint.pth`) |

## What the checkpoint loader allows (`main_ddp.py`, lines 417–468)

- **`--seq_len` can differ from the starting model.** When sizes don't match, the loader swaps in the new
  model's freshly initialised sequence-length-dependent parts (`pos_embed`, `attention_mask`). So starting
  from a 512-length model with `--seq_len 2048` is fine.
- **`--num_bins` can't.** The position-token table (`token_embed`) only gets one row added, for an extra
  class token, so a different grid size fails to load. That's why `train_osm.sh` uses 32.
- **Extra parts in the checkpoint are skipped.** It loads with `strict=False`. The Hugging Face `cubicasa5k`
  model includes a label head, which the stage-1 model doesn't have, so those weights are listed as
  "Unexpected Keys" and ignored rather than causing an error.
