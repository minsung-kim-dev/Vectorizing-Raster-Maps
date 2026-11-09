# `main_ddp.py` walkthrough

## Before `main` (lines 612–628)
1. Parse the arguments (`get_args_parser`, line 25).
2. Set the run name to `--job_name` and the output folder to `<output_dir>/<job_name>`, and create it.
3. Turn `--lr_drop "400"` into a list of epochs. `--debug` forces batch size 1.

## `main(args)`
1. **Multi-GPU setup** (197–206): one process per GPU (`torchrun`), with GPU 0 as rank 0. Set the random seeds.
2. **wandb** (210–212), rank 0 only: read or ask for the key (`wandb_keys/`), then `wandb.init(project="Raster2Seq", id=job_name)`.
3. **Data** (215–308):
   - Build the train and val datasets with the loader (`datasets/poly_data.py` for `waffle`), which resizes to `--image_size` and turns shapes into token sequences.
   - Get the tokenizer and vocabulary size, which depend on `--num_bins`.
   - Make the data loaders, which split batches across GPUs. `--debug` trains on 1 val image.
4. **Model** (311–314):
   - Build Raster2Seq and its loss (`criterion`).
   - Copy the model as the **EMA** model, a slowly updated average of the weights.
   - Wrap the model for multi-GPU training.
5. **Freezing** (330–336): with `--per_token_sem_loss` but not `--jointly_train`, train only the label classifier.
6. **Optimizer and schedule** (338–377): AdamW at `--lr`, 10× lower for the backbone (`--lr_backbone`) and attention offsets, with a ×0.1 drop at each `--lr_drop` epoch.
7. **`--resume`** (379–420): if `checkpoint.pth` exists, restore the model, optimizer, schedule and epoch, so an interrupted run carries on.
8. **`--start_from_checkpoint`** (422–477): load pretrained weights (e.g. CubiCasa), adapting tensors whose shapes differ (class head, token embedding, positional embedding, attention mask, input projection, attention offsets/weights). `query_embed` (the anchors, `seq_len × 2`) has no adaptation branch, so a different `--seq_len` fails.
9. **Sync the EMA** (480) to the starting weights.
10. **Epoch loop** (485–), for each epoch:
    - **Train** (`engine.train_one_epoch`): for each batch, forward with teacher forcing, compute the weighted loss (`cls_loss_coef·loss_ce + coords_loss_coef·loss_coords`), backpropagate, clip gradients (`--clip_max_norm`), take an optimizer step and update the EMA (decay 0.999). It stops if the loss becomes NaN.
    - **Learning-rate schedule:** step it.
    - **Checkpoint** every `--ckpt_every_epoch` epochs: save `checkpoint.pth` plus `checkpointNNNN.pth`, each with the model, EMA, optimizer and epoch.
    - **Evaluate** every `--eval_every_epoch` epochs (`engine.evaluate`): run real generation on val with the raw model, or the EMA with `--ema4eval`, compute metrics and save plots.
    - **Log** losses, learning rate and metrics to wandb and `log.txt`.
11. **Finish:** shut down the multi-GPU processes.
