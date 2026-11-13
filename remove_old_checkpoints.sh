#!/usr/bin/env bash
# all numbered checkpoints of all runs → bucket (uploads, then deletes the local copies)
for d in save_models/*/; do
    ls "$d"checkpoint0*.pth >/dev/null 2>&1 && \
    gcloud storage mv "$d"checkpoint0*.pth gs://vectorizing-raster-maps/checkpoints/$(basename "$d")/
done