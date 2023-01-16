#!/bin/bash
# exit when any command fails
set -e
dvc pull -f
python -u src/models/train_model.py