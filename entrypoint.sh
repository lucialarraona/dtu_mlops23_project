#!/bin/bash
dvc pull -f
python -u src/models/train_model.py