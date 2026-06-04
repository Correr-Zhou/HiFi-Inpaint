#!/bin/bash

# HiFi-Inpaint Training Script
# Usage: bash train/scripts/train.sh

# Set config path
export XFL_CONFIG="train/config/subject_512.yaml"

# Optional: set run name
export RUN_NAME="hifi_inpaint"

# Optional: set wandb API key for logging
# export WANDB_API_KEY="your_wandb_api_key"

# Single GPU training
python -m src.train.train

# Multi-GPU training with torchrun
# torchrun --nproc_per_node=8 -m src.train.train
