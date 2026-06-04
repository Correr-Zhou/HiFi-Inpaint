#!/bin/bash

# HiFi-Inpaint Inference Script
# Usage: bash scripts/run_inference.sh

set -e

# ========== 配置路径 ==========
FLUX_PATH="/path/to/FLUX.1-dev"  # TODO: 改成你的 FLUX.1-dev 路径
LORA_PATH="/primus_xpfs_workspace_T04/liuyichen/HiFi-Inpaint-raw/ckpts/20250727-082134gen+real_data_caption+loss+texture/ckpt/10000"
HARD_CASE_DIR="/primus_xpfs_workspace_T04/liuyichen/HiFi-Inpaint-raw/code-training/ominicontrol-lyc/hard_case/input"
REAL_HARD_CASE_DIR="/primus_xpfs_workspace_T04/liuyichen/HiFi-Inpaint-raw/code-training/ominicontrol-lyc/real_hard_case/input"
OUTPUT_DIR="./output"

cd "$(dirname "$0")/.."

# ========== Case 1: hard_case_1 (草地+玻璃瓶) ==========
echo "===== Running hard_case_1 ====="
python scripts/infer_hard_case.py \
    --flux_path "${FLUX_PATH}" \
    --lora_path "${LORA_PATH}" \
    --ref_image "${HARD_CASE_DIR}/hard_case_1_ref.jpg" \
    --mask_image "${HARD_CASE_DIR}/hard_case_1_mask.png" \
    --prompt "A glass bottle labeled Hifi-Inpaint placed on the grass field" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42

# ========== Case 2: real_hard_case_1 (人物手持产品) ==========
echo "===== Running real_hard_case_1 ====="
python scripts/infer_hard_case.py \
    --flux_path "${FLUX_PATH}" \
    --lora_path "${LORA_PATH}" \
    --ref_image "${REAL_HARD_CASE_DIR}/real_hard_case_1.png" \
    --mask_image "${REAL_HARD_CASE_DIR}/real_hard_case_1_masked_img.png" \
    --mask_bw "${REAL_HARD_CASE_DIR}/real_hard_case_1_mask.png" \
    --prompt "A person holding a small bottle" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42

echo "===== Done! Results saved to ${OUTPUT_DIR} ====="
