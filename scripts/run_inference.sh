#!/bin/bash
set -e

# ========== Path Configuration ==========
FLUX_PATH="black-forest-labs/FLUX.1-dev"
LORA_PATH="<path-to-lora-weights>"
DEMO_DIR="./assets/demo"
OUTPUT_DIR="./output"

cd "$(dirname "$0")/.."

# ========== Case 1: Black jar + person holding ==========
echo "===== Running Case 1: person holding black jar ====="
python scripts/infer_hard_case.py \
    --base_model_path "${FLUX_PATH}" \
    --lora_path "${LORA_PATH}" \
    --ref_image "${DEMO_DIR}/case1_ref.jpg" \
    --mask_image "${DEMO_DIR}/case1_condition.png" \
    --mask_bw "${DEMO_DIR}/case1_mask.png" \
    --prompt "A person wearing a white top holding a matte black jar labeled \"Hifi-Inpaint perfect restoration, every time\" with a golden lotus logo. The background is a neutral color with minimalist style." \
    --output "${OUTPUT_DIR}/case1_result.png" \
    --target_size 576 1024 \
    --seed 42

# ========== Case 2: Glass bottle + close-up ==========
echo "===== Running Case 2: close-up person holding glass bottle ====="
python scripts/infer_hard_case.py \
    --base_model_path "${FLUX_PATH}" \
    --lora_path "${LORA_PATH}" \
    --ref_image "${DEMO_DIR}/case2_ref.jpg" \
    --mask_image "${DEMO_DIR}/case2_condition.png" \
    --mask_bw "${DEMO_DIR}/case2_mask.png" \
    --prompt "A close-up of a person holding a clear glass bottle with a wooden cap labeled \"Hifi-Inpaint artistry meets precision\" with circular pattern decorations. The person has neatly manicured nails, focusing on the bottle." \
    --output "${OUTPUT_DIR}/case2_result.png" \
    --target_size 576 1024 \
    --seed 42

echo "===== Done! Results saved to ${OUTPUT_DIR} ====="
