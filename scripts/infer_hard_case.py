"""
HiFi-Inpaint inference on hard_case test images.

Usage:
    python scripts/infer_hard_case.py \
        --flux_path /path/to/FLUX.1-dev \
        --lora_path /path/to/ckpt/10000 \
        --ref_image hard_case/input/hard_case_1_ref.jpg \
        --mask_image hard_case/input/hard_case_1_mask.png \
        --output_dir ./output \
        --prompt "A glass bottle labeled Hifi-Inpaint placed on the grass field"
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from diffusers.pipelines import FluxPipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.flux.condition import Condition
from src.flux.generate import generate


def apply_high_pass_filter(image: Image.Image) -> Image.Image:
    gray_image_np = np.array(image.convert('L'))
    rows, cols = gray_image_np.shape
    dft = cv2.dft(np.float32(gray_image_np), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    crow, ccol = rows // 2, cols // 2
    r = 30
    mask = np.ones((rows, cols, 2), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow)**2 + (y - ccol)**2 <= r**2
    mask[mask_area] = 0
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    max_val = np.max(img_back)
    if max_val > 0:
        img_back = (img_back / max_val) * 255
    return Image.fromarray(np.uint8(img_back))


def extract_mask(image: Image.Image) -> Image.Image:
    gray = np.array(image.convert('L'))
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area, max_rect = 0, None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > max_area:
            max_area = w * h
            max_rect = (x, y, w, h)
    mask = np.zeros_like(gray)
    if max_rect:
        x, y, w, h = max_rect
        mask[y:y+h, x:x+w] = 255
    return Image.fromarray(mask)


def load_pipeline(flux_path, lora_path, device="cuda"):
    print(f"Loading FLUX from {flux_path}...")
    pipe = FluxPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16)
    pipe.to(device)

    print(f"Loading LoRA from {lora_path}...")
    pipe.load_lora_weights(
        lora_path,
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="subject",
    )

    alpha_path = os.path.join(lora_path, "alpha_blocks.pt")
    if os.path.exists(alpha_path):
        print(f"Loading alpha weights from {alpha_path}...")
        alpha_dict = torch.load(alpha_path, map_location=device)
        for i, block in enumerate(pipe.transformer.transformer_blocks):
            if f"alpha_block_{i}" in alpha_dict:
                setattr(block, 'alpha', nn.Parameter(alpha_dict[f"alpha_block_{i}"].clone().detach()))
    return pipe


def run_inference(pipe, ref_image, mask_image, prompt, mask_bw_image=None, seed=42):
    device = pipe.device
    generator = torch.Generator(device=device).manual_seed(seed)

    ref_img = ref_image.convert("RGB")
    masked_img = mask_image.convert("RGB")
    mask_bw = mask_bw_image if mask_bw_image else extract_mask(mask_image)
    texture = apply_high_pass_filter(ref_img).convert("RGB")

    height = np.array(ref_img).shape[0]
    position_delta = [0, -height // 16]

    ref_condition = Condition("subject", ref_img, position_delta=position_delta)
    masked_condition = Condition("subject", masked_img, position_delta=None)
    texture_condition = Condition("subject", texture, position_delta=None)

    conditions = [ref_condition, masked_condition]
    conditions_texture = [texture_condition, masked_condition]

    result = generate(
        pipe,
        prompt=prompt,
        conditions=conditions,
        conditions_texture=conditions_texture,
        conditions_texture_mask=mask_bw,
        image=ref_img,
        mask_image=masked_img,
        height=masked_img.height,
        width=masked_img.width,
        generator=generator,
        use_texture_image=True,
        default_lora=True,
    )
    return result.images[0]


def parse_args():
    parser = argparse.ArgumentParser(description="HiFi-Inpaint Hard Case Inference")
    parser.add_argument("--flux_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--mask_image", type=str, required=True)
    parser.add_argument("--mask_bw", type=str, default=None, help="B/W mask image (auto-extracted from mask_image if not provided)")
    parser.add_argument("--prompt", type=str, default="A glass bottle placed on the grass field")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pipe = load_pipeline(args.flux_path, args.lora_path)

    ref_image = Image.open(args.ref_image)
    mask_image = Image.open(args.mask_image)
    mask_bw_image = Image.open(args.mask_bw) if args.mask_bw else None

    print(f"Running inference: ref={args.ref_image}, mask={args.mask_image}")
    result = run_inference(pipe, ref_image, mask_image, args.prompt, mask_bw_image, args.seed)

    out_name = os.path.splitext(os.path.basename(args.mask_image))[0] + "_result.png"
    out_path = os.path.join(args.output_dir, out_name)
    result.save(out_path)
    print(f"Result saved to {out_path}")


if __name__ == "__main__":
    main()
