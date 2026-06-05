"""
HiFi-Inpaint inference script.
"""
import os, sys
import cv2
import torch
import torch.nn as nn
import json
import numpy as np
from PIL import Image, ImageOps
from diffusers.utils import load_image
from diffusers.pipelines import FluxPipeline
import argparse

from src.flux.condition import Condition
from src.flux.generate import generate, seed_everything


def extract_mask(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_rect = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area:
            max_area = area
            max_rect = (x, y, w, h)
    mask = np.zeros_like(image)
    if max_rect:
        x, y, w, h = max_rect
        mask[y:y+h, x:x+w] = 255
    return Image.fromarray(mask)


def apply_high_pass_filter(image: Image.Image) -> Image.Image:
    gray_image_np = np.array(image.convert('L'))
    rows, cols = gray_image_np.shape
    dft = cv2.dft(np.float32(gray_image_np), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    crow, ccol = rows // 2, cols // 2
    r = 30
    mask = np.ones((rows, cols, 2), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0])**2 + (y - center[1])**2 <= r**2
    mask[mask_area] = 0
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    max_val = np.max(img_back)
    if max_val > 0:
        img_back = (img_back / max_val) * 255
    return Image.fromarray(np.uint8(img_back))


class ImageProcessor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = self._setup_pipeline()
        self.generator = torch.Generator(device=self.device).manual_seed(config['seed'])

    def _setup_pipeline(self):
        print("Setting up the pipeline...")
        pipe = FluxPipeline.from_pretrained(
            self.config['base_model_path'], torch_dtype=torch.bfloat16
        )
        pipe.to(self.device)

        print(f"Loading LoRA weights from: {self.config['lora_path']}")
        pipe.load_lora_weights(
            self.config['lora_path'],
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="subject",
        )
        self._load_alpha_weights(pipe.transformer, self.config['lora_path'])
        print("Pipeline setup complete.")
        return pipe

    def _load_alpha_weights(self, transformer, lora_path):
        alpha_path = os.path.join(lora_path, "alpha_blocks.pt")
        if not os.path.exists(alpha_path):
            print(f"Warning: Alpha weights not found at {alpha_path}. Skipping.")
            return
        print(f"Loading alpha weights from {alpha_path}...")
        alpha_dict = torch.load(alpha_path, map_location=self.device)
        for i, block in enumerate(transformer.transformer_blocks):
            alpha_value = alpha_dict[f"alpha_block_{i}"]
            setattr(block, 'alpha', nn.Parameter(alpha_value.clone().detach()))

    def _align_size(self, w, h, divisor=16):
        return (w // divisor * divisor, h // divisor * divisor)

    def process_single(self, ref_image_path, masked_img_path, mask_bw_path=None, prompt=" ", save_path="output.png", target_size=None, ref_size=None):
        control_image = load_image(ref_image_path).convert("RGB")
        masked_img = load_image(masked_img_path).convert("RGB")

        if ref_size:
            control_image = control_image.resize(ref_size)

        if target_size:
            masked_img = masked_img.resize(target_size)
            aligned_w, aligned_h = target_size
        else:
            aligned_w, aligned_h = self._align_size(masked_img.width, masked_img.height)
            if (aligned_w, aligned_h) != masked_img.size:
                masked_img = masked_img.resize((aligned_w, aligned_h))

        if mask_bw_path:
            mask = load_image(mask_bw_path).convert("L").resize((aligned_w, aligned_h))
        else:
            mask = extract_mask(masked_img_path).resize((aligned_w, aligned_h))

        position_delta = [0, -aligned_h // 16]

        ref_image_condition = Condition(
            condition_type="subject",
            condition=control_image,
            position_delta=position_delta,
        )
        masked_image_condition = Condition(
            condition_type="subject",
            condition=masked_img,
            position_delta=None,
        )
        conditions = [ref_image_condition, masked_image_condition]

        if self.config['use_texture_image']:
            texture_image = apply_high_pass_filter(control_image).convert('RGB')
            texture_img_condition = Condition(
                condition_type="subject",
                condition=texture_image,
                position_delta=None,
            )
            conditions_texture = [texture_img_condition, masked_image_condition]
            conditions_texture_mask = mask
        else:
            conditions_texture = None
            conditions_texture_mask = None

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        res_image = generate(
            self.pipe,
            prompt=prompt,
            conditions=conditions,
            conditions_texture=conditions_texture if self.config['use_texture_image'] else None,
            conditions_texture_mask=conditions_texture_mask if self.config['use_texture_image'] else None,
            image=control_image,
            mask_image=masked_img,
            height=aligned_h,
            width=aligned_w,
            generator=self.generator,
            use_texture_image=self.config['use_texture_image'],
            default_lora=True,
        ).images[0]

        res_image.save(save_path)
        print(f"Result saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="HiFi-Inpaint Inference")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--mask_image", type=str, required=True)
    parser.add_argument("--mask_bw", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=" ")
    parser.add_argument("--output", type=str, default="./output/result.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_texture_image", action="store_true", default=True)
    parser.add_argument("--target_size", type=int, nargs=2, default=None,
                        metavar=("W", "H"), help="Resize masked_img to WxH (e.g. 576 1024)")
    parser.add_argument("--ref_size", type=int, nargs=2, default=None,
                        metavar=("W", "H"), help="Resize ref image to WxH (e.g. 768 768)")
    args = parser.parse_args()

    config = {
        'base_model_path': args.base_model_path,
        'lora_path': args.lora_path,
        'seed': args.seed,
        'use_caption': True,
        'use_texture_image': args.use_texture_image,
    }

    processor = ImageProcessor(config)
    processor.process_single(
        ref_image_path=args.ref_image,
        masked_img_path=args.mask_image,
        mask_bw_path=args.mask_bw,
        prompt=args.prompt,
        save_path=args.output,
        target_size=tuple(args.target_size) if args.target_size else None,
        ref_size=tuple(args.ref_size) if args.ref_size else None,
    )


if __name__ == "__main__":
    main()
