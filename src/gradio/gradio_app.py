import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers.pipelines import FluxPipeline
import torch.nn as nn

from ..flux.condition import Condition
from ..flux.generate import generate


pipe = None


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


def init_pipeline(base_model_path="black-forest-labs/FLUX.1-dev",
                  lora_path=None, alpha_path=None):
    global pipe
    pipe = FluxPipeline.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cuda")

    if lora_path:
        pipe.load_lora_weights(
            lora_path,
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="subject",
        )
        _alpha_path = alpha_path or f"{lora_path}/alpha_blocks.pt"
        if _alpha_path and torch.cuda.is_available():
            import os
            if os.path.exists(_alpha_path):
                alpha_dict = torch.load(_alpha_path, map_location="cuda")
                for i, block in enumerate(pipe.transformer.transformer_blocks):
                    if f"alpha_block_{i}" in alpha_dict:
                        setattr(block, 'alpha', nn.Parameter(alpha_dict[f"alpha_block_{i}"].clone().detach()))


def process_image_and_text(ref_image, masked_image, mask_bw, text, use_texture=True):
    if pipe is None:
        raise RuntimeError("Pipeline not initialized. Call init_pipeline() first.")

    ref_image = ref_image.convert("RGB")
    masked_image = masked_image.convert("RGB")

    height = np.array(ref_image).shape[0]
    position_delta = [0, -height // 16]

    ref_condition = Condition("subject", ref_image, position_delta=position_delta)
    masked_condition = Condition("subject", masked_image, position_delta=None)
    conditions = [ref_condition, masked_condition]

    conditions_texture = None
    conditions_texture_mask = None
    if use_texture:
        texture = apply_high_pass_filter(ref_image).convert("RGB")
        texture_condition = Condition("subject", texture, position_delta=None)
        conditions_texture = [texture_condition, masked_condition]
        conditions_texture_mask = mask_bw

    result = generate(
        pipe,
        prompt=text.strip(),
        conditions=conditions,
        conditions_texture=conditions_texture,
        conditions_texture_mask=conditions_texture_mask,
        image=ref_image,
        mask_image=masked_image,
        height=masked_image.height,
        width=masked_image.width,
        use_texture_image=use_texture,
        default_lora=True,
    )
    return result.images[0]


demo = gr.Interface(
    fn=process_image_and_text,
    inputs=[
        gr.Image(type="pil", label="Reference Image"),
        gr.Image(type="pil", label="Masked Image"),
        gr.Image(type="pil", label="Mask (B/W)"),
        gr.Textbox(lines=2, label="Prompt"),
        gr.Checkbox(value=True, label="Use Texture (SEA)"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="HiFi-Inpaint: Reference-Based Image Inpainting",
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--lora_path", type=str, required=True)
    args = parser.parse_args()

    init_pipeline(args.base_model_path, args.lora_path)
    demo.launch(debug=True)
