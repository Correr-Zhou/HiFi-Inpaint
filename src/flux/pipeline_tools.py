from diffusers.pipelines import FluxPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import logging
logger = logging.get_logger(__name__)
from torch import Tensor
import torch

def encode_images(pipeline: FluxPipeline, images: Tensor):
    images = pipeline.image_processor.preprocess(images)
    images = images.to(pipeline.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    images_tokens = pipeline._pack_latents(images, *images.shape)
    images_ids = pipeline._prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        pipeline.device,
        pipeline.dtype,
    )
    if images_tokens.shape[1] != images_ids.shape[0]:
        images_ids = pipeline._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2] // 2,
            images.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    return images_tokens, images_ids

import torch.nn.functional as F

def encode_mask(pipeline: FluxPipeline, mask: torch.Tensor):
    """
    Convert a [0,1] mask image (B,1,H,W) into latent-aligned tokens and IDs for conditioning.
    """
    B, _, H, W = mask.shape
    H_latent, W_latent = H // 8, W // 8

    mask = F.interpolate(mask, size=(H_latent, W_latent), mode="bilinear", align_corners=False)
    mask = (mask > 0.5).float()

    C_latent = 16
    mask = mask.repeat(1, C_latent, 1, 1)

    mask = mask.to(pipeline.device).to(pipeline.dtype)

    mask_tokens = pipeline._pack_latents(mask, *mask.shape)

    mask_ids = pipeline._prepare_latent_image_ids(
        B, H_latent, W_latent, pipeline.device, pipeline.dtype
    )

    if mask_tokens.shape[1] != mask_ids.shape[0]:
        mask_ids = pipeline._prepare_latent_image_ids(
            B, H_latent // 2, W_latent // 2, pipeline.device, pipeline.dtype
        )

    return mask_tokens, mask_ids

def prepare_text_input(pipeline: FluxPipeline, prompts, max_sequence_length=512):
    # Turn off warnings (CLIP overflow)
    logger.setLevel(logging.ERROR)
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompts,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=pipeline.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    # Turn on warnings
    logger.setLevel(logging.WARNING)
    return prompt_embeds, pooled_prompt_embeds, text_ids

def prepare_mask_input(pipeline: FluxPipeline, images, mask_images):
    batch_size = images.shape[0]
    height = images.shape[2]
    width = images.shape[3]
    device = pipeline.device
    dtype = pipeline.dtype
    # num_channels_latents = images.shape[1]
    num_channels_latents = pipeline.vae.config.latent_channels
    num_images_per_prompt = 1
    generator = None

    image = pipeline.image_processor.preprocess(images)
    mask_image = pipeline.mask_processor.preprocess(mask_images)

    masked_image = image * (1 - mask_image)
    masked_image = masked_image.to(device=device, dtype=dtype)

    height, width = image.shape[-2:]

    mask, masked_image_latents = pipeline.prepare_mask_latents(
        mask_image,
        masked_image,
        batch_size,
        num_channels_latents,
        num_images_per_prompt,
        height,
        width,
        dtype,
        device,
        generator,
    )

    masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)
    return masked_image_latents

    