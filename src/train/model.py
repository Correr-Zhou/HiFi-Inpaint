import lightning as L
from diffusers.pipelines import FluxPipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import os
from peft import LoraConfig, get_peft_model_state_dict
import numpy as np
from PIL import Image
import prodigyopt
import cv2
from ..flux.transformer import tranformer_forward
from ..flux.condition import Condition
from ..flux.pipeline_tools import encode_images, prepare_text_input, prepare_mask_input, encode_mask


def apply_mask_to_image(images, mask_images):
    mask_images = mask_images.to(images.device).to(images.dtype)
    if mask_images.dim() == 4:
        if mask_images.size(1) != images.size(1):
            if mask_images.size(1) == 1:
                mask_images = mask_images.expand(-1, images.size(1), -1, -1)
            else:
                raise ValueError(f"Mask channels {mask_images.size(1)} must be 1 or match image channels {images.size(1)}")
    masked_images = images * mask_images
    return masked_images


def apply_high_pass_filter(image_tensor: torch.Tensor,
                          radius: int = 30) -> torch.Tensor:
    """
    Batch high-pass filter (supports autograd).
    Args:
        image_tensor: [B, C, H, W] range [0,1]
        radius: cutoff frequency radius
    Returns:
        high_pass: [B, 1, H, W] range [0,1]
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(1)

    if image_tensor.size(1) == 3:
        gray = 0.2126 * image_tensor[:,0] + 0.7152 * image_tensor[:,1] + 0.0722 * image_tensor[:,2]
    else:
        gray = image_tensor.mean(dim=1)

    gray_255 = (gray * 255).to(torch.float32)
    with torch.cuda.amp.autocast(enabled=image_tensor.dtype==torch.bfloat16):
        fft = torch.fft.fft2(gray_255)
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))

        B, H, W = gray_255.shape
        y = torch.linspace(-H//2, H//2-1, H, device=image_tensor.device)
        x = torch.linspace(-W//2, W//2-1, W, device=image_tensor.device)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        mask = (X**2 + Y**2) > radius**2
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        mask = mask.to(fft_shift.dtype)

        filtered = fft_shift * mask
        inv_shift = torch.fft.ifftshift(filtered, dim=(-2, -1))
        inv_fft = torch.fft.ifft2(inv_shift)

        magnitude = torch.abs(inv_fft)

        min_val = magnitude.amin(dim=(-1, -2), keepdim=True)
        max_val = magnitude.amax(dim=(-1, -2), keepdim=True)
        normalized = (magnitude - min_val) / (max_val - min_val + 1e-6)

        return normalized.unsqueeze(1)


class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        fft_loss_config: str = '0',
        use_texture_image: bool = False,
    ):
        super().__init__()
        self.fft_loss_config = fft_loss_config
        self.use_texture_image = use_texture_image
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        self.flux_pipe: FluxPipeline = (
            FluxPipeline.from_pretrained(flux_pipe_id).to(dtype=dtype).to(device)
        )
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()

        self.lora_layers = self.init_lora(lora_path, lora_config)
        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config, "Must provide either lora_path or lora_config."

        adapter_name = "subject"

        if lora_path:
            print(f"[LoRA] Loading LoRA weights from {lora_path}")
            self.flux_pipe.load_lora_weights(
                lora_path,
                weight_name="pytorch_lora_weights.safetensors",
                adapter_name=adapter_name,
            )
        else:
            print(f"[LoRA] Initializing new LoRA adapter '{adapter_name}' from config")
            self.transformer.add_adapter(LoraConfig(**lora_config), adapter_name=adapter_name)

        if self.use_texture_image:
            self.alpha_blocks = []
            for index_block, block in enumerate(self.transformer.transformer_blocks):
                alpha = nn.Parameter(torch.tensor(0.0))
                setattr(block, 'alpha', alpha)
                self.register_parameter(f'alpha_{index_block}', alpha)
                self.alpha_blocks.append(alpha)
            self.trainable_params = [p for n, p in self.transformer.named_parameters() if p.requires_grad] + self.alpha_blocks

            if lora_path:
                alpha_path = os.path.join(lora_path, "alpha_blocks.pt")
                if os.path.exists(alpha_path):
                    print(f"[LoRA] Loading alpha weights from {alpha_path}")
                    alpha_dict = torch.load(alpha_path)
                    for i, block in enumerate(self.transformer.transformer_blocks):
                        if f"alpha_block_{i}" in alpha_dict:
                            block.alpha.data.copy_(alpha_dict[f"alpha_block_{i}"])

            return self.trainable_params

        self.trainable_params = [
            p for n, p in self.transformer.named_parameters() if p.requires_grad
        ]
        return self.trainable_params

    def save_lora(self, path: str):
        try:
            adapter_name = "subject"
            if adapter_name not in self.transformer.peft_config:
                print(f"[save_lora] Adapter '{adapter_name}' not found. Skipping.")
                return

            transformer_lora_layers = get_peft_model_state_dict(self.transformer, adapter_name=adapter_name)
            FluxPipeline.save_lora_weights(
                save_directory=path,
                transformer_lora_layers=transformer_lora_layers,
                safe_serialization=True,
            )
            print(f"[save_lora] LoRA weights saved to {path}")

            if self.use_texture_image:
                alpha_dict = {
                    f"alpha_block_{i}": alpha.detach().cpu()
                    for i, alpha in enumerate(getattr(self, "alpha_blocks", []))
                }
                torch.save(alpha_dict, os.path.join(path, "alpha_blocks.pt"))
                print("[save_lora] alpha_blocks.pt saved.")

        except Exception as e:
            print(f"[save_lora] Failed to save LoRA. Reason: {e}")

    def configure_optimizers(self):
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        self.trainable_params = self.lora_layers
        if self.use_texture_image:
            self.trainable_params += getattr(self.transformer, "alpha_blocks", [])

        for p in self.trainable_params:
            p.requires_grad_(True)

        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def step(self, batch):
        imgs = batch["image"]
        masks = batch["mask"]
        mask_area = batch["mask_area"]
        conditions = batch["condition"]
        condition_types = batch["condition_type"]
        prompts = batch["description"]
        texture = batch['texture']
        position_delta = batch["position_delta"][0]

        with torch.no_grad():
            x_0, img_ids = encode_images(self.flux_pipe, imgs)
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_pipe, prompts
            )

            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            condition_latents, condition_ids = encode_images(self.flux_pipe, conditions)
            masked_image_latents, masked_image_ids = encode_images(self.flux_pipe, masks)

            condition_latents = torch.cat((condition_latents, masked_image_latents), dim=1)
            if self.use_texture_image:
                condition_latents_texture_mask, condition_ids_texture = encode_mask(self.flux_pipe, mask_area)
                texture_latents, texture_ids = encode_images(self.flux_pipe, texture)
                condition_latents_texture = torch.cat((texture_latents, masked_image_latents), dim=1)

            condition_ids = torch.cat((condition_ids, masked_image_ids), dim=0)

            condition_type_ids = torch.tensor(
                [Condition.get_type_id(ct) for ct in condition_types]
            ).to(self.device)
            condition_type_ids = (
                torch.ones_like(condition_ids[:, 0]) * condition_type_ids[0]
            ).unsqueeze(1)

            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        transformer_out = tranformer_forward(
            self.transformer,
            model_config=self.model_config,
            condition_latents=condition_latents,
            condition_latents_texture=condition_latents_texture if self.use_texture_image else None,
            condition_latents_texture_mask=condition_latents_texture_mask if self.use_texture_image else None,
            condition_ids=condition_ids,
            condition_type_ids=condition_type_ids,
            hidden_states=x_t,
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
            use_texture_image=self.use_texture_image,
        )
        pred = transformer_out[0]

        if self.fft_loss_config != '0':
            latents = (x_t - t_ * pred)
            height = imgs[0].shape[1]
            width = imgs[0].shape[2]
            latents = self.flux_pipe._unpack_latents(latents, height, width, self.flux_pipe.vae_scale_factor)
            latents = (
                latents / self.flux_pipe.vae.config.scaling_factor
            ) + self.flux_pipe.vae.config.shift_factor
            latents = latents.to(pred.dtype)
            image = self.flux_pipe.vae.decode(latents, return_dict=False)[0]

            image_pred = apply_mask_to_image(image, mask_area)
            image_gt = apply_mask_to_image(imgs, mask_area)
            image_pred = self.flux_pipe.image_processor.postprocess(image_pred, output_type="pt")
            image_gt = self.flux_pipe.image_processor.postprocess(image_gt, output_type="pt")

            if self.fft_loss_config == '1':
                pred_detail = apply_high_pass_filter(image_pred)
                gt_detail = apply_high_pass_filter(image_gt)
                fft_loss = F.l1_loss(pred_detail, gt_detail, reduction="mean")
            elif self.fft_loss_config == '2':
                pred_detail = apply_high_pass_filter(image_pred)
                gt_detail = apply_high_pass_filter(image_gt)
                fft_loss = F.mse_loss(pred_detail, gt_detail, reduction="mean")

            loss = F.mse_loss(pred, (x_1 - x_0), reduction="mean") + fft_loss
        else:
            loss = F.mse_loss(pred, (x_1 - x_0), reduction="mean")

        self.last_t = t.mean().item()
        return loss
