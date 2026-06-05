import lightning as L
from PIL import Image
import numpy as np
import cv2
import torch
import os
import io
from torchvision import transforms
from diffusers.utils import load_image

try:
    import wandb
except ImportError:
    wandb = None

from ..flux.condition import Condition
from ..flux.generate import generate


def apply_high_pass_filter(image_or_path) -> Image.Image:
    if isinstance(image_or_path, str):
        image = cv2.imread(image_or_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read image from path: {image_or_path}")
        gray_image_np = image
    elif isinstance(image_or_path, Image.Image):
        gray_image_np = np.array(image_or_path.convert('L'))
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
    detail_image = Image.fromarray(np.uint8(img_back))
    return detail_image


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )
        self.use_texture_image = self.training_config.get("use_texture_image", False)
        self.total_steps = 0
        self.use_caption = training_config.get("use_caption", False)

        self.test_images_dir = training_config.get("test_images_dir", "assets/test")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        if self.use_wandb:
            report_dict = {
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, "
                f"Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, "
                f"Max gradient size: {max_gradient_size:.4f}"
            )

        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        if self.total_steps % self.sample_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            self.generate_a_sample(
                trainer,
                pl_module,
                f"{self.save_path}/{self.run_name}/output",
                f"lora_{self.total_steps}",
                batch["condition_type"][0],
            )

    def get_resize_and_crop_transforms_first_image(self, resolution):
        return transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
        ])

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
        condition_type="super_resolution",
    ):
        generator = torch.Generator(device=pl_module.device)
        generator.manual_seed(42)

        test_list = self._load_test_cases(condition_type)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i, (condition_img, background_img, mask, texture_img, position_delta, prompt) in enumerate(test_list):
            if not self.use_caption:
                prompt = " "

            texture_img = texture_img.convert("RGB")

            texture_img_condition = Condition(
                condition_type=condition_type,
                condition=texture_img,
                position_delta=None,
            )
            masked_image_condition = Condition(
                condition_type=condition_type,
                condition=background_img.resize(background_img.size).convert("RGB"),
                position_delta=None,
            )
            ref_image_condition = Condition(
                condition_type=condition_type,
                condition=condition_img.convert("RGB"),
                position_delta=position_delta,
            )

            if self.use_texture_image:
                conditions_texture = [texture_img_condition, masked_image_condition]
                conditions_texture_mask = mask
            conditions = [ref_image_condition, masked_image_condition]

            res = generate(
                pl_module.flux_pipe,
                prompt=prompt,
                conditions=conditions,
                conditions_texture=conditions_texture if self.use_texture_image else None,
                conditions_texture_mask=conditions_texture_mask if self.use_texture_image else None,
                image=condition_img,
                mask_image=background_img,
                height=background_img.height,
                width=background_img.width,
                generator=generator,
                model_config=pl_module.model_config,
                use_texture_image=self.use_texture_image,
                default_lora=True,
            )
            res.images[0].save(
                os.path.join(save_path, f"{file_name}_{condition_type}_{i}.jpg")
            )

    def _load_test_cases(self, condition_type):
        """Load test cases from test_images_dir. Override this method to customize."""
        test_list = []
        if condition_type != "subject":
            return test_list

        test_dir = self.test_images_dir
        if not os.path.exists(test_dir):
            print(f"[WARNING] Test images directory not found: {test_dir}")
            return test_list

        import glob
        ref_files = sorted(glob.glob(os.path.join(test_dir, "*_ref.*")))
        for ref_path in ref_files:
            prefix = ref_path.rsplit("_ref", 1)[0]
            ext = os.path.splitext(ref_path)[1]

            mask_path = f"{prefix}_mask{ext}"
            mask_bw_path = f"{prefix}_mask_bw{ext}"
            prompt_path = f"{prefix}_prompt.txt"

            if not os.path.exists(mask_path) or not os.path.exists(mask_bw_path):
                continue

            ref_img = Image.open(ref_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("RGB")
            mask_bw = Image.open(mask_bw_path)
            texture = apply_high_pass_filter(ref_img)
            height = np.array(ref_img).shape[0]

            prompt = ""
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r') as f:
                    prompt = f.read().strip()

            test_list.append((
                ref_img, mask_img, mask_bw, texture,
                [0, -height // 16], prompt
            ))

        return test_list
