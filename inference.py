import os
import time
import yaml
import torch
import argparse

from src.train.callbacks import TrainingCallback
from src.train.model import OminiModel


def parse_args():
    parser = argparse.ArgumentParser(description="HiFi-Inpaint Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to trained LoRA weights")
    parser.add_argument("--save_path", type=str, default="./output", help="Path to save generated images")
    parser.add_argument("--test_images_dir", type=str, default="assets/test", help="Directory containing test images")
    return parser.parse_args()


def main():
    args = parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = rank == 0
    torch.cuda.set_device(rank)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S") + "_inference"

    model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_path=args.lora_path,
        lora_config=training_config.get("lora_config", None),
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        fft_loss_config=training_config.get("fft_loss_config", '0'),
        use_texture_image=training_config.get("use_texture_image", False),
    )

    training_config_with_test = dict(training_config)
    training_config_with_test["test_images_dir"] = args.test_images_dir
    callback = TrainingCallback(run_name, training_config=training_config_with_test)

    if is_main_process:
        callback.generate_a_sample(
            trainer=None,
            pl_module=model,
            save_path=args.save_path,
            file_name='inference',
            condition_type="subject",
        )

    print(f"Inference complete. Results saved to {args.save_path}")


if __name__ == "__main__":
    main()
