from torch.utils.data import DataLoader
import torch
import lightning as L
import yaml
import os
import time

from .data import DatasetGenAI
from .model import OminiModel
from .callbacks import TrainingCallback

torch.cuda.empty_cache()


def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


def main():
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    config = get_config()
    training_config = config["train"]
    if os.environ.get("RUN_NAME") is not None:
        run_name = time.strftime("%Y%m%d-%H%M%S") + os.environ.get("RUN_NAME")
    else:
        run_name = time.strftime("%Y%m%d-%H%M%S")

    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    if training_config["condition_type"] in ["subject"]:
        data_path = training_config["dataset"]["data_path"]
        dataset = DatasetGenAI(
            data_path,
            resolution=(1024, 576),
            use_caption=training_config.get("use_caption", False),
            random_mask=training_config.get("random_mask", False),
        )
    else:
        raise NotImplementedError

    print("Dataset length:", len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )

    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_path=config.get("lora_path"),
        lora_config=training_config["lora_config"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        fft_loss_config=training_config.get("fft_loss_config", '0'),
        use_texture_image=training_config.get("use_texture_image", False),
    )

    training_callbacks = (
        [TrainingCallback(run_name, training_config=training_config)]
        if is_main_process
        else []
    )

    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_checkpointing=True,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    trainer.fit(trainable_model, train_loader)


if __name__ == "__main__":
    main()
