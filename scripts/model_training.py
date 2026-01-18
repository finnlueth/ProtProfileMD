"""
Train a ProtProfileMD model.

Example Usage:
python ./scripts/model_training.py \
    --config_path './configs/protprofile_train.yaml'
"""

import argparse
import os
import random
from datetime import datetime

import accelerate
import datasets
import matplotlib.pyplot as plt
import pandas as pd
import torch
import transformers
import yaml
from accelerate import Accelerator

from protprofilemd.data.train_processing import select_entries_by_column, prepare_dataset_for_training
from protprofilemd.model.model_manager import build_model, build_trainer, save_weights
from protprofilemd.model.protein_tokenizer import ProstT5Tokenizer
from protprofilemd.utils.helpers import initialize_wandb, print_model_device_info
from protprofilemd.utils.logging import get_logger, setup_default_logging

SEED = 42
accelerate.utils.set_seed(SEED + 1)
transformers.set_seed(SEED + 2)
torch.manual_seed(SEED + 3)
random.seed(SEED + 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    run_id = config["metadata"]["run_id"]
    setup_default_logging(os.path.join(config["metadata"]["log_path"], f"{run_id}_training.log"))

    logger = get_logger("model_training")
    logger.info(f"Run ID for training: {run_id}")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config["training"]["devices"]
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    ds = datasets.load_from_disk(config["dataset"]["directory"])
    logger.info(f"Loaded dataset from {config['dataset']['directory']}")
    logger.info(f"Dataset info: {ds}")

    for column, values in config["dataset"]["filter_on_column"].items():
        for split in ds:
            ds[split] = select_entries_by_column(dataset=ds[split], column=column, entries=values)
    ds = prepare_dataset_for_training(
        ds_dict=ds,
        drop_model="protT5",
        keep_model="prostT5",
    )
    logger.info("Prepared dataset for training")

    # ! REMEMBER: Remove this before training
    # for split in ds:
    #     ds[split] = ds[split].select(range(0, 69))
    # ds["train"] = ds["train"].select(range(0, 123))

    logger.info(f"Dataset info: {ds}")

    model = build_model(config)

    accelerator = None
    wandb_run = None
    if "ddp" in config["training"] and config["training"]["ddp"] is True:
        logger.info("Using DDP")
        accelerator = Accelerator()
        logger.info(f"Total number of processes: {accelerator.num_processes}")
        # print(f"Is main process: {accelerator.is_main_process}")
        if accelerator.is_main_process:
            wandb_run = initialize_wandb(config)
    else:
        wandb_run = initialize_wandb(config)

    trainer = build_trainer(
        model=model,
        tokenizer=ProstT5Tokenizer.from_pretrained(),
        config=config,
        dataset=ds,
    )

    if accelerator is not None:
        model, trainer = accelerator.prepare(model, trainer)
        if not accelerator.is_main_process:
            trainer.args.report_to = []

    trainer.train()
    trainer.evaluate()

    if accelerator is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            model = accelerator.unwrap_model(model)
            logger.info("Unwrapped model")
            save_weights(model=model, config=config, log_history=trainer.state.log_history)
            logger.info("Saved weights, training configuration, and training history")
        accelerator.wait_for_everyone()
    else:
        save_weights(model=model, config=config, log_history=trainer.state.log_history)
        logger.info("Saved weights, training configuration, and training history")

    if wandb_run is not None:
        wandb_run.finish()

    logger.info("Training completed successfully")
