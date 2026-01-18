import argparse
import gc
import os
import time
from contextlib import contextmanager

import torch

import wandb
from protprofilemd.utils.logging import get_logger

logger = get_logger(__name__)


def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


def check_parameters_equal(model1, model2, print_non_equal=True, print_equal=False):
    """Check if all parameters in two models are equal"""
    model1_params = dict(model1.named_parameters())
    model2_params = dict(model2.named_parameters())

    if set(model1_params.keys()) != set(model2_params.keys()):
        print("Models have different parameter names")
        return False

    all_equal = True
    for name in model1_params:
        if not torch.equal(model1_params[name], model2_params[name]):
            if print_non_equal:
                print(f"Parameter {name} is not equal")
            all_equal = False
        elif print_equal:
            print(f"Parameter {name} is equal")

    return all_equal


def has_lora_layers(model):
    """Check if model has LoRA parameters by looking at parameter names."""
    return any("lora_A" in name or "lora_B" in name for name in model.state_dict().keys())


def detect_device_type() -> str:
    """Prefer to use CUDA if available, use MPS if available, otherwise fallback to CPU."""
    
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device_type = "cuda" if n_gpus > 1 else "cuda:0"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    logger.info(f"Autodetected device type as {device_type}")
    return device_type


def print_model_device_info(model):
    """Print parameter names, devices, and device summary for a model."""
    devices_found = set()
    print("Parameter devices:")
    print("-" * 50)

    for name, param in model.named_parameters():
        device = param.device
        devices_found.add(str(device))
        print(f"{name}: {device}")

    print("\n" + "=" * 50)
    print("Device Summary:")
    print(f"Total unique devices found: {len(devices_found)}")
    for device in sorted(devices_found):
        print(f"  - {device}")


@contextmanager
def timer(description="Operation"):
    print(f"Starting {description}...")
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{description} completed in {elapsed:.4f} seconds")


def initialize_wandb(config: dict) -> wandb.Run:
    if "wandb" in config and config["wandb"] is not None:
        wandb.init(project=config["wandb"]["project"], name=config["metadata"]["run_id"])
        run = wandb.init(project=config["wandb"]["project"], name=config["metadata"]["run_id"])

    else:
        os.environ["WANDB_DISABLED"] = "true"
        run = None
    return run


def str_to_bool(value: str) -> bool:
    """Convert string to boolean for argparse."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
