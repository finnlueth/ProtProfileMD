import os

import matplotlib.pyplot as plt
import pandas as pd
import safetensors
import torch
import yaml
from datasets import DatasetDict
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
)
from transformers import T5Tokenizer, Trainer

from protprofilemd.model.protprofilemd import ProtProfileMD, ProtProfileMDConfig
from protprofilemd.plots import plot_training_history
from protprofilemd.utils.helpers import detect_device_type
from protprofilemd.utils.logging import get_logger

logger = get_logger(__name__)


def save_weights(model: ProtProfileMD, config: dict = None, log_history: dict = None) -> None:
    """
    Saves the base model lora adapter and decoder weights as a checkpoint to a directory.
    """
    checkpoint_directory = config["training"]["save_dir"]
    os.makedirs(checkpoint_directory, exist_ok=True)

    if isinstance(model, PeftModel):
        logger.info("Saving PEFT LoRA model")
        model.save_pretrained(checkpoint_directory, adapter_name=config["metadata"]["adapter_name"])
    else:
        logger.info("Saving full model")
        model.save_pretrained(checkpoint_directory)

    if config is not None:
        logger.info("Saving training configuration")
        with open(os.path.join(checkpoint_directory, "train_config.yaml"), "w") as f:
            yaml.dump(config, f)

    if log_history is not None:
        logger.info("Saving training history and plot")
        log_history_df = pd.DataFrame(log_history)
        log_history_df.to_csv(os.path.join(checkpoint_directory, "training_history.csv"), index=False)
        plt.close(plot_training_history(log_history_df).savefig(os.path.join(checkpoint_directory, "training_history.png")))


def load_weights(model: ProtProfileMD, checkpoint_directory: str, config: dict = None) -> None:
    """
    Loads adapter and profile head weights into an existing ProtProfileMD model.

    Args:
        model: The ProtProfileMD model instance to load weights into
        checkpoint_directory: Path to the checkpoint directory containing the weights
    """

    if isinstance(model, PeftModel):
        logger.info("Loading LoRA adapter weights")
        model.load_adapter(
            model_id=os.path.join(checkpoint_directory, config["metadata"]["adapter_name"]),
            adapter_name=config["metadata"]["adapter_name"],
            is_trainable=True,
        )
    else:
        logger.info("Loading full model weights")
        model.load_state_dict(
            safetensors.torch.load_file(os.path.join(checkpoint_directory, "model.safetensors")), strict=False, assign=True
        )


def build_model(config: dict) -> ProtProfileMD:
    """
    Builds a ProtProfileMD model from a configuration dictionary.
    Case 1: Training a new model
    Case 2: Loading a pretrained model with adapter weights for training
    Case 3: Loading a pretrained model with adapter weights for inference
    Args:
        config: Configuration dictionary containing the model configuration
    Returns:
        ProtProfileMD model instance
    """

    logger.info("Loading ProtProfileMD configuration")
    model_config = ProtProfileMDConfig(**config["model"])

    logger.info("Loading ProtProfileMD model")
    model = ProtProfileMD(config=model_config)

    if "lora" in config["training"] and config["training"]["lora"] is not None:
        logger.info("Using LoRA")
        lora_config = LoraConfig(**config["training"]["lora"])

        model = get_peft_model(model, peft_config=lora_config, adapter_name=config["metadata"]["adapter_name"])

        protein_encoder_trainable_params, protein_encoder_all_params = model.get_nb_trainable_parameters()
        profile_head_params = sum(p.numel() for p in model.profile_head.parameters())
        protein_encoder_trainable_params -= profile_head_params
        protein_encoder_all_params -= profile_head_params
        logger.info(
            f"Protein encoder parameters: trainable params: {protein_encoder_trainable_params:,d}, all params: {protein_encoder_all_params:,d}, trainable percentage: {100 * protein_encoder_trainable_params / protein_encoder_all_params:.4f}%"
        )
        logger.info(f"Profile head parameters: {profile_head_params:,d}")
    else:
        logger.info("Using full model")
        model_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {model_params:,d}")

    if "weights" in config["metadata"] and config["metadata"]["weights"] is not None:
        logger.info("Loading existing weights")
        load_weights(model, config["metadata"]["weights"], config)

        if "inference" in config["metadata"] and config["metadata"]["inference"] is True:
            logger.info("Setting model to inference mode and merging adapter weights")
            # model.set_adapter(config["metadata"]["adapter_name"], inference_mode=True)
            model.merge_and_unload()


    device_type = detect_device_type()
    logger.info(f"Moving model to {device_type}")
    model.to(device_type)
    
    if "quantize" in config["training"] and config["training"]["quantize"] is not None:
        logger.info(f"Quantizing model to {config['training']['quantize']}")
        model = model.to(dtype=getattr(torch, config["training"]["quantize"]))

    return model


def load_model_from_directory(directory: str) -> ProtProfileMD:
    with open(os.path.join(directory, "train_config.yaml"), "r") as f:
        config = yaml.safe_load(f)
        config["metadata"]["weights"] = directory
    return build_model(config)


def load_model_from_huggingface(model_name_or_path: str) -> ProtProfileMD:
    # todo: implement loading model from HuggingFace
    raise NotImplementedError("Loading model from HuggingFace not supported yet")


def build_trainer(model: ProtProfileMD, tokenizer: T5Tokenizer, config: dict, dataset: DatasetDict) -> Trainer:
    """
    Builds a trainer for the model.
    Args:
        model: The ProtProfileMD model instance
        tokenizer: The tokenizer instance
        config: The configuration dictionary
        dataset: The dataset dictionary
    Returns:
        Trainer instance
    """
    from transformers.training_args import TrainingArguments

    from protprofilemd.model.data_collator_training_args import DataCollatorProfile

    data_collator = DataCollatorProfile(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(**config["training"]["training_args"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )

    return trainer
