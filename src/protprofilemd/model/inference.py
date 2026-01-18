import os
from typing import Any

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..data.profile_csvs import profile_to_csv
from ..utils.helpers import detect_device_type
from ..utils.logging import get_logger
from .data_collator_training_args import DataCollatorProfile
from .protprofilemd import ProtProfileMD

logger = get_logger(__name__)


def profile_inference(
    model: ProtProfileMD,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    batch_size: int,
    save_path: str,
    pad_to_multiple_of: int = None,
) -> None:
    """
    Perform inference using HuggingFace DataLoader and DataCollator and all available GPUs (DataParallel if >1 GPU).

    Args:
        model: The ProtProfileMD model for inference
        dataset: HuggingFace Dataset containing tokenized sequences
        tokenizer: Tokenizer used for encoding sequences
        batch_size: Batch size for inference
        save_path: Path to save the predictions
    """
    logger.info("Starting profile inference")

    device = detect_device_type()
    logger.info(f"Device: {device}")
    
    if "cuda" in device and torch.cuda.device_count() > 1:
        batch_size = batch_size * torch.cuda.device_count()
        model = torch.nn.DataParallel(model)
        logger.info(f"Model parallelized on {torch.cuda.device_count()} GPUs")
    
    model.to(device)
    logger.info("Model moved to device")
    model.eval()
    logger.info("Model set to evaluation mode")
    
    data_collator = DataCollatorProfile(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=pad_to_multiple_of,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with torch.inference_mode():
        for batch_index in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset.select(range(batch_index, min(batch_index + batch_size, len(dataset))))

            columns_to_keep = ["input_ids", "attention_mask"]
            batch_input_features = data_collator(
                batch.remove_columns([col for col in batch.column_names if col not in columns_to_keep])
            )
            
            input_ids = batch_input_features["input_ids"].to(device, non_blocking=True)
            attention_mask = batch_input_features["attention_mask"].to(device, non_blocking=True)

            model_output = model(input_ids=input_ids, attention_mask=attention_mask)
            # torch.cuda.empty_cache()

            # logger.info(f"Generating profiles for {len(batch)} sequences")
            profile_tsvs = []
            for name, sequence, profile, mask in zip(
                batch["id"], batch["sequence"], model_output["profiles"], model_output["masks"]
            ):
                parsed_profile = profile[mask.cpu().numpy().astype(bool)].detach().cpu().numpy()
                profile_tsvs.append(profile_to_csv(name, parsed_profile))
                # logger.info(
                #     f"Generating profile for sequence {name} of sequence length {len(sequence)} with profile shape {parsed_profile.shape}"
                # )

            # logger.info(f"Saving profiles to {save_path}")
            with open(save_path, "a") as f:
                f.write("".join(profile_tsvs))

    logger.info("Profile inference completed successfully")
