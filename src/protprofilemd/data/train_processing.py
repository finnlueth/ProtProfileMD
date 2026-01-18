import numpy as np
from datasets import Dataset, DatasetDict

from protprofilemd.utils.logging import log_dataset_info, get_logger

logger = get_logger(__name__)


def select_entries_by_column(dataset: Dataset, column: str, entries: list[str] | set[str]) -> Dataset:
    logger.info(f"Filtering dataset by column {column} with values {entries}")
    mask = np.isin(dataset[column], entries)
    return dataset.select(np.where(mask)[0])


def prepare_dataset_for_training(dataset: DatasetDict, drop_model: str, keep_model: str) -> DatasetDict:
    for split in dataset:
        logger.info(f"Processing dataset {split} split")
        dataset[split] = dataset[split].remove_columns(
            ["domain_name", "aa_sequence", "replica", "temperature", f"input_ids_{drop_model}", f"attention_mask_{drop_model}"]
        )
        dataset[split] = dataset[split].rename_column(f"input_ids_{keep_model}", "input_ids")
        dataset[split] = dataset[split].rename_column(f"attention_mask_{keep_model}", "attention_mask")
    log_dataset_info(dataset, logger)
    return dataset
