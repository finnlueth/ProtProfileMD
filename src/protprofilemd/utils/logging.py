import logging
import os

from datasets import DatasetDict


def setup_default_logging(filename: str = None):

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    handlers = [stream_handler]
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file_handler = logging.FileHandler(filename, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_dataset_info(ds: DatasetDict, logger: logging.Logger):
    for split in ds:
        logger.info(
            f"Dataset {split} split info: "
            f"Number of examples: {len(ds[split])}, " 
            f"Features: {' '.join(ds[split].column_names)}"
        )
