import argparse
import os
import uuid

import yaml
from datasets import Dataset

from protprofilemd.data.fasta import fasta_to_dict
from protprofilemd.data.profile_csvs import parse_profiles
from protprofilemd.model.model_manager import load_model_from_directory
from protprofilemd.model.utils import TOKENIZER_MAP
from protprofilemd.utils.logging import setup_default_logging, get_logger
from protprofilemd.model.inference import profile_inference
from protprofilemd.utils.helpers import str_to_bool
from protprofilemd.data.train_processing import select_entries_by_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Inference for Profile Generation")
    parser.add_argument("--model_adapter_path", type=str, required=False, help="Path to the model directory")
    parser.add_argument("--input", type=str, required=True, help="Path to the FASTA file")
    parser.add_argument("--output", type=str, required=True, help="Path and name to the .tsv output file")
    parser.add_argument("--batch_size", type=int, required=False, default=8, help="Batch size for inference")
    parser.add_argument(
        "--resume_from_tsv",
        type=str_to_bool,
        required=False,
        default=True,
        help="Resume from existing TSV. Sequences present in the TSV are skipped, else TSV is overwritten.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_default_logging()
    
    logger = get_logger("profile_inference")
    logger.info(f"Run ID for inference: {str(uuid.uuid4())}")
    
    if not args.model_adapter_path:
        from huggingface_hub import hf_hub_download
        os.system("hf download finnlueth/ProtProfileMD")
        cached_file_path = hf_hub_download(repo_id="finnlueth/ProtProfileMD", filename="train_config.yaml")
        args.model_adapter_path = os.path.dirname(cached_file_path)
        print(f"Using HuggingFace model adapter at {args.model_adapter_path}")

    with open(os.path.join(args.model_adapter_path, "train_config.yaml"), "r") as f:
        train_config = yaml.safe_load(f)
    

    with open(args.input, "r") as f:
        parsed_fasta = fasta_to_dict(f.read())
    logger.info(f"Parsed {len(parsed_fasta)} sequences from FASTA file.")

    tokenizer = TOKENIZER_MAP[train_config["model"]["base_model"]].from_pretrained()
    logger.info(f"Loaded tokenizer for {train_config['model']['base_model']}")

    ds = Dataset.from_list([{"id": key, "sequence": value} for key, value in parsed_fasta.items()])
    logger.info(f"Loaded {len(ds)} sequences from FASTA file and converted to dataset.")
    logger.info(f"Dataset info: {ds}")

    tokenized_sequences = tokenizer.protein_encode(ds["sequence"])
    ds = ds.add_column("input_ids", tokenized_sequences["input_ids"])
    ds = ds.add_column("attention_mask", tokenized_sequences["attention_mask"])

    if os.path.exists(args.output) and args.resume_from_tsv:
        existing_profiles = dict(parse_profiles(args.output))
        existing_ids = set(existing_profiles.keys())
        missing_ids = set(ds["id"]) - existing_ids
        # logger.info(f"Existing IDs: {existing_ids}")
        # logger.info(f"Missing IDs: {missing_ids}")
        # ds = ds.filter(lambda x: x["id"] not in existing_ids)
        ds = select_entries_by_column(ds, "id", list(missing_ids))
        logger.info(f"Resuming from existing TSV. {len(existing_ids)} sequences already processed. Processing {len(missing_ids)} sequences.")
    elif os.path.exists(args.output) and not args.resume_from_tsv:
        os.remove(args.output)
        logger.info("Overwriting existing TSV.")
    logger.info(f"Dataset info: {ds}")

    logger.info(f"{len(ds)} sequences remaining to process.")

    model = load_model_from_directory(args.model_adapter_path)
    
    profile_inference(model=model, dataset=ds, tokenizer=tokenizer, batch_size=args.batch_size, save_path=args.output)
