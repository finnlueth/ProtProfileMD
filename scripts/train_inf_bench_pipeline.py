"""
Usage:
python ./scripts/train_inf_bench_pipeline.py

or

nohup python /home/repos/bachelor-thesis/protprofilemd/scripts/train_inf_bench_pipeline.py \
    --config_path /home/repos/bachelor-thesis/protprofilemd/configs/protprofile_train.yaml \
    > log_7.log &
"""

import subprocess
import yaml
import tempfile
from datetime import datetime
import os
from pprint import pprint
import argparse

from protprofilemd.utils.logging import setup_default_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=False, default="./configs/protprofile_train.yaml")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    run_id = config["metadata"]["name"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + config["metadata"]["run_id"]
    
    run_dir = os.path.join("./tmp/runs/", run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    config["metadata"]["run_id"] = run_id
    config["training"]["save_dir"] = os.path.join(run_dir, "model")
    config["training"]["training_args"]["output_dir"] = os.path.join(run_dir, "training")
    config["metadata"]["log_path"] = os.path.join(run_dir, "logs")
    
    setup_default_logging(os.path.join(run_dir, "logs", f"{run_id}_pipeline.log"))
    logger = get_logger("train_inf_bench_pipeline")
    logger.info("Running training and inference benchmark pipeline")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Run directory: {run_dir}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_config:
        temp_config_path = temp_config.name
        yaml.dump(config, temp_config)

    if "ddp" in config["training"] and config["training"]["ddp"] is True:
        cmd_train = [
            "accelerate",
            "launch",
            "--config_file",
            "./configs/accelerate_default_config.yaml",
            "./scripts/model_training.py",
            "--config_path",
            temp_config_path,
        ]
    else:
        cmd_train = [
            "python",
            "./scripts/model_training.py",
            "--config_path",
            temp_config_path,
        ]
    logger.info("Starting training")
    subprocess.run(cmd_train, check=True, capture_output=False, text=True)
    logger.info("Training completed successfully")
    
    cmd_benchmark = [
        "python",
        "./scripts/model_benchmarks.py",
        "--targets",
        ",".join(config["benchmarks"]["targets"]) + ",",
        "--fasta_aa",
        "./tmp/data/scope/scope40_sequences_AA.fasta", #! REMEMBER: Change to the actual FASTA file
        "--benchmark_dir",
        f"{run_dir}/benchmarks",
        "--model_adapter_path",
        config["training"]["save_dir"],
        "--batch_size",
        str(config["benchmarks"]["batch_size"]),
    ]
    logger.info("Starting benchmarking")
    subprocess.run(cmd_benchmark, check=True, capture_output=False, text=True)
    logger.info("Benchmarking completed successfully")
    
    logger.info("Training, Inference, and Benchmarking pipeline completed successfully")

if __name__ == "__main__":
    main()
