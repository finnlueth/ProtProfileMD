#!/usr/bin/env python3
"""
Usage:
nohup \
python ./scripts/model_benchmarks.py \
    --targets "scope" \
    --fasta_aa "./tmp/data/scope/scope40_sequences_AA.fasta" \
    --benchmark_dir "./tmp/runs/ProtProfileMD_20251113_063248_large_batch_size/benchmarks" \
    --model_adapter_path "./tmp/runs/ProtProfileMD_20251113_063248_large_batch_size/model" \
    --batch_size "32" \
    >> custom_output_benchmark.log &
"""

import argparse
import subprocess

from protprofilemd.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the model")
    parser.add_argument("--targets", type=str, required=True, help="List of targets to benchmark")
    parser.add_argument("--fasta_aa", type=str, required=True, help="Path to the AA FASTA file")
    parser.add_argument("--benchmark_dir", type=str, required=True, help="Path to the directory")
    parser.add_argument("--model_adapter_path", type=str, required=True, help="Path to the model adapter")
    parser.add_argument("--batch_size", type=str, required=False, default="32", help="Batch size for inference")
    return parser.parse_args()


def main():
    args = parse_args()

    logger = get_logger("model_benchmarks")
    logger.info(f"Benchmarking targets: {args.targets}")
    logger.info(f"Fasta AA: {args.fasta_aa}")
    logger.info(f"Benchmark directory: {args.benchmark_dir}")
    logger.info(f"Model adapter path: {args.model_adapter_path}")
    logger.info(f"Batch size: {args.batch_size}")

    if "scope" in args.targets.split(","):
        cmd_inference = [
            "python",
            "./scripts/model_inference.py",
            "--model_adapter_path",
            args.model_adapter_path,
            "--input",
            args.fasta_aa,
            "--output",
            f"{args.benchmark_dir}/scope_40_benchmark/scope40_profiles.tsv",
            "--batch_size",
            args.batch_size,
            "--resume_from_tsv",
            "True",
        ]
        subprocess.run(cmd_inference, check=True, capture_output=False, text=True)
        logger.info("Inference completed successfully")

        cmd_argmax_profiles_without_background = [
            "python",
            "./src/protprofilemdanalysis/scripts-data/argmax_profiles.py",
            "--in_profile_path",
            f"{args.benchmark_dir}/scope_40_benchmark/scope40_profiles.tsv",
            "--out_fasta_path",
            f"{args.benchmark_dir}/scope_40_benchmark/scope40_3Di_argmax_without_background.fasta",
            "--subtract_background",
            "True",
        ]
        subprocess.run(cmd_argmax_profiles_without_background, check=True, capture_output=False, text=True)
        
        cmd_argmax_profiles_with_background = [
            "python",
            "./src/protprofilemdanalysis/scripts-data/argmax_profiles.py",
            "--in_profile_path",
            f"{args.benchmark_dir}/scope_40_benchmark/scope40_profiles.tsv",
            "--out_fasta_path",
            f"{args.benchmark_dir}/scope_40_benchmark/scope40_3Di_argmax_with_background.fasta",
            "--subtract_background",
            "False",
        ]
        subprocess.run(cmd_argmax_profiles_with_background, check=True, capture_output=False, text=True)
        logger.info("Argmax profiles completed successfully")

        scope_benchmark_profiles_without_background = [
            "bash",
            "./src/protprofilemdanalysis/scripts-benchmarks/runFoldseek_Profile_vs_3Di.sh",
            "--scope-bench-dir",
            "./src/protprofilemdanalysis/foldseek-analysis/scopbenchmark/",
            "--aa-fasta",
            args.fasta_aa,
            "--3di-fasta",
            f"{args.benchmark_dir}/scope_40_benchmark/scope40_3Di_argmax_without_background.fasta",
            "--profiles",
            f"{args.benchmark_dir}/scope_40_benchmark/scope40_profiles.tsv",
            "--data-script-dir",
            "./src/protprofilemdanalysis/scripts-data/",
            "--out-dir",
            f"{args.benchmark_dir}/scope_40_benchmark/without_background",
        ]
        subprocess.run(scope_benchmark_profiles_without_background, check=True, capture_output=False, text=True)
        
        scope_benchmark_profiles_with_background = [
            "bash",
            "./src/protprofilemdanalysis/scripts-benchmarks/runFoldseek_Profile_vs_3Di.sh",
            "--scope-bench-dir",
            "./src/protprofilemdanalysis/foldseek-analysis/scopbenchmark/",
            "--aa-fasta",
            args.fasta_aa,
            "--3di-fasta",
            f"{args.benchmark_dir}/scope_40_benchmark/scope40_3Di_argmax_with_background.fasta",
            "--profiles",
            f"{args.benchmark_dir}/scope_40_benchmark/scope40_profiles.tsv",
            "--data-script-dir",
            "./src/protprofilemdanalysis/scripts-data/",
            "--out-dir",
            f"{args.benchmark_dir}/scope_40_benchmark/with_background",
        ]
        subprocess.run(scope_benchmark_profiles_with_background, check=True, capture_output=False, text=True)
        
        logger.info("FoldSeek SCOPe benchmark completed successfully")


if __name__ == "__main__":
    main()
