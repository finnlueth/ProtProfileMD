#!/usr/bin/env python3

"""
Usage:
python ./src/protprofilemdanalysis/scripts-data/argmax_profiles.py \
    --in_profile_path ./tmp/runs/ProtProfileMD_20251113_072928_large_batch_size_again/benchmarks/scope_40_benchmark/scope40_profiles_without_background_subtractive.tsv \
    --out_fasta_path ./tmp/runs/ProtProfileMD_20251113_072928_large_batch_size_again/benchmarks/scope_40_benchmark/scope40_3Di_argmax_without_background_subtractive.fasta \
    --subtract_background True
"""

import argparse
import re

import numpy as np
from tqdm import tqdm

from protprofilemd.utils.helpers import str_to_bool
from protprofilemd.data.profile_csvs import profile_to_csv, parse_profiles

BACKGROUND_3Di = np.array(
    [
        0.0489372,
        0.0306991,
        0.101049,
        0.0329671,
        0.0276149,
        0.0416262,
        0.0452521,
        0.030876,
        0.0297251,
        0.0607036,
        0.0150238,
        0.0215826,
        0.0783843,
        0.0512926,
        0.0264886,
        0.0610702,
        0.0201311,
        0.215998,
        0.0310265,
        0.0295417,
    ]
)
ALPHABET_AA = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
ALPHABET_3Di = [x.lower() for x in ALPHABET_AA]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_profile_path", type=str, required=True)
    parser.add_argument("--out_fasta_path", type=str, required=True)
    parser.add_argument("--subtract_background", type=str_to_bool, default=True)
    parser.add_argument("--out_profile_path", type=str, required=False)
    return parser.parse_args()


def main():
    import os
    import sys

    args = parse_args()
    in_profile_path = args.in_profile_path
    out_fasta_path = args.out_fasta_path
    subtract_background = args.subtract_background
    out_profile_path = args.out_profile_path

    profile_dict = parse_profiles(in_profile_path)
    profile_dict = dict(profile_dict)

    argmax_fasta = []
    profile_tsvs = []

    for k, v in tqdm(profile_dict.items(), desc="Processing profiles"):
        if subtract_background:
            v = v - BACKGROUND_3Di
            if out_profile_path:
                profile_tsvs.append(profile_to_csv(k, v))
        argmax_fasta.append(f">{k}\n" + "".join([ALPHABET_AA[x] for x in v.argmax(axis=1)]))

    argmax_fasta = "\n".join(argmax_fasta)
    
    if out_profile_path:
        with open(out_profile_path, "w") as f:
            f.write("".join(profile_tsvs))

    with open(out_fasta_path, "w") as f:
        f.write(argmax_fasta)


if __name__ == "__main__":
    main()
