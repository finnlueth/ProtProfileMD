#!/usr/bin/env python3

"""
Usage:
python ./src/protprofilemdanalysis/scripts-data/subtract_background.py \
    --in_profile_path ./tmp/runs/ProtProfileMD_20251113_072928_large_batch_size_again/benchmarks/scope_40_benchmark/scope40_profiles.tsv\
    --out_profile_path ./tmp/runs/ProtProfileMD_20251113_072928_large_batch_size_again/benchmarks/scope_40_benchmark/scope40_profiles_without_background_logodds.tsv
"""

import argparse

import numpy as np
from scipy.special import softmax
from tqdm import tqdm

from protprofilemd.data.profile_csvs import parse_profiles, profile_to_csv

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_profile_path", type=str, required=True)
    parser.add_argument("--out_profile_path", type=str, required=True)
    return parser.parse_args()


def background_corrected_profile(P, B, eps=1e-8, method="logodds"):
    P, B = np.array(P), np.array(B)
    if method == "logodds":
        # L = np.log((P + eps) / (B + eps))
        # return softmax(L, axis=1)
        
        # return softmax(P - B, axis=1)
        
        # sigmoid = 1 / (1 + np.exp(-L))  # sigmoid bounded [0,1]
        # return sigmoid / sigmoid.sum(axis=1, keepdims=True)  # normalize to sum to 1
        # return sigmoid
        pass
    elif method == "subtractive":
        diff = np.clip(P - B, 0, None)
        if diff.sum() == 0:
            return np.zeros_like(diff)
        return diff / diff.sum(axis=1, keepdims=True)
    elif method == "ratio":
        R = P / (B + eps)
        return R / R.sum(axis=1, keepdims=True)
    elif method == "naive":
        return P - B
    else:
        raise ValueError("method must be one of ['logodds','subtractive','ratio','naive]")


def main():
    args = parse_args()
    in_profile_path = args.in_profile_path
    out_profile_path = args.out_profile_path

    profile_dict = dict(parse_profiles(in_profile_path))

    profile_tsvs = []
    for name, profile in tqdm(profile_dict.items(), desc="Subtracting background"):
        corrected_profile = background_corrected_profile(P=profile, B=BACKGROUND_3Di, method="logodds")
        # print(BACKGROUND_3Di.shape)
        # print(profile.shape)
        # corrected_profile = softmax(corrected_profile, axis=1)
        profile_tsvs.append(profile_to_csv(name, corrected_profile))
        # if len(profile_tsvs) == 3:
        #     break

    with open(out_profile_path, "w") as f:
        f.write("".join(profile_tsvs))


if __name__ == "__main__":
    main()
