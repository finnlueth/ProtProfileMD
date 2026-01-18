import re
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.definitions import AA_ALPHABET


def profile_to_csv(name: str, profile: np.ndarray) -> str:
    df_profile = pd.DataFrame(profile)
    df_profile = df_profile.round(4)

    tsv_string = f"Query profile of sequence {name}\n"
    tsv_string += "     " + "      ".join(AA_ALPHABET) + "      \n"

    df_string = df_profile.to_csv(index=False, sep=" ", float_format="%.4f", header=False, lineterminator="\n")
    df_string = df_string.replace("\n", " \n")  # Add trailing space before newlines
    tsv_string += df_string

    return tsv_string


def csv_to_profile(tsv_string: str) -> Dict[str, np.ndarray]:
    lines = tsv_string.strip().split("\n")
    profiles = []
    current_block = []
    header_pattern = re.compile(r"Query profile of sequence\s+(.*)")

    for line in tqdm(lines, desc="Parsing profile lines"):
        if header_pattern.match(line) and current_block:
            profiles.append(current_block)
            current_block = []
        current_block.append(line)
    if current_block:
        profiles.append(current_block)

    profile_data = []
    for block in tqdm(profiles, desc="Processing profile blocks"):
        if len(block) < 3:
            continue
        header_line = block[0].strip()
        m = header_pattern.match(header_line)
        key = m.group(1).strip() if m else ""
        profile_matrix = parse_profile_from_block(block)
        profile_data.append((key, profile_matrix))

    return profile_data


def parse_profile_from_block(block_lines):
    # Skip the first two header lines (header and amino acid header)
    data_lines = block_lines[2:]
    profile_rows = []
    for line in data_lines:
        line = line.strip()
        if line:  # ignore blank lines
            values = list(map(float, line.split()))
            if len(values) != 20:
                print(f"Warning: expected 20 values per line but got {len(values)}. Skipping line: {line}")
                continue
            profile_rows.append(values)
    profile_matrix = np.array(profile_rows)
    return profile_matrix


def parse_profiles(filename):
    with open(filename, "r") as f:
        lines = f.read()

    return csv_to_profile(lines)
