import h5py
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

from protprofilemd.model.protein_tokenizer import ProteinTokenizer


def parse_cath_families(cath_families_lookup_file_path: str) -> pd.DataFrame:
    cath_families = pd.read_csv(
        cath_families_lookup_file_path,
        sep=r"\s+",
        header=None,
        names=[
            "domain_name",  # Column 1: CATH domain name (7 chars)
            "class_num",  # Column 2: Class number
            "arch_num",  # Column 3: Architecture number
            "topo_num",  # Column 4: Topology number
            "homsf_num",  # Column 5: Homologous superfamily number
            "s35_num",  # Column 6: S35 sequence cluster number
            "s60_num",  # Column 7: S60 sequence cluster number
            "s95_num",  # Column 8: S95 sequence cluster number
            "s100_num",  # Column 9: S100 sequence cluster number
            "s100_count",  # Column 10: S100 sequence count number
            "domain_length",  # Column 11: Domain length
            "resolution",  # Column 12: Structure resolution (Angstroms)
        ],
    )
    return cath_families


def parse_cath_target_domains(cath_domains_file_path: str) -> pd.DataFrame:
    return pd.read_csv(cath_domains_file_path, header=None, names=["domain_name"])


def segmentation_by_hierarchy(
    cath_families_lookup: pd.DataFrame,
    cath_target_domains: pd.DataFrame,
    test_size: int,
    split_by_hierarchy: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    merged_cath_lookup_domains = cath_target_domains.merge(cath_families_lookup, on="domain_name", how="left")

    hierarchy_counts = merged_cath_lookup_domains[split_by_hierarchy].value_counts()
    non_singleton_non_large_hierarchies = hierarchy_counts[(hierarchy_counts > 1) & (hierarchy_counts < 30)].index

    print(f"{split_by_hierarchy} hierarchy unique values: {len(set(merged_cath_lookup_domains[split_by_hierarchy]))}")
    print(f"{split_by_hierarchy} non-singleton non-large hierarchies: {len(non_singleton_non_large_hierarchies)}")

    test_split_hierarchies = rng.choice(
        non_singleton_non_large_hierarchies, size=min(test_size, len(non_singleton_non_large_hierarchies)), replace=False
    ).tolist()
    train_split_hierarchies = [
        hierarchy for hierarchy in non_singleton_non_large_hierarchies if hierarchy not in test_split_hierarchies
    ]

    test_indices = merged_cath_lookup_domains[
        merged_cath_lookup_domains[split_by_hierarchy].isin(test_split_hierarchies)
    ].index.tolist()
    train_indices = list(set(range(len(merged_cath_lookup_domains))) - set(test_indices))
    print(f"Test set Size: {len(test_indices)} (Fraction: {len(test_indices) / len(merged_cath_lookup_domains):.4f})")
    print(f"Train set size: {len(train_indices)} (Fraction: {len(train_indices) / len(merged_cath_lookup_domains):.4f})")

    stats = {
        "test_size": len(test_indices),
        "test_fraction": len(test_indices) / len(merged_cath_lookup_domains),
        "train_size": len(train_indices),
        "train_fraction": len(train_indices) / len(merged_cath_lookup_domains),
    }

    test_set_domains = merged_cath_lookup_domains.iloc[test_indices]
    train_set_domains = merged_cath_lookup_domains.iloc[train_indices]

    return test_split_hierarchies, test_set_domains, train_split_hierarchies, train_set_domains, stats


def h5_profiles_to_hf_dataset_generator(
    h5_profiles_path: str,
    tokenizer_prostT5: ProteinTokenizer,
    tokenizer_protT5: ProteinTokenizer,
    **kwargs,
):
    with h5py.File(h5_profiles_path, "r") as f:
        for protein_id in f:
            for trajectory_profile in f[protein_id]:
                tokenized_prostT5 = tokenizer_prostT5.protein_encode(
                    text=f[protein_id].attrs["sequence"], padding=False, truncation=False
                )

                tokenized_protT5 = tokenizer_protT5.protein_encode(
                    text=f[protein_id].attrs["sequence"], padding=False, truncation=False
                )

                yield {
                    "domain_name": f[protein_id].attrs["name"],
                    "temperature": trajectory_profile.split("_")[0],
                    "replica": trajectory_profile.split("_")[1],
                    "aa_sequence": f[protein_id].attrs["sequence"],
                    "input_ids_prostT5": tokenized_prostT5["input_ids"][0],
                    "attention_mask_prostT5": tokenized_prostT5["attention_mask"][0],
                    "input_ids_protT5": tokenized_protT5["input_ids"][0],
                    "attention_mask_protT5": tokenized_protT5["attention_mask"][0],
                    "profiles": f[protein_id][trajectory_profile][:],
                }


def h5_profiles_to_hf_dataset(
    h5_profiles_path: str,
    tokenizer_prostT5: ProteinTokenizer,
    tokenizer_protT5: ProteinTokenizer,
) -> DatasetDict:
    # import tempfile
    import time
    
    ds = Dataset.from_generator(
        h5_profiles_to_hf_dataset_generator,
        gen_kwargs={
            "h5_profiles_path": h5_profiles_path,
            "tokenizer_prostT5": tokenizer_prostT5,
            "tokenizer_protT5": tokenizer_protT5,
        },
        split="full",
        # cache_dir=tempfile.mkdtemp()
        fingerprint=str(time.time()),
    )
    return ds


def split_by_keys(ds: Dataset, key_col: str, train_keys: list[str], test_keys: list[str]) -> DatasetDict:
    train_set = set(train_keys)
    test_set = set(test_keys)

    overlap = train_set & test_set
    if overlap:
        raise ValueError(f"Keys in both splits: {len(overlap)} examples")

    col = ds[key_col]

    train_idx = [i for i, v in enumerate(col) if v in train_set]
    test_idx = [i for i, v in enumerate(col) if v in test_set]

    if set(train_idx) & set(test_idx):
        raise AssertionError("Index overlap between train and test after masking")

    return DatasetDict(
        train=ds.select(train_idx),
        test=ds.select(test_idx),
    )
