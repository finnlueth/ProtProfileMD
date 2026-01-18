import os
import shlex
import subprocess
import tempfile
import typing as T

import numpy as np
import pandas as pd


def profile_from_aligned_3Di_strings(msa: T.List[str]) -> np.ndarray:
    unique_chars, inverse = np.unique(msa, return_inverse=True)

    one_hot = np.eye(len(unique_chars), dtype=float)[inverse]
    profile_ppm = one_hot.mean(axis=0).T
    
    return profile_ppm


def profile_from_trajectory_pdbs_foldseek(trajectory_pdbs: T.List[str]) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "inputs")
        os.makedirs(input_dir)

        for i, pdb in enumerate(trajectory_pdbs):
            pdb_path = os.path.join(input_dir, f"structure_{i+1:05d}.pdb")
            with open(pdb_path, "w") as f:
                f.write(pdb)

        subprocess.run(
            ["foldseek", "createdb", input_dir, os.path.join(tmpdir, "inputdb")],
            check=True,
            stdout=subprocess.DEVNULL,
            # stderr=subprocess.STDOUT,
        )

        with open(os.path.join(tmpdir, "inputdb.index")) as f:
            index_data = f.readlines()

        # with open(os.path.join(tmpdir, "fake_aln.tsv"), "w") as f:
        #     for line in index_data:
        #         parts = line.split()
        #         length = int(parts[2]) - 2
        #         f.write(f"0\t{parts[0]}\t{length*4}\t1.00\t0\t0\t{length-1}\t{length}\t0\t{length-1}\t{length}\t{length}M\n")
        
        
        with open(os.path.join(tmpdir, "fake_aln.tsv"), "w") as f:
            subprocess.run(
                ["awk", 
                """{ len = $3 - 2; print "0\t"$1"\t0\t1.00\t0\t0\t"(len-1)"\t"len"\t0\t"(len-1)"\t"len"\t"len"M"; }""",
                os.path.join(tmpdir, "inputdb.index")],
                stdout=f,
                check=True,
            )

        subprocess.run(
            [
                "foldseek",
                "tsv2db",
                os.path.join(tmpdir, "fake_aln.tsv"),
                os.path.join(tmpdir, "fake_aln_db"),
                "--output-dbtype",
                "5",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            # stderr=subprocess.STDOUT,
        )

        victors_params = [
            "--pca",
            "0",
            "--pcb",
            "0",
            "--profile-output-mode",
            "1",
            # "--sub-mat",
            # "/home/finnlueth/repos/bachelor-thesis/prot-md-pssm-legacy/submodules/foldseek/data/mat3di.out",
            "--mask-profile",
            "0",
            "--comp-bias-corr",
            "0",
            "--e-profile",
            "inf",
            "-e",
            "inf",
            "--gap-open",
            "11",
            "--gap-extend",
            "1",
        ]

        subprocess.run(
            [
                "mmseqs",
                "result2profile",
                os.path.join(tmpdir, "inputdb_ss"),
                os.path.join(tmpdir, "inputdb_ss"),
                os.path.join(tmpdir, "fake_aln_db"),
                os.path.join(tmpdir, "profile.tsv"),
            ]
            + victors_params,
            check=True,
            stdout=subprocess.DEVNULL,
            # stderr=subprocess.STDOUT,
        )

        profile_data = pd.read_csv(os.path.join(tmpdir, "profile.tsv"), sep=" ", header=None, skiprows=2)
        profile_data = profile_data.iloc[:, :-1]

        return profile_data.to_numpy()




def get_3di_sequences_from_memory(pdb_files: T.List[str]):
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_paths = []
        for i, content in enumerate(pdb_files):
            pdb_path = os.path.join(tmpdir, f"file_{i:05d}.pdb")
            with open(pdb_path, "w") as file:
                file.write(content)
            pdb_paths.append(pdb_path)

        pdb_file_string = " ".join(pdb_paths)
        pdb_dir_name = hash(pdb_file_string)
        db_name = f"{tmpdir}/{pdb_dir_name}"

        FSEEK_BASE_CMD = f"foldseek createdb {pdb_file_string} {db_name}"
        proc = subprocess.Popen(shlex.split(FSEEK_BASE_CMD), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()

        seq_file_path = f"{db_name}_ss"
        lookup_file_path = f"{db_name}.lookup"

        if os.path.exists(seq_file_path):
            with open(seq_file_path, "r") as seq_file:
                seqs = [line.strip().strip("\x00") for line in seq_file]
                seqs.remove("")
        else:
            raise FileNotFoundError(f"No sequence file found at {seq_file_path}")

        if os.path.exists(lookup_file_path):
            with open(lookup_file_path, "r") as name_file:
                names = [line.strip().split()[1].split(".")[0] for line in name_file]
        else:
            raise FileNotFoundError(f"No lookup file found at {lookup_file_path}")
        # return names, seqs
        return seqs