# ProtProfileMD

Manuscript: Coming soon.

## Abstract

**Motivation:** Proteins function through motion. Yet, most discoveries still commence with static representations of protein structures.
Here, we investigated the feasibility of leveraging protein dynamics to improve homology detection.

**Results:** We introduce ProtProfileMD, a sequence-to-3D-probability model that predicts, from an amino acid sequence, a profile of discrete structural representations capturing protein dynamics.
We applied supervised parameter-efficient finetuning of the ProstT5 protein Language Model (pLM) to predict per-residue distributions over Foldseek's 3Di alphabet derived from motions observed in molecular dynamics.
This original result reveals that the 3Di tokens, despite being coarse-grained descriptors of 3D structure, still offer sufficient resolution to capture aspects of conformational changes. This is evidenced by a correlation between fluctuations in the 3D protein structure over the course of a molecular dynamics trajectory and the entropy of 3Di states.
Based on this insight, we introduce a proof-of-concept for making remote homology detection of proteins more sensitive by leveraging a protein's distinctive dynamic fingerprint captured by our model.
Our method recovers flexibility signals with a fidelity that is biologically relevant, improving search and complementing protein structure predictions, for example, by flagging flexible, disordered, or other functionally relevant regions.

**Availability and Implementation:** ProtProfileMD is available at github.com/finnlueth/ProtProfileMD.
The associated training data and model weights are available at https://huggingface.co/datasets/finnlueth/ProtProfileMD and https://huggingface.co/finnlueth/ProtProfileMD.

## Setup and Use

Initialize the virtual environment with all dependencies. UV is required for installatio.

```sh
uv sync

source .venv/bin/activate
```

Predict _FlexProfiles_ for any single line FASTA file. Path to input FASTA file, and output TSV file are required. Optionally set batch size, and resume from (append to) existing tsv file.

```sh
python ./scripts/model_inference.py --input ./example/input.fasta --output ./example/output.tsv --resume_from_tsv True --batch_size 8
```
