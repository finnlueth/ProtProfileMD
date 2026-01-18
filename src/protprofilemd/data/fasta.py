def fasta_to_dict(fasta: str):
    sequences = {}
    current_sequence = None

    for entry in fasta.split(">"):
        if entry:
            lines = entry.strip().split("\n")
            current_sequence = lines[0]
            sequence = "".join(lines[1:])
            sequences[current_sequence] = sequence
    return sequences


def dict_to_fasta(sequences: dict):
    fasta = []
    for header, sequence in sequences.items():
        fasta.append(f">{header}\n{sequence}\n")
    return "".join(fasta)


def multiline_fasta_to_dict(fasta: str):
    sequences = {}
    current_sequence = None

    for entry in fasta.split(">"):
        if entry:
            lines = entry.strip().split("\n")
            current_sequence = lines[0]
            sequence = "".join(lines[1:])
            if current_sequence in sequences:
                sequences[current_sequence] += sequence
            else:
                sequences[current_sequence] = sequence
    return sequences


def multisequence_fasta_to_dict(fasta: str):
    sequences = {}
    current_sequence = None

    for entry in fasta.split(">"):
        if entry:
            lines = entry.strip().split("\n")
            current_sequence = lines[0]
            sequence = lines[1:]
            sequences[current_sequence] = sequence
    return sequences

