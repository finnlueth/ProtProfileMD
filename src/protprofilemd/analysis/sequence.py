"""Functions for sequence identity calculations using Biotite."""

from typing import List, Tuple, Optional
import biotite.sequence as seq
import biotite.sequence.align as align


def _align_sequences(
    seq1: str, seq2: str, matrix: Optional[align.SubstitutionMatrix], gap_penalty: int, local: bool
) -> align.Alignment:
    """
    Perform optimal sequence alignment.

    Returns:
        Best alignment object

    Raises:
        ValueError: If sequences are empty or alignment fails
    """
    if not seq1 or not seq2:
        raise ValueError("Both sequences must be non-empty")

    biotite_seq1 = seq.ProteinSequence(seq1)
    biotite_seq2 = seq.ProteinSequence(seq2)
    matrix = matrix or align.SubstitutionMatrix.std_protein_matrix()

    alignments = align.align_optimal(biotite_seq1, biotite_seq2, matrix, gap_penalty=gap_penalty, local=local)

    if not alignments:
        raise ValueError("No alignment could be generated")

    return alignments[0]


def calculate_pairwise_identity(
    seq1: str, seq2: str, matrix: Optional[align.SubstitutionMatrix] = None, gap_penalty: int = -10, local: bool = False
) -> float:
    """
    Calculate percent identity between two sequences using Biotite alignment.

    Args:
        seq1: First sequence string
        seq2: Second sequence string
        matrix: Substitution matrix for alignment. If None, uses BLOSUM62
        gap_penalty: Gap penalty for alignment (default: -10)
        local: If True, perform local alignment; if False, global alignment

    Returns:
        Percent identity (0-100)
    """
    if not seq1 or not seq2:
        return 0.0

    alignment = _align_sequences(seq1, seq2, matrix, gap_penalty, local)
    return align.get_sequence_identity(alignment) * 100.0


def max_sequence_identity(
    sequences_set1: List[str],
    sequences_set2: List[str],
    matrix: Optional[align.SubstitutionMatrix] = None,
    gap_penalty: int = -10,
    local: bool = False,
    return_details: bool = False,
) -> float | Tuple[float, int, int]:
    """
    Calculate the maximum percent sequence identity between two sets of sequences.

    This function performs pairwise alignments between all combinations of sequences
    from set1 and set2 using Biotite, and returns the maximum identity found.

    Args:
        sequences_set1: First set of sequences (strings)
        sequences_set2: Second set of sequences (strings)
        matrix: Substitution matrix for alignment. If None, uses BLOSUM62
        gap_penalty: Gap penalty for alignment (default: -10)
        local: If True, perform local alignment; if False, global alignment
        return_details: If True, return tuple of (max_identity, idx1, idx2)

    Returns:
        If return_details is False: Maximum percent identity (0-100)
        If return_details is True: Tuple of (max_identity, idx1, idx2) where
            idx1 and idx2 are the indices of the sequences with max identity

    Raises:
        ValueError: If either set is empty

    Examples:
        >>> set1 = ["ACDEFGHIKLMNPQRSTVWY", "TGCAEFGHIKLMNPQRSTVWY"]
        >>> set2 = ["ACDEFGHIKLMNPQRSTVWY", "AAAAAAAAAAAAAAAAAAAA"]
        >>> max_sequence_identity(set1, set2)
        100.0

        >>> max_sequence_identity(set1, set2, return_details=True)
        (100.0, 0, 0)
    """
    if not sequences_set1 or not sequences_set2:
        raise ValueError("Both sequence sets must contain at least one sequence")

    max_identity = 0.0
    max_seq1, max_seq2 = None, None

    for i, s1 in enumerate(sequences_set1):
        for j, s2 in enumerate(sequences_set2):
            try:
                identity = calculate_pairwise_identity(s1, s2, matrix, gap_penalty, local)
                if identity > max_identity:
                    max_identity = identity
                    max_seq1, max_seq2 = s1, s2
            except Exception as e:
                raise ValueError(f"Error comparing sequence {i} from set1 with sequence {j} from set2: {e}") from e

    return (max_identity, max_seq1, max_seq2) if return_details else max_identity
