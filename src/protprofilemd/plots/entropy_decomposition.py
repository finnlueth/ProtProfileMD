import numpy as np
from typing import Tuple

def compute_native_contact_pairs(
    coords_ref: np.ndarray,
    cutoff: float = 8.0,
    min_seq_sep: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute indices of native residue-residue contacts from a reference structure.

    Parameters
    ----------
    coords_ref : (L, 3) array
        Cartesian coordinates for L residues (e.g. Cα positions).
    cutoff : float
        Distance cutoff (Å) for defining a native contact.
    min_seq_sep : int
        Minimum sequence separation |i - j| to consider (to ignore trivial neighbors).

    Returns
    -------
    i_idx, j_idx : 1D int arrays
        Arrays of residue indices defining contact pairs (i_idx[k], j_idx[k]).
    """
    coords_ref = np.asarray(coords_ref, dtype=np.float32)

    diff = coords_ref[:, None, :] - coords_ref[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)

    i_idx, j_idx = np.where(dists < cutoff)
    mask = (np.abs(i_idx - j_idx) >= min_seq_sep) & (i_idx < j_idx)

    return i_idx[mask], j_idx[mask]


def compute_Q_for_trajectory(
    traj_coords: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    cutoff: float = 8.0,
) -> np.ndarray:
    """
    Compute native contact fraction Q(t) for each frame of a trajectory.

    Parameters
    ----------
    traj_coords : (T, L, 3) array
        Cartesian coordinates for T frames, L residues.
    i_idx, j_idx : 1D int arrays
        Contact pairs from compute_native_contact_pairs.
    cutoff : float
        Distance cutoff (Å) used for contact definition.

    Returns
    -------
    Q : (T,) array
        Fraction of native contacts present in each frame.
    """
    traj_coords = np.asarray(traj_coords, dtype=np.float32)
    T = traj_coords.shape[0]

    if len(i_idx) == 0:
        raise ValueError("No native contacts found; check cutoff or input coordinates.")

    # Extract contact pair coordinates for all frames
    # shape: (T, n_contacts, 3)
    ci = traj_coords[:, i_idx, :]
    cj = traj_coords[:, j_idx, :]
    dists = np.linalg.norm(ci - cj, axis=-1)  # (T, n_contacts)

    present = dists < cutoff
    Q = present.sum(axis=1) / present.shape[1]

    return Q


def choose_Q_min_by_quantile(
    Q_values_320K: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    Choose Q_min as a lower quantile of the 320K Q distribution.

    Parameters
    ----------
    Q_values_320K : (T_320,) array
        Q(t) values for 320K trajectories pooled across all domains/replicas.
    alpha : float
        Lower tail probability. alpha=0.05 keeps 95% of 320K frames as native-like.

    Returns
    -------
    Q_min : float
        Native-basin threshold on Q.
    """
    Q_values_320K = np.asarray(Q_values_320K, dtype=np.float32)
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha should be in (0, 0.5)")

    Q_min = float(np.quantile(Q_values_320K, alpha))
    return Q_min
