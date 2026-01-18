import io

import matplotlib.pyplot as plt
import MDAnalysis as mda
import nglview as nv
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from MDAnalysis.analysis import align
from MDAnalysis.coordinates.memory import MemoryReader
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from protprofilemd.utils.definitions import STRUCTURE_ALPHABET


def shannon_entropy_bits(P: np.ndarray, eps: float = 1e-12, alphabet_size: int = 20, correct_ends: bool = False) -> np.ndarray:
    """
    P: (L, A) row-stochastic (rows sum to 1). Non-negative entries. L is the length. A is the Alphabet size.
    Returns: H per row in bits, shape (L,)
    """
    if P.ndim != 2:
        raise ValueError("P must be 2D")
    if P.shape[1] != alphabet_size:
        raise ValueError(f"P must have exactly {alphabet_size} rows. It has {P.shape[0]}.")

    P = np.clip(P, eps, 1.0)
    H = -(P * np.log2(P))
    H = H.sum(axis=1).round(5)
    
    if correct_ends:
        H[0] = H[1]
        H[-1] = H[-2]

    return H


def processed_entropy(H: np.ndarray, method: str = None, normalize: bool = False) -> np.ndarray:
    if method == "savgol":
        entropy_values_processed = savgol_filter(H, window_length=20, polyorder=3)
    elif method == "spline":
        entropy_values_processed = UnivariateSpline(np.arange(len(H)), H, s=len(H)/4)(np.arange(len(H)))
    elif method == "gaussian":
        entropy_values_processed = gaussian_filter1d(H, sigma=2)
    elif method is None:
        entropy_values_processed = H

    if normalize:
        entropy_values_processed = entropy_values_processed / np.max(entropy_values_processed)

    return entropy_values_processed


def color_by_entropy(pdb_string: str, entropy: np.ndarray, traj_coords: np.ndarray = None, domain: str = None) -> None:
    
    u = mda.Universe(io.StringIO(pdb_string), format="PDB")
    # chain_not_b = u.select_atoms("not segid B")
    # u = mda.Merge(chain_not_b)
    
    if domain:
        selection = f"segid {domain} and protein"
    else:
        selection = "protein"
        
    base_protein = u.select_atoms(selection)
    from MDAnalysis.lib.util import convert_aa_code

    seq = "".join([convert_aa_code(res.resname) for res in base_protein.residues])
    print(seq)
    
    # sequence = ''.join([aa_3to1.get(res.resname, 'X') for res in base_protein.residues])
    # print(f"Protein sequence ({len(base_protein.residues)} residues): {sequence}")
    
    print(len(base_protein.residues), len(entropy))
    assert len(base_protein.residues) == len(entropy), "Length of protein residues and entropy must be the same"

    for res, H in zip(base_protein.residues, entropy, strict=True):
        for atom in res.atoms:
            atom.tempfactor = float(H)

    if traj_coords is not None:
        u.load_new(traj_coords, format=MemoryReader)
        aligner = align.AlignTraj(u, u.select_atoms("protein"), select="protein", in_memory=True)
        aligner.run()

    return base_protein


def show_entropy_colored_protein(u, H_min_max: tuple = None, standard_color_scale: bool = False, width="640px", height="512px"):
    """
    Display an NGLView of a protein colored by entropy (bfactor).

    Parameters:
        u : MDAnalysis Universe
            Universe to visualize.
        H_min : float
            Minimum value of color domain.
        H_max : float or None
            Maximum value of color domain. If None, set to log2(20).
        width : str
            Width of the view (e.g., "1000px").
        height : str
            Height of the view (e.g., "600px").

    Returns:
        view : nglview.NGLWidget
            Configured NGL widget for display.
    """
    import math

    if standard_color_scale:
        H_min_max = (0.0, math.log2(20))

    view = nv.show_mdanalysis(u)
    view.clear_representations()
    view.add_cartoon(
        color="bfactor",
        colorScheme="bfactor",
        colorDomain=H_min_max,
        colorScale=['#ca0020', '#f4a582', '#f7f7f7', '#92c5de', '#0571b0'][::-1],
        # colorScale="RdBu",
    )
    view._set_size(width, height)
    return view


def plot_shannon_entropy(entropy_values, entropy_values_processed, figsize=(10, 4)):
    """
    Plot Shannon entropy and processed entropy values per position.

    Parameters:
        entropy_values: np.ndarray
            Raw entropy values.
        entropy_values_processed: np.ndarray
            Processed entropy values.
        figsize: tuple
            Figure size.
    """
    plt.figure(figsize=figsize)
    plt.plot(entropy_values, marker="", linestyle="-", label="Raw Entropy")
    plt.plot(entropy_values_processed, marker="", linestyle="--", label="Processed Entropy")
    plt.title("Shannon Entropy (bits) per Position")
    plt.xlabel("Position")
    plt.ylabel("Entropy (bits)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_profile_heatmap_labels(
    name: str,
    profile: np.ndarray,
    entropy: np.ndarray = None,
    save_path: str = None,
    font_size: int = 16,
    font_family: str = None,
    annotation: list = None,
    label_name: str = "Labeled Residues",
    show=True,
):
    """
    Plots a PSSM heatmap with an optional binary color bar below and entropy line plot.

    Args:
        name (str): Protein identifier.
        profile (np.ndarray): Profile array of shape (L, A).
        entropy (np.ndarray): Pre-computed entropy values of shape (L,).
        save_path (str, optional): Path to save the plot. If None, does not save.
        font_size (int, optional): Font size for labels.
        font_family (str, optional): Font family for labels.
        label (list[bool], optional): Binary mask for color bar (0: black, 1: orange).
        label_name (str, optional): Label name for the binary color bar.
        show (bool, optional): Whether to display the plot.
    """
    structure_alphabet = STRUCTURE_ALPHABET
    viridis = plt.cm.get_cmap("viridis", 256)
    colors = viridis(np.linspace(0, 1, 256))
    colors[0] = [0, 0, 0, 1]
    custom_cmap = ListedColormap(colors)

    # Create a gridspec with 4 rows: heatmap, mask bar, entropy, residue ticks; and 2 columns: main, colorbar
    fig = plt.figure(figsize=(12, 11))
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(4, 2, width_ratios=[20, 1], height_ratios=[20, 1, 3, 1], wspace=0.05, hspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_entropy = fig.add_subplot(gs[2, 0], sharex=ax)
    ax_ticks = fig.add_subplot(gs[3, 0], sharex=ax)
    cbar_ax = fig.add_subplot(gs[:, 1])
    # Increase vertical spacing between plots

    # Plot heatmap with colorbar in right axis
    sns.heatmap(
        profile.T,
        cmap=custom_cmap,
        vmin=0,
        vmax=1,
        cbar=True,
        ax=ax,
        cbar_ax=cbar_ax,
        linewidths=0,
        linecolor=None,
    )
    ax.set_title(
        f"PSSM Heatmap predicted with ProtPSSM for PDB {name} (len={profile.shape[0]})", fontfamily=font_family, fontsize=font_size
    )
    ax.set_xticks([])

    if structure_alphabet is not None:
        ax.set_yticks(np.arange(len(structure_alphabet)) + 0.5)
        yticklabels = []
        for i, label in enumerate(structure_alphabet):
            if i % 2 == 1:
                yticklabels.append(label + "  ")
            else:
                yticklabels.append(label)
        ax.set_yticklabels(yticklabels, rotation=0, fontfamily=font_family, fontsize=font_size)

    # plt.suptitle(f"PSSM Heatmap", fontfamily=font_family, fontsize=font_size)

    # Add binary color bar below heatmap only
    if annotation is not None:
        L = profile.shape[0]
        if annotation.shape[0] != L:
            raise ValueError(f"Mask length ({annotation.shape[0]}) does not match PSSM length ({L})")

        # Ensure annotation is integer type and only contains 0 and 1
        annotation_arr = np.asarray(annotation).astype(int)
        if not np.all(np.isin(annotation_arr, [0, 1])):
            raise ValueError("Annotation array must be binary (0 or 1)")

        color_map = np.array([[0, 0, 0], [1, 0.5, 0]])  # black, orange
        # Map each position according to its value in annotation_arr
        bar_colors = color_map[annotation_arr]
        ax_bar.imshow(bar_colors[np.newaxis, :, :], aspect="auto")
        # Add label above the binary color bar
        ax_bar.text(
            0.5,
            1.2,
            label_name,
            ha="center",
            va="bottom",
            transform=ax_bar.transAxes,
            fontsize=font_size,
            fontfamily=font_family,
        )
        ax_bar.axis("off")

    # Plot residue position ticks as a separate axis
    L = profile.shape[0]
    
    # Always get 11 labels, including first and last
    tick_positions = np.linspace(0, L - 1, L, dtype=int) if L < 11 else np.linspace(0, L - 1, 11, dtype=int)
    tick_labels = [str(pos + 1) for pos in tick_positions]
    ax_ticks.set_xlim(0, L - 1)
    ax_ticks.set_xticks(tick_positions)
    ax_ticks.set_xticklabels(tick_labels, fontsize=font_size - 4, fontfamily=font_family)
    ax_ticks.set_yticks([])
    ax_ticks.set_xlabel("Residue Position", fontsize=font_size - 2, fontfamily=font_family)
    ax_ticks.tick_params(axis="x", length=6)

    # Add mini ticks for each other tick that is not a multiple of 10
    # mini_tick_positions = list(range(L))
    # ax_ticks.set_xticks(mini_tick_positions, minor=True)
    # ax_ticks.tick_params(axis="x", which="minor", length=2)

    # ax_ticks.spines["top"].set_visible(False)
    # ax_ticks.spines["right"].set_visible(False)
    # ax_ticks.spines["left"].set_visible(False)

    # Plot entropy line below mask bar
    if entropy is not None:
        if not isinstance(entropy, list):
            entropy = [entropy]
        for entropy_value in entropy:
            # Plot each entropy line with a different color
            colors = [
                "tab:blue", "tab:orange", "tab:green", "tab:red",
                "tab:purple", "tab:brown", "tab:pink", "tab:gray",
                "tab:olive", "tab:cyan"
            ]
            color_idx = 0
            for entropy_idx, entropy_value in enumerate(entropy):
                color = colors[entropy_idx % len(colors)]
                ax_entropy.plot(
                    np.arange(len(entropy_value)),
                    entropy_value,
                    color=color,
                    label=f"Entropy {entropy_idx + 1}" if len(entropy) > 1 else None,
                )
            ax_entropy.set_title("Shannon Entropy per Position", fontfamily=font_family, fontsize=font_size)
            ax_entropy.set_ylabel("Entropy (bits)")
            ax_entropy.set_xlabel("Position")
            ax_entropy.set_xlim(0, len(entropy[0]) - 1)
            ax_entropy.grid(True, linestyle="--", alpha=0.3)
            ax_entropy.xaxis.set_visible(False)
            
            # ax_entropy.set_ylim(0, np.log2(20))
            
            # Add legend if more than one entropy line
            if len(entropy) > 1:
                ax_entropy.legend(fontsize=font_size-4, frameon=False)
            legend = ax_entropy.get_legend()
            if legend is not None:
                legend.set_visible(False)

    # Hide x-axis for ax_bar and ax
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax_bar.get_xticklabels(), visible=False)
    plt.setp(ax.get_xticklines(), visible=False)
    plt.setp(ax_bar.get_xticklines(), visible=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight")

    if show:
        plt.show()
