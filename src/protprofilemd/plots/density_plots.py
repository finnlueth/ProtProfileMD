import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

def plot_entropy_density(
    scop_avg_entropy,
    fs_avg_entropy,
    disprot_avg_entropy,
    save_path="../tmp/plots/fs_disorder_scope_entropy_density.svg"
):
    """
    Plots kernel density estimates of average Shannon entropy for three datasets.

    Args:
        fs_avg_entropy: Dict of Fold Switching protein avg entropies.
        scop_avg_entropy: Dict of SCOPe40 protein avg entropies.
        disprot_avg_entropy: Dict of DisProt protein avg entropies.
        save_path: Path to save the resulting svg plot.
    """

    fs_entropy_values = np.array(list(fs_avg_entropy.values()))
    scop_entropy_values = np.array(list(scop_avg_entropy.values()))
    disprot_entropy_values = np.array(list(disprot_avg_entropy.values()))

    group_labels = [
        ("Fold Switching", fs_entropy_values, "orange"),
        ("SCOPe40", scop_entropy_values, "blue"),
        ("DisProt Assessment", disprot_entropy_values, "purple"),
    ]

    fig, ax = plt.subplots(figsize=(4, 1.9))

    for spine in ax.spines.values():
        spine.set_linewidth(0.5) 

    ax.tick_params(width=0.5)

    for label, data, color in group_labels:
        if len(data) > 1:
            sns.kdeplot(
                data,
                ax=ax,
                fill=True,
                label=label,
                color=color,
                alpha=0.2,
                linewidth=1,
            )

    ax.set_xlabel("Average Shannon Entropy", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=7, title_fontsize=8)

    plt.tight_layout()
    
    if save_path is not None:
        format = save_path.split(".")[-1]
        plt.savefig(save_path, format=format, bbox_inches="tight")
    plt.show()

    return fig, ax