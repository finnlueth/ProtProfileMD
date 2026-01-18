import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

from protprofilemd.utils.definitions import STRUCTURE_ALPHABET


def profile_multi_heatmap(
    profiles: dict[str, np.ndarray], name: str = None, sequence: str = None, title: str = None, save_path: str = None
) -> plt.Figure:
    font_size = 6
    # font_family = "sans-serif"

    viridis = plt.cm.get_cmap("viridis", 256)
    colors = viridis(np.linspace(0, 1, 256))
    colors[0] = [0, 0, 0, 1]  # Set the first color (for 0) to black
    custom_cmap = ListedColormap(colors)

    # Make tick and line widths thinner
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.5
    plt.rcParams["axes.linewidth"] = 0.5

    n_profiles = len(profiles)
    fig, axes = plt.subplots(n_profiles, 1, figsize=(3.392, 1.1 * n_profiles), squeeze=False)

    for i, ((name, profile), ax) in enumerate(zip(profiles.items(), axes[:, 0])):
        ax.set_title(name, fontsize=font_size, y=0.935)
        sns.heatmap(
            profile.T,
            cmap=custom_cmap,
            vmin=0,
            vmax=1,
            # cbar_kws={"label": "Probability"},
            # linewidths=0.5,
            linewidths=0.08,
            linecolor="black",
            cbar=False,
            ax=ax,
        )
        if i < n_profiles - 1:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_xticks(np.arange(profile.shape[0]) + 0.5)

            # num_xticks = 8
            # x_locs = np.linspace(0, profile.shape[0], num_xticks, endpoint=False) + 0.5
            # x_labels = [str(int(i)) for i in np.linspace(0, profile.shape[0] - 1, num_xticks, endpoint=True)]

            x_locs = np.arange(0, profile.shape[0], 10) + 0.5
            x_labels = [str(int(i)) for i in np.arange(0, profile.shape[0], 10)]
            ax.set_xticks(np.arange(0, profile.shape[0], 5) + 0.5, minor=True)

            ax.set_xticks(x_locs)
            ax.set_xticklabels(x_labels, rotation=0, fontsize=font_size)

        yticklabels = []
        for i, label in enumerate(STRUCTURE_ALPHABET):
            if i % 2 == 1:
                yticklabels.append(label + "   ")
            else:
                yticklabels.append(label)

        for ax in axes[:, 0]:
            n_labels = len(yticklabels)
            ax.set_yticks(np.arange(n_labels) + 0.5)
            ax.set_yticklabels(yticklabels, rotation=0, fontsize=font_size)

    if title is not None:
        fig.suptitle(title, fontsize=font_size + 2, y=0.99)

    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])  # [left, bottom, width, height] (moved to right)
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    # cbar.set_label('Probability', fontsize=font_size, labelpad=-6)
    cbar.ax.tick_params(labelsize=font_size)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(size=0)  # remove tick lines from colorbar

    top_margin = 1.07 if title is not None else 1
    plt.tight_layout(rect=[-0.04, -0.04, 1.1, top_margin], h_pad=0.5)

    fig.subplots_adjust(right=0.88, hspace=0.15)

    if save_path is not None:
        format = save_path.split(".")[-1]
        plt.savefig(save_path, format=format)

    plt.figure()
    plt.show()
