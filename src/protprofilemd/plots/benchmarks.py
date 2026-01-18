import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_scope_benchmarks(file_paths: dict[str, str], add_aurocs=False, save=False, impaired_symbols=False):
    """
    Plots SCOPe benchmark sensitivity up to the first false positive for multiple .rocx files.

    Parameters:
        file_paths (dict): Dictionary mapping display names to .rocx file paths.
    """
    plt.rcParams["figure.dpi"] = 300
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6), gridspec_kw={"width_ratios": [1, 1, 1]}, sharey=True)

    titles = ["Family", "Superfamily", "Fold"]
    y_labels = ["FAM", "SFAM", "FOLD"]
    
    font_size = 13

    colors = [
        "black",
        "blue",
        "red",
        "orange",
        "green",
        "purple",
        "gray",
        "olive",
        "cyan",
        "magenta",
        "lime",
        "teal",
        "navy",
        "maroon",
        "gold",
        "silver",
        "indigo",
        "turquoise",
        "coral",
        "khaki",
        "lime",
        "teal",
        "navy",
        "maroon",
        "gold",
        "silver",
        "indigo",
        "turquoise",
        "coral",
        "khaki",
    ]

    markers = [
        "o",  # circle
        "s",  # square
        "^",  # triangle_up
        "p",  # pentagon
        "v",  # triangle_down
        "D",  # diamond
        "<",  # triangle_left
        ">",  # triangle_right
        "*",  # star
        "h",  # hexagon1
        "H",  # hexagon2
        "+",  # plus
        "x",  # x
        "d",  # thin_diamond
        "|",  # vline
        "_",  # hline
        ".",  # point
        "1",  # tri_down
        "2",  # tri_up
        "3",  # tri_left
        "4",  # tri_right
        "8",  # octagon
        "P",  # plus_filled
        "X",  # x_filled
    ]

    for idx, (name, path) in enumerate(file_paths.items()):
        df = pd.read_csv(path, sep="\t")
        fraction_queries = np.linspace(0, 1, len(df))

        aucs = []
        for i, ylabel in enumerate(y_labels):
            df_sorted = df.sort_values(by=ylabel, ascending=False)

            auc = df_sorted[ylabel].mean()
            aucs.append(auc)

            plot_kwargs = {
                "label": f"{name}",
                "color": colors[idx % len(colors)],
                "linewidth": 1.6,
                "linestyle": "-",
            }
            
            if impaired_symbols:
                # Only show markers where y < 1
                mask = (df_sorted[ylabel] < 1) & (df_sorted[ylabel] > 0)
                if mask.any():
                    # Thin markers: show every Nth point (similar to original markevery logic)
                    filtered_indices = np.where(mask)[0]
                    markevery = max(1, len(df_sorted) // 20)
                    thinned_indices = filtered_indices[::markevery].tolist()
                    
                    plot_kwargs["marker"] = markers[idx % len(markers)]
                    plot_kwargs["markersize"] = 5
                    plot_kwargs["markevery"] = thinned_indices

            axes[i].plot(
                fraction_queries,
                df_sorted[ylabel],
                **plot_kwargs,
            )

            if i == 0:
                axes[i].set_ylabel("Sensitivity up to the 1st FP", fontsize=font_size)
            if i == 1:
                axes[i].set_xlabel("Fraction of Queries", fontsize=font_size)
            axes[i].set_title(f"SCOPe {titles[i]}", fontsize=font_size)
            axes[i].set_xlim(0, 1)
            axes[i].set_ylim(0, 1)
            axes[i].set_aspect("equal", adjustable="box")
            axes[i].grid(True)

        # Add AUCs to the label
        if add_aurocs:
            aucs_str = f"AUROCs: {aucs[0]:.3f}, {aucs[1]:.3f}, {aucs[2]:.3f}"
            # aucs_str = f"AUCs: {aucs[0]}, {aucs[1]}, {aucs[2]}"
            axes[0].lines[-1].set_label(f"{name}\n{aucs_str}")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        # title="Method",
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.16),
        # loc="center left",
        # bbox_to_anchor=(0.83, 0.6),
        # ncol=1,
        fontsize=font_size,
        title_fontsize=font_size,
        frameon=False,
    )

    # fig.suptitle("SCOPe Benchmark", fontsize=font_size, y=1.02)

    plt.subplots_adjust(wspace=-0.3, bottom=0.32)
    if type(save) is bool and save:
        plt.savefig("./plots/scope_benchmark.pdf", format="pdf", bbox_inches="tight")
    elif type(save) is str:
        plt.savefig(f"{save}/scope_benchmark.pdf", format="pdf", bbox_inches="tight")
    plt.show()
