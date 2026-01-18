import importlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde, pearsonr


def get_submat(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with open(path, "r") as f:
        lines = f.readlines()
        background = np.array([float(x) for x in lines[1][37:].split(" ")])
        lambda_ = float(lines[2][37:])
        alphabet = lines[3].strip().replace("    ", "")[::4]

    submat = pd.read_csv(path, sep="\s+", header=0, skiprows=3, index_col=0).to_numpy()

    return {
        "submat": submat,
        "background": background,
        "lambda": lambda_,
        "alphabet": alphabet,
    }


def weighted_hamming_distance(ref, seq, submat: np.ndarray, alphabet: list[str]):
    sym2idx = {c: i for i, c in enumerate(alphabet)}

    # C[a,b] = S[a,a] - S[a,b]; per-row min-max normalize
    diag = np.diag(submat)
    C = diag[:, None] - submat
    lo = C.min(axis=1, keepdims=True)
    hi = C.max(axis=1, keepdims=True)
    denom = np.where(hi > lo, hi - lo, 1.0)

    idx_ref = [sym2idx[c] for c in ref]
    idx_seq = [sym2idx[c] for c in seq]
    # print(idx_ref)
    # print(idx_seq)
    vals = (C[idx_ref, idx_seq] - lo[idx_ref, 0]) / denom[idx_ref, 0]
    return float(vals.mean())


def hamming_distance(ref, seq, alphabet: list[str]):
    assert len(ref) == len(seq)

    mismatches = sum(r != s for r, s in zip(ref, seq))
    return float(mismatches / len(ref))


def get_weighted_hamming_diffs(entry, f_3Di: h5py.File, submat_3Di: dict):
    domain = entry["domain_name"]
    temp = entry["temperature"]
    repl = entry["replica"]

    weighted_hamming_diffs = []
    # hamming_diffs = []

    ref = f_3Di["foldseek"][domain][f"{temp}_{repl}"][0]

    for seq in f_3Di["foldseek"][domain][f"{temp}_{repl}"]:
        weighted_hamming_diffs.append(
            weighted_hamming_distance(
                ref=ref.decode("utf-8"), seq=seq.decode("utf-8"), submat=submat_3Di["submat"], alphabet=submat_3Di["alphabet"]
            )
        )
        # hamming_diffs.append(hamming_distance(ref=ref.decode("utf-8"), seq=seq.decode("utf-8"), alphabet=submat_3Di["alphabet"]))

    # print(f"Domain: {domain}, Temp: {temp}, Repl: {repl}")
    # print(hamming_diffs)
    # print(weighted_hamming_diffs)
    # print(pearsonr(hamming_diffs, entry['rmsd'])[0])
    # print(pearsonr(weighted_hamming_diffs, entry['rmsd'])[0])

    return weighted_hamming_diffs


def normalize(arr):
    arr = np.array(arr)
    min_val = arr.min()
    max_val = arr.max()
    return (arr - min_val) / (max_val - min_val)


def plot_scatter_weighted_hamming_vs_rmsd(all_hamming, all_rmsd):
    """
    Plots a scatterplot of Weighted Hamming distance vs RMSD with progression coloring per entry.

    Parameters
    ----------
    all_hamming : list of array-like
        List of arrays, one per entry, containing weighted Hamming distances
    all_rmsd : list of array-like
        List of arrays, one per entry, containing RMSD values
    """
    plt.figure(figsize=(8, 6))

    for i, (hamming, rmsd) in enumerate(zip(all_hamming, all_rmsd)):
        colors = plt.cm.viridis(np.linspace(0, 1, len(hamming)))
        # print(pearsonr(hamming, rmsd)[0])
        plt.scatter(
            hamming,
            rmsd,
            c=colors,
            label=f"Entry {i}",
            alpha=0.7,
            s=15,
        )

    plt.xlabel("Normalized Weighted Hamming distance")
    plt.ylabel("Normalized RMSD")
    plt.title("Scatterplot: Weighted Hamming distance vs RMSD (colored by progression per entry)")
    plt.tight_layout()

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation="vertical", pad=0.02, location="right")
    cbar.set_label("Relative Progression through trajectory")

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.show()


def plot_scatter_weighted_hamming_vs_rmsd_avg(ds, save_path=None):
    """
    Plots a scatterplot of Weighted Hamming distance vs RMSD with progression coloring per entry.
    Includes marginal distributions: RMSD on top, Hamming distance on right.
    """
    temperatures = sorted(set(ds["temperature"]))
    cmap = plt.get_cmap("viridis")
    n_colors = max(1, len(temperatures))
    colors = cmap(np.linspace(0, 1, n_colors))
    temp2color = {temp: colors[i] for i, temp in enumerate(temperatures)}

    # Create figure with GridSpec for marginal plots
    fig = plt.figure(figsize=(4, 4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.05, wspace=0.05, width_ratios=[7, 1], height_ratios=[1, 7])

    # Main scatter plot
    ax_main = fig.add_subplot(gs[1, 0])
    # Top marginal (RMSD distribution)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    # Right marginal (Hamming distance distribution)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Store data per temperature for marginal plots
    temp_data = {}
    all_hamming = []
    all_rmsd = []

    for temp in temperatures:
        idxs = [i for i, t in enumerate(ds["temperature"]) if t == temp]
        avg_hamming = [np.mean(ds["weighted_hamming_distance"][i]) for i in idxs]
        avg_rmsd = [np.mean(ds["rmsd"][i]) for i in idxs]
        temp_data[temp] = {"hamming": avg_hamming, "rmsd": avg_rmsd}
        all_hamming.extend(avg_hamming)
        all_rmsd.extend(avg_rmsd)
        label = f"{str(temp)}K, PCC: {pearsonr(avg_hamming, avg_rmsd)[0]:.2f}"
        ax_main.scatter(
            avg_rmsd,
            avg_hamming,
            alpha=0.5,
            s=18,
            label=label,
            color=temp2color[temp],
        )

    all_hamming = np.array(all_hamming)
    all_rmsd = np.array(all_rmsd)
    pearson_corr, pearson_p = pearsonr(all_hamming, all_rmsd)

    ax_main.set_xlabel("Mean Trajectory RMSD (Å)")
    ax_main.set_ylabel("Mean Weighted 3Di Hamming Distance")

    ax_main.legend(title="Temperature", fontsize=7, title_fontsize=8)

    ax_main.set_xlim(left=0)
    ax_main.set_ylim(bottom=0)
    ax_main.yaxis.set_major_locator(MultipleLocator(0.1))

    # Top marginal: RMSD distributions per temperature (KDE line plot)
    x_range = np.linspace(0, max(all_rmsd) * 1.1, 200)
    for temp in temperatures:
        data = temp_data[temp]["rmsd"]
        if len(data) > 1:
            kde = gaussian_kde(data)
            ax_top.plot(x_range, kde(x_range), color=temp2color[temp], alpha=0.8)
            ax_top.fill_between(x_range, kde(x_range), alpha=0.2, color=temp2color[temp])
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.axis("off")

    # Right marginal: Hamming distance distributions per temperature (KDE line plot)
    y_range = np.linspace(0, max(all_hamming) * 1.1, 200)
    for temp in temperatures:
        data = temp_data[temp]["hamming"]
        if len(data) > 1:
            kde = gaussian_kde(data)
            ax_right.plot(kde(y_range), y_range, color=temp2color[temp], alpha=0.8)
            ax_right.fill_betweenx(y_range, kde(y_range), alpha=0.2, color=temp2color[temp])
    ax_right.set_ylim(ax_main.get_ylim())
    ax_right.axis("off")

    if save_path is not None:
        format = save_path.split(".")[-1]
        plt.savefig(
            save_path,
            format=format,
            bbox_inches="tight",
        )

    plt.show()

    return pearson_corr, pearson_p


def boxplot_hamming_rmsd(ds, save_path=None):
    """
    Plots a barplot of Weighted Hamming distance vs RMSD.
    """
    fig, ax_rmsd = plt.subplots(figsize=(4, 2))

    temperatures = sorted(set(ds["temperature"]))
    cmap = plt.get_cmap("viridis")
    n_colors = max(1, len(temperatures))
    colors = cmap(np.linspace(0, 1, n_colors))
    temp2color = {temp: colors[i] for i, temp in enumerate(temperatures)}
    alpha = 0.85

    hamming_data = []
    rmsd_data = []
    labels = []
    box_colors = []

    for temp in temperatures:
        idxs = [i for i, t in enumerate(ds["temperature"]) if t == temp]
        avg_hamming = [np.mean(ds["weighted_hamming_distance"][i]) for i in idxs]
        avg_rmsd = [np.mean(ds["rmsd"][i]) for i in idxs]
        hamming_data.append(avg_hamming)
        rmsd_data.append(avg_rmsd)
        labels.append(f"{str(temp)}K")
        box_colors.append(temp2color[temp])

    positions = np.arange(1, len(temperatures) + 1)  # Increased spacing between groups
    width = 0.33  # Width of each box
    intra_group_spacing = 0.22  # Spacing between boxes within a group

    # Plot RMSD on left y-axis
    bplot_rmsd = ax_rmsd.boxplot(
        rmsd_data,
        patch_artist=True,
        positions=positions - intra_group_spacing,
        widths=width,
        labels=labels,
        medianprops=dict(color="black"),
    )
    for patch, color in zip(bplot_rmsd["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(alpha)
    ax_rmsd.set_ylabel("Mean Trajectory RMSD")
    ax_rmsd.set_ylim(bottom=0)

    # Make right-hand y-axis for Hamming
    ax_hamming = ax_rmsd.twinx()
    bplot_hamming = ax_hamming.boxplot(
        hamming_data,
        patch_artist=True,
        positions=positions + intra_group_spacing,
        widths=width,
        labels=[""] * len(labels),  # Don't repeat labels on the right y-axis
        medianprops=dict(color="black"),
    )
    for patch, color in zip(bplot_hamming["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(alpha)
    ax_hamming.set_ylabel("Mean Weighted 3Di\nHamming Distance")
    ax_hamming.set_ylim(bottom=0)

    # Create two rows of x-axis labels
    # First, set positions for individual boxes
    rmsd_positions = positions - intra_group_spacing
    hamming_positions = positions + intra_group_spacing

    # Set all box positions as x-ticks
    all_positions = []
    metric_labels = []
    for i in range(len(temperatures)):
        all_positions.extend([rmsd_positions[i], hamming_positions[i]])
        metric_labels.extend(["RMSD", "D"])

    ax_rmsd.set_xticks(all_positions)
    ax_rmsd.set_xticklabels(metric_labels)
    ax_rmsd.tick_params(axis="x", which="major", pad=2)

    # Add second row of labels for temperature
    for i, (pos, label) in enumerate(zip(positions, labels)):
        ax_rmsd.text(pos, -0.14, label, transform=ax_rmsd.get_xaxis_transform(), ha="center", va="top")

    # Set x-axis limits to reduce edge margins
    x_margin = 0.3  # Small margin at edges
    ax_rmsd.set_xlim(positions[0] - width / 2 - x_margin, positions[-1] + width / 2 + x_margin)

    # Add "Temperature" label below the temperature values
    # fig.text(0.5, 0.04, 'Temperature', ha='center', va='center')

    # plt.title("Weighted 3Di Hamming and RMSD by Temperature")
    plt.tight_layout(rect=[0, 0.03, 1, 1])  # Leave space for the temperature label

    plt.tight_layout()

    if save_path is not None:
        format = save_path.split(".")[-1]
        plt.savefig(save_path, format=format)

    plt.show()


def plot_gradient_weighted_hamming_vs_rmsd(all_hamming, all_rmsd, sigma=5.0, resolution=300):
    """
    Plots a smooth gradient that follows the data shape, colored by progression.
    The gradient is created by interpolating progression values across the data distribution.

    Parameters
    ----------
    all_hamming : list of array-like
        List of arrays, one per entry, containing weighted Hamming distances
    all_rmsd : list of array-like
        List of arrays, one per entry, containing RMSD values
    sigma : float, optional
        Smoothing parameter (higher = smoother gradient), default 5.0
    resolution : int, optional
        Grid resolution for the smooth plot, default 300
    """
    plt.figure(figsize=(8, 6))

    # Collect all points with their progression values
    all_points = []
    all_progressions = []

    for i, (hamming, rmsd) in enumerate(zip(all_hamming, all_rmsd)):
        n_points = len(hamming)
        progression_values = np.linspace(0, 1, n_points)

        for x, y, prog in zip(hamming, rmsd, progression_values):
            all_points.append([x, y])
            all_progressions.append(prog)

    all_points = np.array(all_points)
    all_progressions = np.array(all_progressions)

    # Create grids for density and progression
    density_grid = np.zeros((resolution, resolution))
    progression_grid = np.zeros((resolution, resolution))

    # Bin the data to the grid
    for point, prog in zip(all_points, all_progressions):
        x_idx = int(point[0] * (resolution - 1))
        y_idx = int(point[1] * (resolution - 1))

        x_idx = np.clip(x_idx, 0, resolution - 1)
        y_idx = np.clip(y_idx, 0, resolution - 1)

        density_grid[y_idx, x_idx] += 1
        progression_grid[y_idx, x_idx] += prog

    # Don't average yet - keep the weighted sum for proper blurring
    # progression_grid currently contains sum of progression values

    # Apply Gaussian blur to the weighted sum and the weights separately
    # This prevents edge artifacts from averaging with zeros
    weighted_progression = gaussian_filter(progression_grid, sigma=sigma)
    density_smooth = gaussian_filter(density_grid, sigma=sigma * 1.5)

    # Now divide to get the weighted average (avoid division by zero)
    progression_smooth = np.zeros_like(weighted_progression)
    mask = density_smooth > 0
    progression_smooth[mask] = weighted_progression[mask] / density_smooth[mask]

    # Enhance brightness by applying power transformation
    # This makes bright values (yellows) even brighter
    progression_enhanced = progression_smooth**0.7  # Values < 1 raised to power < 1 get brighter

    # Create RGBA image
    extent = [0, 1, 0, 1]
    rgba_image = np.zeros((resolution, resolution, 4))

    # Get colors from viridis colormap based on progression
    viridis = plt.cm.get_cmap("viridis")
    for i in range(resolution):
        for j in range(resolution):
            color = viridis(progression_enhanced[i, j])
            rgba_image[i, j] = color

    # Set alpha channel based on density (fade to transparent only where there's no data)
    # Normalize density to [0, 1] range for alpha
    max_density = density_smooth.max()
    if max_density > 0:
        alpha = density_smooth / max_density
        # Apply strong power transformation to keep most areas fully opaque
        alpha = alpha**0.15  # Very low power = most data areas at full opacity
        # Clip to ensure values stay in [0, 1]
        alpha = np.clip(alpha, 0, 1)
        rgba_image[:, :, 3] = alpha

    # Set black background for plot area only
    ax = plt.gca()
    ax.set_facecolor("black")

    # Display the RGBA image
    plt.imshow(rgba_image, extent=extent, origin="lower", aspect="auto", interpolation="bilinear")

    plt.xlabel("Weighted Hamming distance")
    plt.ylabel("RMSD")
    plt.title("Gradient Plot: Weighted Hamming distance vs RMSD (colored by progression)")
    plt.tight_layout()

    # Create colorbar with viridis colors
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, location="right")
    cbar.set_label("Progression")

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.show()


def plot_gradient_shape_weighted_hamming_vs_rmsd(all_hamming, all_rmsd, resolution=500, padding_percentile=95):
    """
    Plots a smooth gradient within a geometric shape fitted to the data.
    Uses interpolation to create smooth color transitions based on progression.

    Parameters
    ----------
    all_hamming : list of array-like
        List of arrays, one per entry, containing weighted Hamming distances
    all_rmsd : list of array-like
        List of arrays, one per entry, containing RMSD values
    resolution : int, optional
        Grid resolution for the smooth plot, default 500
    padding_percentile : float, optional
        Percentile to use for determining data boundary (default 95)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Collect all points with their progression values
    all_points = []
    all_progressions = []

    for i, (hamming, rmsd) in enumerate(zip(all_hamming, all_rmsd)):
        n_points = len(hamming)
        progression_values = np.linspace(0, 1, n_points)

        for x, y, prog in zip(hamming, rmsd, progression_values):
            all_points.append([x, y])
            all_progressions.append(prog)

    all_points = np.array(all_points)
    all_progressions = np.array(all_progressions)

    # Create a grid for interpolation
    x = all_points[:, 0]
    y = all_points[:, 1]

    xi = np.linspace(0, 1, resolution)
    yi = np.linspace(0, 1, resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # Interpolate progression values using cubic interpolation
    zi = griddata(all_points, all_progressions, (xi_grid, yi_grid), method="cubic", fill_value=np.nan)

    # Create a mask based on distance from actual data points
    # Use a density-based approach to find the shape boundary
    from scipy.spatial import distance_matrix

    # For each grid point, find distance to nearest data point
    grid_points = np.c_[xi_grid.ravel(), yi_grid.ravel()]

    # Sample data points to speed up computation (use every Nth point)
    sample_rate = max(1, len(all_points) // 5000)
    sampled_points = all_points[::sample_rate]

    # Compute distance to nearest data point for each grid point
    dist_matrix = distance_matrix(grid_points, sampled_points)
    min_distances = dist_matrix.min(axis=1).reshape(resolution, resolution)

    # Determine threshold based on data density
    # Use a percentile of the distances at actual data points
    threshold = np.percentile(min_distances[~np.isnan(zi)], padding_percentile)

    # Create mask: only show where we're close enough to actual data
    mask = min_distances > threshold

    # Apply mask to interpolated values
    zi_masked = np.ma.masked_where(mask, zi)

    # Enhance brightness
    zi_masked_enhanced = zi_masked**0.7

    # Set black background
    ax.set_facecolor("black")

    # Plot the gradient
    im = ax.imshow(
        zi_masked_enhanced,
        extent=[0, 1, 0, 1],
        origin="lower",
        cmap="viridis",
        aspect="auto",
        interpolation="bilinear",
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel("Weighted Hamming distance")
    ax.set_ylabel("RMSD")
    ax.set_title("Gradient Shape Plot: Weighted Hamming distance vs RMSD (colored by progression)")
    plt.tight_layout()

    # Create colorbar
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, location="right")
    cbar.set_label("Progression")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.show()


def plot_kde_weighted_hamming_vs_rmsd(all_hamming, all_rmsd):
    """
    Plots a KDE plot of Weighted Hamming distance vs RMSD as a single density blob.

    Parameters
    ----------
    all_hamming : list of array-like
        List of arrays, one per entry, containing weighted Hamming distances
    all_rmsd : list of array-like
        List of arrays, one per entry, containing RMSD values
    """
    plt.figure(figsize=(8, 6))

    hamming_flat = np.concatenate([np.array(h) for h in all_hamming])
    rmsd_flat = np.concatenate([np.array(r) for r in all_rmsd])

    # Create a viridis colormap with 0 mapped to black
    viridis = plt.cm.get_cmap("viridis", 256)
    viridis_colors = viridis(np.linspace(0, 1, 256))
    viridis_colors[0] = [0, 0, 0, 1]  # Set first color to black, keep alpha=1
    black_viridis = ListedColormap(viridis_colors)

    sns.kdeplot(
        x=hamming_flat,
        y=rmsd_flat,
        cmap=black_viridis,
        fill=True,
        # thresh=0,
        levels=10,
        # alpha=1,
        # cut=0,
        # gridsize=4,
        # clip=((0, 2), (0, 2)),
        clip=None,
    )

    plt.xlabel("Weighted Hamming distance")
    plt.ylabel("RMSD")
    plt.title("KDE Plot: Weighted Hamming distance vs RMSD")
    plt.tight_layout()

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    mappable = plt.cm.ScalarMappable(cmap=black_viridis)
    cbar = plt.colorbar(mappable, ax=plt.gca(), orientation="vertical", pad=0.02, location="right")
    cbar.set_label("Density")

    plt.show()
