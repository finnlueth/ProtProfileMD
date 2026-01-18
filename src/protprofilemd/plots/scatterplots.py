from scipy.stats.stats import pearsonr
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_entropy_vs_rmsf_by_temperature(dataset, replica_filter=None, save_path=None):
    """
    Plots scatter of average entropy vs RMSF per domain, colored by temperature.
    Includes marginal distributions: RMSF on top, entropy on right.
    Optionally filters by replica.

    Args:
        dataset: list-like or HuggingFace Dataset with fields "replica", "temperature", "entropy", "rmsf"
        replica_filter: Only include entries where replica == this value. Set to None to skip filtering.
        save_path: Path to save the figure.
    """
    if replica_filter is not None:
        if not isinstance(replica_filter, list):
            replica_filter = [replica_filter]
        mask = np.isin(dataset["replica"], replica_filter)
        ds_masked = dataset.select(np.where(mask)[0])
    else:
        ds_masked = dataset

    temperatures = [row["temperature"] for row in ds_masked]
    unique_temps = sorted(set(temperatures))
    cmap = plt.get_cmap("viridis")
    n_colors = max(1, len(unique_temps))
    colors = cmap(np.linspace(0, 1, n_colors))
    temp2color = {temp: colors[i] for i, temp in enumerate(unique_temps)}

    entropies = [np.mean(row["entropy"]) for row in ds_masked]
    rmsfs = [np.mean(row["rmsf"]) for row in ds_masked]

    # Create figure with GridSpec for marginal plots
    fig = plt.figure(figsize=(4, 4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.05, wspace=0.05, width_ratios=[7, 1], height_ratios=[1, 7])

    # Main scatter plot
    ax_main = fig.add_subplot(gs[1, 0])
    # Top marginal (RMSF distribution)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    # Right marginal (Entropy distribution)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Store data per temperature for marginal plots
    temp_data = {}
    for temp in unique_temps:
        idxs = [i for i, t in enumerate(temperatures) if t == temp]
        e = [entropies[i] for i in idxs]
        r = [rmsfs[i] for i in idxs]
        temp_data[temp] = {"entropy": e, "rmsf": r}
        label = f"{str(temp)}K, PCC: {pearsonr(e, r)[0]:.2f}"
        ax_main.scatter(r, e, alpha=0.5, s=18, label=label, color=temp2color[temp])
        # Best fit line for this temperature
        # if len(r) > 1:
        #     coeffs = np.polyfit(r, e, 1)
        #     x_fit = np.linspace(min(r), max(r), 100)
        #     y_fit = np.polyval(coeffs, x_fit)
        #     ax_main.plot(x_fit, y_fit, color=temp2color[temp], alpha=0.8, linewidth=1.5)

    ds_predicted = dataset.select(np.where(np.isin(dataset["replica"], ["3"]))[0])
    ds_predicted = ds_predicted.select(np.where(np.isin(ds_predicted["temperature"], ["320"]))[0])

    entropies_predicted = [np.mean(row["profile_predicted_entropy"]) for row in ds_predicted]
    rmsfs_predicted = [np.mean(row["rmsf"]) for row in ds_predicted]

    ax_main.scatter(
        rmsfs_predicted,
        entropies_predicted,
        alpha=0.5,
        s=18,
        label=f"ProtProfileMD,\nPCC: {pearsonr(entropies_predicted, rmsfs_predicted)[0]:.2f}",
        color="orange",
    )
    # Best fit line for ProtProfileMD predicted
    # if len(rmsfs_predicted) > 1:
    #     coeffs = np.polyfit(rmsfs_predicted, entropies_predicted, 1)
    #     x_fit = np.linspace(min(rmsfs_predicted), max(rmsfs_predicted), 100)
    #     y_fit = np.polyval(coeffs, x_fit)
    #     ax_main.plot(x_fit, y_fit, color="orange", alpha=0.8, linewidth=1.5)

    entropy_all = np.array(entropies)
    rmsf_all = np.array(rmsfs)
    pearson_corr, pearson_p = pearsonr(entropy_all, rmsf_all)

    ax_main.set_xlabel("Mean Trajectory RMSF (Å)")
    ax_main.set_ylabel("Mean Profile Shannon Entropy (bits)")

    ax_main.legend(title="Temperature or Model", fontsize=7, title_fontsize=8)

    ax_main.set_xlim(left=0)
    ax_main.set_ylim(bottom=0)

    # Top marginal: RMSF distributions per temperature (KDE line plot)
    x_range = np.linspace(0, max(rmsf_all) * 1.1, 200)
    for temp in unique_temps:
        data = temp_data[temp]["rmsf"]
        if len(data) > 1:
            kde = gaussian_kde(data)
            ax_top.plot(x_range, kde(x_range), color=temp2color[temp], alpha=0.8)
            ax_top.fill_between(x_range, kde(x_range), alpha=0.2, color=temp2color[temp])
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.axis("off")

    # Right marginal: Entropy distributions per temperature (KDE line plot)
    y_range = np.linspace(0, max(max(entropy_all), max(entropies_predicted)) * 1.1, 200)
    for temp in unique_temps:
        data = temp_data[temp]["entropy"]
        if len(data) > 1:
            kde = gaussian_kde(data)
            ax_right.plot(kde(y_range), y_range, color=temp2color[temp], alpha=0.8)
            ax_right.fill_betweenx(y_range, kde(y_range), alpha=0.2, color=temp2color[temp])
    # Add ProtProfileMD predicted entropy KDE
    if len(entropies_predicted) > 1:
        kde_pred = gaussian_kde(entropies_predicted)
        ax_right.plot(kde_pred(y_range), y_range, color="orange", alpha=0.8)
        ax_right.fill_betweenx(y_range, kde_pred(y_range), alpha=0.2, color="orange")
    ax_right.set_ylim(ax_main.get_ylim())
    ax_right.axis("off")

    # fig.suptitle("Mean per Protein:\nProfile Shannon Entropy vs RMSF", fontsize=10, y=0.98)

    if save_path is not None:
        format = save_path.split(".")[-1]
        plt.savefig(
            save_path,
            format=format,
            bbox_inches="tight",
        )

    plt.show()

    return pearson_corr, pearson_p


def plot_predicted_entropy_vs_rmsf_by_temperature(dataset, replica_filter=None, save_path=None):
    """
    Plots scatter of predicted profile Shannon entropy vs ground truth RMSF per domain,
    colored by temperature. Includes marginal distributions: RMSF on top, entropy on right.
    
    This shows how the temperature-independent predicted entropy correlates with
    RMSF at different simulation temperatures.

    Args:
        dataset: list-like or HuggingFace Dataset with fields "replica", "temperature", 
                 "profile_predicted_entropy", "rmsf"
        replica_filter: Only include entries where replica == this value. Set to None to skip filtering.
        save_path: Path to save the figure.
    
    Returns:
        dict: Pearson correlation coefficient and p-value per temperature
    """
    if replica_filter is not None:
        if not isinstance(replica_filter, list):
            replica_filter = [replica_filter]
        mask = np.isin(dataset["replica"], replica_filter)
        ds_masked = dataset.select(np.where(mask)[0])
    else:
        ds_masked = dataset

    temperatures = [row["temperature"] for row in ds_masked]
    unique_temps = sorted(set(temperatures))
    cmap = plt.get_cmap("viridis")
    n_colors = max(1, len(unique_temps))
    colors = cmap(np.linspace(0, 1, n_colors))
    temp2color = {temp: colors[i] for i, temp in enumerate(unique_temps)}

    # Use predicted entropy instead of ground truth entropy
    entropies_predicted = [np.mean(row["profile_predicted_entropy"]) for row in ds_masked]
    rmsfs = [np.mean(row["rmsf"]) for row in ds_masked]

    # Create figure with GridSpec for marginal plots
    fig = plt.figure(figsize=(4, 4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.05, wspace=0.05, width_ratios=[7, 1], height_ratios=[1, 7])

    # Main scatter plot
    ax_main = fig.add_subplot(gs[1, 0])
    # Top marginal (RMSF distribution)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    # Right marginal (Entropy distribution)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Store data per temperature for marginal plots and correlations
    temp_data = {}
    correlations = {}
    
    for temp in unique_temps:
        idxs = [i for i, t in enumerate(temperatures) if t == temp]
        e = [entropies_predicted[i] for i in idxs]
        r = [rmsfs[i] for i in idxs]
        temp_data[temp] = {"entropy": e, "rmsf": r}
        
        pcc, pval = pearsonr(e, r)
        correlations[temp] = {"pcc": pcc, "p_value": pval}
        
        label = f"{str(temp)}K, PCC: {pcc:.2f}"
        ax_main.scatter(r, e, alpha=0.5, s=18, label=label, color=temp2color[temp])

    entropy_all = np.array(entropies_predicted)
    rmsf_all = np.array(rmsfs)
    pearson_corr_all, pearson_p_all = pearsonr(entropy_all, rmsf_all)
    correlations["all"] = {"pcc": pearson_corr_all, "p_value": pearson_p_all}

    ax_main.set_xlabel("Mean Trajectory RMSF (Å)")
    ax_main.set_ylabel("Mean Predicted Profile Shannon Entropy (bits)")

    ax_main.legend(title="Temperature", fontsize=7, title_fontsize=8)

    ax_main.set_xlim(left=0)
    ax_main.set_ylim(bottom=0)

    # Top marginal: RMSF distributions per temperature (KDE line plot)
    x_range = np.linspace(0, max(rmsf_all) * 1.1, 200)
    for temp in unique_temps:
        data = temp_data[temp]["rmsf"]
        if len(data) > 1:
            kde = gaussian_kde(data)
            ax_top.plot(x_range, kde(x_range), color=temp2color[temp], alpha=0.8)
            ax_top.fill_between(x_range, kde(x_range), alpha=0.2, color=temp2color[temp])
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.axis("off")

    # Right marginal: Entropy distributions per temperature (KDE line plot)
    y_range = np.linspace(0, max(entropy_all) * 1.1, 200)
    for temp in unique_temps:
        data = temp_data[temp]["entropy"]
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

    return correlations


def boxplot_entropy_rmsf(dataset, replica_filter=None, save_path=None):
    """
    Plots a boxplot of Entropy vs RMSF by temperature.
    One boxplot per temperature, RMSF on left y-axis, Entropy on right y-axis, sharing same x/temperature axis.

    Args:
        dataset: list-like or HuggingFace Dataset with fields "replica", "temperature", "entropy", "rmsf"
        replica_filter: Only include entries where replica == this value. Set to None to skip filtering.
        save_path: Path to save the figure.
    """
    if replica_filter is not None:
        if not isinstance(replica_filter, list):
            replica_filter = [replica_filter]
        mask = np.isin(dataset["replica"], replica_filter)
        ds_masked = dataset.select(np.where(mask)[0])
    else:
        ds_masked = dataset

    fig, ax_rmsf = plt.subplots(figsize=(5, 2.5))

    temperatures = sorted(set(ds_masked["temperature"]))
    cmap = plt.get_cmap("viridis")
    n_colors = max(1, len(temperatures))
    colors = cmap(np.linspace(0, 1, n_colors))
    temp2color = {temp: colors[i] for i, temp in enumerate(temperatures)}
    alpha = 0.85

    entropy_data = []
    rmsf_data = []
    labels = []
    box_colors = []

    for temp in temperatures:
        idxs = [i for i, t in enumerate(ds_masked["temperature"]) if t == temp]
        avg_entropy = [np.mean(ds_masked["entropy"][i]) for i in idxs]
        avg_rmsf = [np.mean(ds_masked["rmsf"][i]) for i in idxs]
        entropy_data.append(avg_entropy)
        rmsf_data.append(avg_rmsf)
        labels.append(f"{str(temp)}K")
        box_colors.append(temp2color[temp])

    # Add prediction data (ProtProfileMD) - entropy only
    ds_predicted = dataset.select(np.where(np.isin(dataset["replica"], ["3"]))[0])
    ds_predicted = ds_predicted.select(np.where(np.isin(ds_predicted["temperature"], ["320"]))[0])

    entropies_predicted = [np.mean(row["profile_predicted_entropy"]) for row in ds_predicted]

    entropy_data.append(entropies_predicted)
    labels.append("Pred")
    box_colors.append("orange")

    # Position temperature groups
    temp_positions = np.arange(1, len(temperatures) + 1)
    width = 0.33
    intra_group_spacing = 0.22

    # Plot RMSF on left y-axis (only for temperatures, not prediction)
    rmsf_positions = temp_positions - intra_group_spacing
    bplot_rmsf = ax_rmsf.boxplot(
        rmsf_data,
        patch_artist=True,
        positions=rmsf_positions,
        widths=width,
        labels=labels[:-1],
        medianprops={"color": "black"},
    )
    for patch, color in zip(bplot_rmsf["boxes"], box_colors[:-1]):
        patch.set_facecolor(color)
        patch.set_alpha(alpha)
    ax_rmsf.set_ylabel("Mean Trajectory RMSF")
    ax_rmsf.set_ylim(bottom=0)

    # Make right-hand y-axis for Entropy (for all including prediction)
    ax_entropy = ax_rmsf.twinx()
    # Temperature entropies: offset to the right
    entropy_positions_temp = temp_positions + intra_group_spacing
    # Prediction entropy: positioned as single column after last entropy
    pred_position = entropy_positions_temp[-1] + intra_group_spacing + width
    entropy_positions = list(entropy_positions_temp) + [pred_position]

    # Positions for centering group labels
    label_positions = list(temp_positions) + [pred_position]

    bplot_entropy = ax_entropy.boxplot(
        entropy_data,
        patch_artist=True,
        positions=entropy_positions,
        widths=width,
        labels=[""] * len(labels),
        medianprops={"color": "black"},
    )
    for patch, color in zip(bplot_entropy["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(alpha)
    ax_entropy.set_ylabel("Mean Profile\nShannon Entropy")
    ax_entropy.set_ylim(bottom=0)

    # Create two rows of x-axis labels
    # Set all box positions as x-ticks
    all_positions = []
    metric_labels = []
    # For temperature data: both RF and EN
    for i in range(len(temperatures)):
        all_positions.extend([rmsf_positions[i], entropy_positions_temp[i]])
        metric_labels.extend(["RMSF", "H"])
    # For prediction: only EN
    all_positions.append(pred_position)
    metric_labels.append("H")

    ax_rmsf.set_xticks(all_positions)
    ax_rmsf.set_xticklabels(metric_labels)
    ax_rmsf.tick_params(axis="x", which="major", pad=2)

    # Add second row of labels for temperature/model
    for i, (pos, label) in enumerate(zip(label_positions, labels)):
        ax_rmsf.text(pos, -0.14, label, transform=ax_rmsf.get_xaxis_transform(), ha="center", va="top")

    # Set x-axis limits to reduce edge margins
    x_margin = 0.3
    ax_rmsf.set_xlim(temp_positions[0] - intra_group_spacing - width / 2 - x_margin, pred_position + width / 2 + x_margin)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.tight_layout()

    if save_path is not None:
        format = save_path.split(".")[-1]
        plt.savefig(save_path, format=format)

    plt.show()


def plot_hamming_vs_gyration_by_temperature(dataset, replica_filter=None, temperature_filter=None, save_path=None):
    """
    Plots scatter of mean hamming distance vs mean gyration radius per domain, colored by temperature.
    Includes marginal distributions: gyration radius on top, hamming distance on right.
    Optionally filters by replica.

    Args:
        dataset: list-like or HuggingFace Dataset with fields "replica", "temperature", "weighted_hamming_distance", "gyration_radius"
        replica_filter: Only include entries where replica == this value. Set to None to skip filtering.
        temperature_filter: Only include entries where temperature == this value. Set to None to skip filtering.
        save_path: Path to save the figure.
    """
    if replica_filter is not None:  
        if not isinstance(replica_filter, list):
            replica_filter = [replica_filter]
        mask = np.isin(dataset["replica"], replica_filter)
        ds_masked = dataset.select(np.where(mask)[0])
    else:
        ds_masked = dataset

    if temperature_filter is not None:
        if not isinstance(temperature_filter, list):
            temperature_filter = [temperature_filter]
        mask = np.isin(ds_masked["temperature"], temperature_filter)
        ds_masked = ds_masked.select(np.where(mask)[0])

    temperatures = [row["temperature"] for row in ds_masked]
    unique_temps = sorted(set(temperatures))
    cmap = plt.get_cmap("viridis")
    n_colors = max(1, len(unique_temps))
    colors = cmap(np.linspace(0, 1, n_colors))
    temp2color = {temp: colors[i] for i, temp in enumerate(unique_temps)}

    hamming_distances = [np.mean(row["weighted_hamming_distance"]) for row in ds_masked]
    gyration_radii = [np.mean(row["gyration_radius"]) for row in ds_masked]  # Convert to nm

    # Create figure with GridSpec for marginal plots
    fig = plt.figure(figsize=(4, 4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.05, wspace=0.05, width_ratios=[7, 1], height_ratios=[1, 7])

    # Main scatter plot
    ax_main = fig.add_subplot(gs[1, 0])
    # Top marginal (Gyration radius distribution)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    # Right marginal (Hamming distance distribution)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Store data per temperature for marginal plots
    temp_data = {}
    for temp in unique_temps:
        idxs = [i for i, t in enumerate(temperatures) if t == temp]
        h = [hamming_distances[i] for i in idxs]
        g = [gyration_radii[i] for i in idxs]
        temp_data[temp] = {"hamming": h, "gyration": g}
        label = f"{str(temp)}K, PCC: {pearsonr(h, g)[0]:.2f}"
        ax_main.scatter(g, h, alpha=0.5, s=18, label=label, color=temp2color[temp])

    hamming_all = np.array(hamming_distances)
    gyration_all = np.array(gyration_radii)
    pearson_corr, pearson_p = pearsonr(hamming_all, gyration_all)

    ax_main.set_xlabel("Mean Gyration Radius (nm)")
    ax_main.set_ylabel("Mean Profile Hamming Distance")

    ax_main.legend(title="Temperature", fontsize=7, title_fontsize=8, )

    ax_main.set_xlim(left=0)
    ax_main.set_ylim(bottom=0)

    # Top marginal: Gyration radius distributions per temperature (KDE line plot)
    x_range = np.linspace(0, max(gyration_all) * 1.1, 200)
    for temp in unique_temps:
        data = temp_data[temp]["gyration"]
        if len(data) > 1:
            kde = gaussian_kde(data)
            ax_top.plot(x_range, kde(x_range), color=temp2color[temp], alpha=0.8)
            ax_top.fill_between(x_range, kde(x_range), alpha=0.2, color=temp2color[temp])
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.axis("off")

    # Right marginal: Hamming distance distributions per temperature (KDE line plot)
    y_range = np.linspace(0, max(hamming_all) * 1.1, 200)
    for temp in unique_temps:
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

    # return pearson_corr, pearson_p
    return None


def boxplot_pearson_correlation(
    dataset,
    column_x,
    column_y,
    ax=None,
    title=None,
    ylabel=None,
    xlabel=None,
    xlim=None,
    ylim=None,
    figsize=(4, 2),
    replica_filter=None,
    temperature_filter=None,
    predicted_column_y=None,
    predicted_replica=None,
    predicted_temperature=None,
    predicted_label="Pred",
    save_path=None,
):
    """
    Plots a boxplot of Pearson correlation values between two dataset columns,
    grouped by temperature.

    For each sample, calculates the Pearson correlation between the two specified columns
    across all indices/positions, then displays these correlations as boxplots per temperature.

    Args:
        dataset: list-like or HuggingFace Dataset with fields "replica", "temperature", and the specified columns.
        column_x: Name of the first column to correlate (e.g., "rmsd").
        column_y: Name of the second column to correlate (e.g., "weighted_hamming_distance").
        ax: Optional matplotlib Axes to plot on. If None, creates a new figure.
            Pass axes from plt.subplots() to align multiple plots.
        title: Optional title for the plot. Defaults to None (no title).
        ylabel: Optional y-axis label. Defaults to "Pearson Correlation\n({column_x} vs {column_y})".
        xlabel: Optional x-axis label. Defaults to "Temperature".
        xlim: Optional tuple (xmin, xmax) to set x-axis limits. Useful for aligning multiple plots.
        ylim: Optional tuple (ymin, ymax) to set y-axis limits. Defaults to (None, 1).
        figsize: Tuple (width, height) for figure size. Defaults to (4, 2). Ignored if ax is provided.
        replica_filter: Only include entries where replica == this value. Set to None to skip filtering.
        temperature_filter: Only include entries where temperature == this value. Set to None to skip filtering.
        predicted_column_y: Column name for predicted values to correlate with column_x. If not None, adds a "Pred" box.
        predicted_replica: Replica to use for predicted values (e.g., "3"). Required if predicted_column_y is set.
        predicted_temperature: Temperature to use for predicted values (e.g., "320"). Required if predicted_column_y is set.
        predicted_label: Label for the predicted box. Defaults to "Pred".
        save_path: Path to save the figure.
    
    Returns:
        ax: The matplotlib Axes object.
    """
    if replica_filter is not None:
        if not isinstance(replica_filter, list):
            replica_filter = [replica_filter]
        mask = np.isin(dataset["replica"], replica_filter)
        ds_masked = dataset.select(np.where(mask)[0])
    else:
        ds_masked = dataset

    if temperature_filter is not None:
        if not isinstance(temperature_filter, list):
            temperature_filter = [temperature_filter]
        mask = np.isin(ds_masked["temperature"], temperature_filter)
        ds_masked = ds_masked.select(np.where(mask)[0])

    temperatures = sorted(set(ds_masked["temperature"]))
    cmap = plt.get_cmap("viridis")
    n_colors = max(1, len(temperatures))
    colors = cmap(np.linspace(0, 1, n_colors))
    temp2color = {temp: colors[i] for i, temp in enumerate(temperatures)}

    # Calculate Pearson correlations per sample
    pearson_data = {temp: [] for temp in temperatures}
    for row in ds_masked:
        data_x = np.array(row[column_x])
        data_y = np.array(row[column_y])
        # Ensure arrays have the same length
        min_len = min(len(data_x), len(data_y))
        if min_len > 1:
            corr, _ = pearsonr(data_x[:min_len], data_y[:min_len])
            if not np.isnan(corr):
                pearson_data[row["temperature"]].append(corr)

    # Prepare data for boxplot
    box_data = []
    labels = []
    box_colors = []
    
    for temp in temperatures:
        box_data.append(pearson_data[temp] if pearson_data[temp] else [0])
        labels.append(f"{temp}K")
        box_colors.append(temp2color[temp])

    # Add predicted values if specified
    if predicted_column_y is not None:
        ds_predicted = dataset
        if predicted_replica is not None:
            if not isinstance(predicted_replica, list):
                predicted_replica = [predicted_replica]
            mask = np.isin(ds_predicted["replica"], predicted_replica)
            ds_predicted = ds_predicted.select(np.where(mask)[0])
        if predicted_temperature is not None:
            if not isinstance(predicted_temperature, list):
                predicted_temperature = [predicted_temperature]
            mask = np.isin(ds_predicted["temperature"], predicted_temperature)
            ds_predicted = ds_predicted.select(np.where(mask)[0])
        
        # Calculate Pearson correlations for predicted values
        predicted_correlations = []
        for row in ds_predicted:
            data_x = np.array(row[column_x])
            data_y_pred = np.array(row[predicted_column_y])
            min_len = min(len(data_x), len(data_y_pred))
            if min_len > 1:
                corr, _ = pearsonr(data_x[:min_len], data_y_pred[:min_len])
                if not np.isnan(corr):
                    predicted_correlations.append(corr)
        
        box_data.append(predicted_correlations if predicted_correlations else [0])
        labels.append(predicted_label)
        box_colors.append("orange")

    # Create figure/axes if not provided
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    
    x_positions = np.arange(1, len(box_data) + 1)
    bplot = ax.boxplot(
        box_data,
        positions=x_positions,
        patch_artist=True,
        widths=0.6,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    
    # Color the boxes
    for patch, color in zip(bplot["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_xlabel(xlabel if xlabel is not None else "Temperature")
    # Move x-label up a little bit by adjusting the labelpad
    ax.set_xlabel(ax.get_xlabel(), labelpad=-5)
    if ylabel is None:
        ylabel = f"Pearson Correlation\n({column_x} vs {column_y})"
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    
    if title is not None:
        ax.set_title(title)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    
    # Set axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(None, 1)
    
    # Set y-axis major ticks at -1, 0, 1 (with labels) and minor ticks at -0.5, 0.5 (no labels)
    y_min, y_max = ax.get_ylim()
    major_ticks = [t for t in [-1, 0, 1] if y_min <= t <= y_max]
    minor_ticks = [t for t in [-0.5, 0.5] if y_min <= t <= y_max]
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.tick_params(axis='y', which='minor', length=3, width=0.8)

    # Only do layout/show/save if we created the figure
    if created_fig:
        plt.tight_layout()

        if save_path is not None:
            format = save_path.split(".")[-1]
            plt.savefig(save_path, format=format, bbox_inches="tight")

        plt.show()

    return ax

