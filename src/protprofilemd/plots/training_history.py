import matplotlib.pyplot as plt

def plot_training_history(log_history, metrics_names=["loss", "eval_loss"]):
    plt.style.use("default")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    train_logs = log_history[log_history["loss"].notna()]
    eval_logs = log_history[log_history["eval_loss"].notna()]

    ax1.plot(train_logs["epoch"], train_logs["loss"], label="Training Loss", color="orange", linewidth=1)
    ax1.plot(eval_logs["epoch"], eval_logs["eval_loss"], label="Eval Loss", color="lightblue", linewidth=1)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("KL Divergence Loss", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.legend(loc="upper right")
    ax1.grid(True)
    plt.tight_layout()
    return fig