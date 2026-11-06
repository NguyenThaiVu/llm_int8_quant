import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Data
# --------------------------------------------------
prompt_lengths = np.array([128, 512, 1024])

bf16_ttft = np.array([78, 206, 411])
int8_ttft = np.array([125, 195, 320])

bf16_itl = np.array([61, 64, 67])
int8_itl = np.array([55, 56, 60])

bf16_throughput = np.array([16, 15, 14])
int8_throughput = np.array([18, 18, 16])

bf16_total = np.array([7.6, 32, 68])
int8_total = np.array([6.8, 28, 61])


# --------------------------------------------------
# Plot configuration
# --------------------------------------------------
x = np.arange(len(prompt_lengths))
width = 0.35

bf16_color = "#8C8C8C"
int8_color = "#5B8FF9"

fig, axes = plt.subplots(2, 2, figsize=(11, 7))

axes = axes.flatten()


# --------------------------------------------------
# Helper function
# --------------------------------------------------
def plot_metric(
    ax,
    bf16,
    int8,
    ylabel,
    title,
    lower_is_better=True,
    ylim=None
):
    bars1 = ax.bar(
        x - width / 2,
        bf16,
        width,
        label="BF16",
        color=bf16_color,
        edgecolor="black",
        linewidth=0.7
    )

    bars2 = ax.bar(
        x + width / 2,
        int8,
        width,
        label="INT8",
        color=int8_color,
        edgecolor="black",
        linewidth=0.7
    )

    ax.set_xticks(x)
    ax.set_xticklabels(prompt_lengths)
    ax.set_xlabel("Prompt Length")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35
    )

    if ylim is not None:
        ax.set_ylim(ylim)

    # Value labels
    ax.bar_label(
        bars1,
        fmt="%g",
        padding=3,
        fontsize=9
    )

    ax.bar_label(
        bars2,
        fmt="%g",
        padding=3,
        fontsize=9
    )

    # INT8 improvement annotation
    for i, (b, q) in enumerate(zip(bf16, int8)):

        if lower_is_better:
            improvement = (b - q) / b * 100
        else:
            improvement = (q - b) / b * 100

        sign = "+" if improvement >= 0 else ""

        max_value = max(b, q)

        ax.text(
            x[i],
            max_value * 1.12,
            f"{sign}{improvement:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )


# --------------------------------------------------
# TTFT
# --------------------------------------------------
plot_metric(
    axes[0],
    bf16_ttft,
    int8_ttft,
    ylabel="TTFT (ms)",
    title="Time to First Token",
    lower_is_better=True,
    ylim=(0, 520)
)


# --------------------------------------------------
# ITL
# --------------------------------------------------
plot_metric(
    axes[1],
    bf16_itl,
    int8_itl,
    ylabel="ITL (ms/token)",
    title="Inter-Token Latency",
    lower_is_better=True,
    ylim=(0, 82)
)


# --------------------------------------------------
# Throughput
# --------------------------------------------------
plot_metric(
    axes[2],
    bf16_throughput,
    int8_throughput,
    ylabel="Throughput (tokens/s)",
    title="Generation Throughput",
    lower_is_better=False,
    ylim=(0, 23)
)


# --------------------------------------------------
# Total decoding time
# --------------------------------------------------
plot_metric(
    axes[3],
    bf16_total,
    int8_total,
    ylabel="Total Time (s)",
    title="Total Generation Time",
    lower_is_better=True,
    ylim=(0, 85)
)


# --------------------------------------------------
# Shared legend and layout
# --------------------------------------------------
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    frameon=False,
    fontsize=11
)

fig.suptitle(
    "Qwen3-14B Decoding Performance",
    fontsize=15,
    fontweight="bold",
    y=1.05
)

plt.tight_layout()

plt.savefig(
    "qwen3_14b_decoding.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "qwen3_14b_decoding.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()