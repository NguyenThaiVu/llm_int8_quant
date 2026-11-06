import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Profiling data
# --------------------------------------------------
models = ["Qwen 4B", "Qwen 8B", "Qwen 14B", "Llama 3.2 1B", "Llama 3.2 3B"]
categories = ["Matmul", "Reduction", "Element-wise", "Others"]

times_ms = {
    "Qwen 4B": [182, 106, 195, 14],
    "Qwen 8B": [304, 108, 220, 9],
    "Qwen 14B": [500, 153, 310, 7],
    "Llama 3.2 1B": [62, 45, 67, 6],
    "Llama 3.2 3B": [144, 58, 113, 6],
}

# Totals reported in the original table
reported_totals_ms = {
    "Qwen 4B": 497,
    "Qwen 8B": 641,
    "Qwen 14B": 970,
    "Llama 3.2 1B": 180,
    "Llama 3.2 3B": 322,
}

# Calculate percentages using the reported totals
percentages = {
    model: [
        100.0 * value / reported_totals_ms[model]
        for value in times_ms[model]
    ]
    for model in models
}

# --------------------------------------------------
# Create plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))

y_positions = np.arange(len(models))
left = np.zeros(len(models))

for category_index, category in enumerate(categories):
    values = np.array([
        times_ms[model][category_index]
        for model in models
    ])

    bars = ax.barh(
        y_positions,
        values,
        left=left,
        height=0.8,
        label=category,
    )

    # Add latency and percentage labels
    for model_index, bar in enumerate(bars):
        width = bar.get_width()
        percentage = percentages[models[model_index]][category_index]
        percentage = int(percentage)  # round to nearest integer

        x_center = left[model_index] + width / 2
        y_center = bar.get_y() + bar.get_height() / 2

        # Place labels inside sufficiently large segments
        if width >= 40:
            ax.text(
                x_center,
                y_center,
                # f"{width:.0f} ms\n({percentage}%)",
                f"{width:.0f}\n({percentage}%)",
                ha="center",
                va="center",
                fontsize=12,
            )

        # Place labels above narrow segments
        else:
            ax.text(
                x_center,
                bar.get_y() - 0.05,
                f"{width:.0f} ms",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    left += values

# --------------------------------------------------
# Add total latency labels
# --------------------------------------------------
for model_index, model in enumerate(models):
    ax.text(
        reported_totals_ms[model] + 25,
        model_index,
        f"Total: {reported_totals_ms[model]} ms",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

# --------------------------------------------------
# Formatting
# --------------------------------------------------
ax.set_yticks(y_positions)
ax.set_yticklabels(models, fontsize=13)
ax.invert_yaxis()

ax.set_xlabel("Prefill time (ms)", fontsize=13)

ax.set_title(
    "Prefill-Time Breakdown at 2,048-Token Context Length",
    fontsize=15,
    pad=14,
)

# Show legend below the plot with 4 columns 
ax.legend(
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    frameon=False,
    fontsize=13
)

ax.grid(axis="x", alpha=0.25)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlim(0, max(reported_totals_ms.values()) * 1.18)

fig.tight_layout()

# Save high-resolution figure
fig.savefig("prefill_time_breakdown.png", dpi=300, bbox_inches="tight")
fig.savefig("prefill_time_breakdown.pdf", bbox_inches="tight")
print(f"[INFO] Saved figure as 'prefill_time_breakdown.png' and 'prefill_time_breakdown.pdf'.")

# plt.show()