import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
gpus = ["A30", "A40", "A100"]

latency_bf16 = [594, 650, 311]
latency_int8 = [458, 420, 282]

energy_bf16 = [95, 174, 77]
energy_int8 = [72, 110, 68]

x = np.arange(len(gpus))
width = 0.34

# -----------------------------
# Figure
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ===== Latency =====
ax = axes[0]

bars1 = ax.bar(
    x - width / 2,
    latency_bf16,
    width,
    label="BF16"
)

bars2 = ax.bar(
    x + width / 2,
    latency_int8,
    width,
    label="INT8"
)

ax.set_title("Inference Latency", fontsize=15, fontweight="bold")
ax.set_ylabel("Latency (ms)", fontsize=12)
ax.set_xlabel("GPU", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(gpus, fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

# Value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 10,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

ax.legend(frameon=False)

# ===== Energy =====
ax = axes[1]

bars3 = ax.bar(
    x - width / 2,
    energy_bf16,
    width,
    label="BF16"
)

bars4 = ax.bar(
    x + width / 2,
    energy_int8,
    width,
    label="INT8"
)

ax.set_title("Energy Consumption", fontsize=15, fontweight="bold")
ax.set_ylabel("Energy per Inference (J)", fontsize=12)
ax.set_xlabel("GPU", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(gpus, fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

# Value labels
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 3,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

ax.legend(frameon=False)

# -----------------------------
# Overall formatting
# -----------------------------
fig.suptitle(
    "Qwen 8B Performance Across GPUs",
    fontsize=18,
    fontweight="bold",
    y=1.02
)

plt.tight_layout()

# Save publication-quality figure
plt.savefig(
    "qwen8b_gpu_comparison.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "qwen8b_gpu_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()