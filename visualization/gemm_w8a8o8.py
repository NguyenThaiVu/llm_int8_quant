import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Data
# --------------------------------------------------
dims = np.array([1024, 2048, 4096, 8192])

latency_fused = np.array([0.02, 0.09, 0.63, 4.85])
latency_separate = np.array([0.03, 0.11, 0.74, 5.38])

energy_fused = np.array([2, 7, 85, 1318])
energy_separate = np.array([2, 9, 117, 1474])

rmse_fused = np.array([252, 231, 270, 360])
rmse_separate = np.array([119, 169, 241, 345])

# --------------------------------------------------
# Figure
# --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

# ==================================================
# (a) Latency
# ==================================================
ax = axes[0]

ax.plot(
    dims,
    latency_fused,
    marker="o",
    linewidth=2,
    markersize=6,
    label="Fused",
)

ax.plot(
    dims,
    latency_separate,
    marker="s",
    linewidth=2,
    markersize=6,
    label="Separate",
)

ax.set_yscale("log")
ax.set_xlabel("GEMM Dimension")
ax.set_ylabel("Latency (ms)")
ax.set_title("(a) Latency")
ax.set_xticks(dims)
ax.set_xticklabels(["1K", "2K", "4K", "8K"])

ax.grid(True, which="major", axis="y", alpha=0.25)
ax.legend(frameon=False)


# ==================================================
# (b) Energy
# ==================================================
ax = axes[1]

ax.plot(
    dims,
    energy_fused,
    marker="o",
    linewidth=2,
    markersize=6,
    label="Fused",
)

ax.plot(
    dims,
    energy_separate,
    marker="s",
    linewidth=2,
    markersize=6,
    label="Separate",
)

ax.set_yscale("log")
ax.set_xlabel("GEMM Dimension")
ax.set_ylabel("Energy (mJ)")
ax.set_title("(b) Energy")
ax.set_xticks(dims)
ax.set_xticklabels(["1K", "2K", "4K", "8K"])

ax.grid(True, which="major", axis="y", alpha=0.25)


# ==================================================
# (c) RMSE
# ==================================================
ax = axes[2]

x = np.arange(len(dims))
width = 0.35

ax.bar(
    x - width / 2,
    rmse_fused,
    width,
    label="Fused",
)

ax.bar(
    x + width / 2,
    rmse_separate,
    width,
    label="Separate",
)

ax.set_xlabel("GEMM Dimension")
ax.set_ylabel("RMSE")
ax.set_title("(c) Numerical Error")

ax.set_xticks(x)
ax.set_xticklabels(["1K", "2K", "4K", "8K"])

ax.grid(True, axis="y", alpha=0.25)


# --------------------------------------------------
# Final layout
# --------------------------------------------------
fig.tight_layout()

plt.savefig(
    "w8a8o8_fused_vs_separate.pdf",
    bbox_inches="tight",
)

plt.savefig(
    "w8a8o8_fused_vs_separate.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()