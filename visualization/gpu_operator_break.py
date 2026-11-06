import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch


# -----------------------------
# Data
# -----------------------------
gpus = ["RTX 5000", "A30", "A40", "A100"]

# BF16
bf16_matmul = np.array([214, 281, 309, 164])
bf16_nonmatmul = np.array([225, 302, 338, 142])

# INT8
int8_matmul = np.array([148, 207, 244, 141])
int8_nonmatmul = np.array([101, 234, 182, 128])

bf16_total = bf16_matmul + bf16_nonmatmul
int8_total = int8_matmul + int8_nonmatmul

x = np.arange(len(gpus))
width = 0.34

fig, ax = plt.subplots(figsize=(8, 5))

# -----------------------------
# Colors
# -----------------------------
bf16_dark = "#2878B5"
bf16_light = "#9CC9E2"

int8_dark = "#E67E22"
int8_light = "#F5B97A"

# -----------------------------
# BF16
# -----------------------------
bf16_bottom = ax.bar(
    x - width / 2,
    bf16_matmul,
    width,
    color=bf16_dark,
    edgecolor="black",
    linewidth=0.7
)

bf16_top = ax.bar(
    x - width / 2,
    bf16_nonmatmul,
    width,
    bottom=bf16_matmul,
    color=bf16_light,
    edgecolor="black",
    linewidth=0.7
)

# -----------------------------
# INT8
# -----------------------------
int8_bottom = ax.bar(
    x + width / 2,
    int8_matmul,
    width,
    color=int8_dark,
    edgecolor="black",
    linewidth=0.7
)

int8_top = ax.bar(
    x + width / 2,
    int8_nonmatmul,
    width,
    bottom=int8_matmul,
    color=int8_light,
    edgecolor="black",
    linewidth=0.7
)

# -----------------------------
# Subtle shadow
# -----------------------------
for bars in [bf16_bottom, bf16_top, int8_bottom, int8_top]:
    for bar in bars:
        bar.set_path_effects([
            pe.SimplePatchShadow(
                offset=(1.5, -1.5),
                alpha=0.15
            ),
            pe.Normal()
        ])

# -----------------------------
# Total labels
# -----------------------------
for i in range(len(gpus)):
    ax.text(
        x[i] - width / 2,
        bf16_total[i] + 8,
        f"{bf16_total[i]}",
        ha="center",
        fontsize=12,
        # fontweight="bold"
    )

    ax.text(
        x[i] + width / 2,
        int8_total[i] + 8,
        f"{int8_total[i]}",
        ha="center",
        fontsize=12,
        # fontweight="bold"
    )
    
# -----------------------------
# Component labels
# -----------------------------
for i in range(len(gpus)):

    # BF16 Matmul
    ax.text(
        x[i] - width / 2,
        bf16_matmul[i] / 2,
        f"{bf16_matmul[i]}",
        ha="center",
        va="center",
        fontsize=8,
        color="white",
        # fontweight="bold"
    )

    # BF16 Non-matmul
    ax.text(
        x[i] - width / 2,
        bf16_matmul[i] + bf16_nonmatmul[i] / 2,
        f"{bf16_nonmatmul[i]}",
        ha="center",
        va="center",
        fontsize=8,
        color="black",
        # fontweight="bold"
    )

    # INT8 Matmul
    ax.text(
        x[i] + width / 2,
        int8_matmul[i] / 2,
        f"{int8_matmul[i]}",
        ha="center",
        va="center",
        fontsize=8,
        color="white",
        # fontweight="bold"
    )

    # INT8 Non-matmul
    ax.text(
        x[i] + width / 2,
        int8_matmul[i] + int8_nonmatmul[i] / 2,
        f"{int8_nonmatmul[i]}",
        ha="center",
        va="center",
        fontsize=8,
        color="black",
        # fontweight="bold"
    )

# -----------------------------
# Precision labels
# -----------------------------
for i in range(len(gpus)):
    ax.text(
        x[i] - width / 2,
        -28,
        "BF16",
        ha="center",
        fontsize=11,
        color=bf16_dark,
        fontweight="bold"
    )

    ax.text(
        x[i] + width / 2,
        -28,
        "INT8",
        ha="center",
        fontsize=11,
        color=int8_dark,
        fontweight="bold"
    )

# -----------------------------
# Axes
# -----------------------------
ax.set_xticks(x)
ax.set_xticklabels(gpus, fontsize=12)

ax.set_ylabel("CUDA Time (ms)", fontsize=13)
ax.set_xlabel("GPU", fontsize=13)

ax.set_ylim(-45, 720)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.25
)

ax.set_axisbelow(True)

# -----------------------------
# Legend
# -----------------------------
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=bf16_dark, edgecolor="black", label="Bf16 matmul"),
    Patch(facecolor=bf16_light, edgecolor="black", label="Bf16 non-matmul"),
    Patch(facecolor=int8_dark, edgecolor="black", label="Int8 matmul"),
    Patch(facecolor=int8_light, edgecolor="black", label="Int8 non-matmul")
]

ax.legend(
    handles=legend_elements,
    frameon=False,
    loc="upper right"
)

plt.tight_layout()

plt.savefig("gpu_breakdown.pdf", bbox_inches="tight")
plt.savefig("gpu_breakdown.png", dpi=300, bbox_inches="tight")

# plt.show()