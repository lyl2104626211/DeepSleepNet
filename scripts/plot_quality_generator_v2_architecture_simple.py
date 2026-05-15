from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"
OUT_PATH = OUT_DIR / "quality_generator_v2_architecture_simple.png"


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#CBD5E1", lw=1.5, fs=11, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color="#263244",
        fontweight=weight,
        linespacing=1.2,
    )
    return patch


def arrow(ax, x1, y1, x2, y2, color="#64748B", lw=1.7):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color=color,
            connectionstyle="arc3,rad=0.0",
        )
    )


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(13.2, 6.4), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.93, "Quality-Guided Generator v2", fontsize=21, fontweight="bold", color="#1F2937")
    ax.text(
        0.04,
        0.875,
        "质量感知 + EEG→EOG 特征恢复 + 可学习 EOG 注入强度",
        fontsize=12,
        color="#4B5563",
    )

    # Main pipeline.
    box(ax, 0.04, 0.66, 0.13, 0.09, "EEG", fc="#E0F7F2", ec="#42D6C1", weight="bold")
    box(ax, 0.23, 0.66, 0.16, 0.09, "EEG Encoder", fc="#F8FAFC")
    box(ax, 0.45, 0.66, 0.13, 0.09, "H_eeg", fc="#E0F7F2", ec="#42D6C1", weight="bold")

    box(ax, 0.04, 0.42, 0.13, 0.09, "EOG", fc="#F0EAFE", ec="#A793F2", weight="bold")
    box(ax, 0.23, 0.42, 0.16, 0.09, "EOG Encoder", fc="#F8FAFC")
    box(ax, 0.45, 0.42, 0.13, 0.09, "H_eog_real", fc="#F0EAFE", ec="#A793F2", weight="bold", fs=10.5)

    box(ax, 0.23, 0.19, 0.16, 0.09, "Quality Sensor\nvariance rule", fc="#F8FAFC", fs=10)
    box(ax, 0.45, 0.19, 0.13, 0.09, "c ∈ {0,1}", fc="#DCFCE7", ec="#86EFAC", weight="bold")

    box(ax, 0.64, 0.66, 0.15, 0.09, "Generator\nMLP", fc="#EEF2FF", ec="#A793F2")
    box(ax, 0.84, 0.66, 0.12, 0.09, "H_eog_fake", fc="#EEF2FF", ec="#A793F2", weight="bold", fs=10.5)

    box(
        ax,
        0.64,
        0.40,
        0.32,
        0.12,
        "Quality-Guided Fusion\nH_final = c·H_real + (1-c)·H_fake",
        fc="#FFFFFF",
        ec="#CBD5E1",
        fs=10.5,
        weight="bold",
    )
    box(
        ax,
        0.64,
        0.19,
        0.32,
        0.11,
        "Residual Injection\nH_fusion = H_eeg + α·H_final",
        fc="#ECFEFF",
        ec="#67E8F9",
        fs=10.5,
        weight="bold",
    )
    box(ax, 0.64, 0.04, 0.15, 0.08, "DeepSleepNet\nBiLSTM", fc="#F8FAFC", fs=10)
    box(ax, 0.84, 0.04, 0.12, 0.08, "Classifier", fc="#F8FAFC", fs=10)

    # Arrows.
    arrow(ax, 0.17, 0.705, 0.23, 0.705)
    arrow(ax, 0.39, 0.705, 0.45, 0.705)
    arrow(ax, 0.58, 0.705, 0.64, 0.705)
    arrow(ax, 0.79, 0.705, 0.84, 0.705)

    arrow(ax, 0.17, 0.465, 0.23, 0.465)
    arrow(ax, 0.39, 0.465, 0.45, 0.465)
    arrow(ax, 0.17, 0.445, 0.23, 0.235)
    arrow(ax, 0.39, 0.235, 0.45, 0.235)

    arrow(ax, 0.58, 0.465, 0.64, 0.46)
    arrow(ax, 0.90, 0.66, 0.90, 0.52)
    arrow(ax, 0.515, 0.28, 0.68, 0.40)
    arrow(ax, 0.80, 0.40, 0.80, 0.30)
    arrow(ax, 0.72, 0.19, 0.72, 0.12)
    arrow(ax, 0.79, 0.08, 0.84, 0.08)

    # H_eeg shortcut to residual injection.
    arrow(ax, 0.515, 0.66, 0.66, 0.30, color="#42A5A1")

    # Training loss note.
    box(
        ax,
        0.04,
        0.05,
        0.42,
        0.10,
        "Training loss:  CE(y, ŷ) + 0.01 · MSE(H_eog_fake, H_eog_clean.detach())\n训练时模块内部随机遮挡 EOG；推理时根据 EOG 质量选择真实/生成 EOG 特征。",
        fc="#FFF7ED",
        ec="#FDBA74",
        fs=9.5,
    )

    # Small note for alpha.
    ax.text(0.64, 0.335, "α 为可学习参数，控制 EOG/伪EOG 注入强度", fontsize=9.5, color="#4B5563")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
