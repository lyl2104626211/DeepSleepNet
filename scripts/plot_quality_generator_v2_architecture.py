from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"
OUT_PATH = OUT_DIR / "quality_generator_v2_architecture.png"


def add_box(ax, xy, w, h, text, fc="#FFFFFF", ec="#CBD5E1", fontsize=10, weight="normal"):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#263244",
        fontweight=weight,
        linespacing=1.25,
    )
    return box


def add_arrow(ax, start, end, color="#64748B", lw=1.6, style="-|>"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    return arrow


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(14.5, 8.2), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.04,
        0.94,
        "Quality-Guided Generator v2：训练与推理流程",
        fontsize=20,
        fontweight="bold",
        color="#1F2937",
        ha="left",
        va="center",
    )
    ax.text(
        0.04,
        0.895,
        "核心：先用干净 EOG 作为特征教师，再在模块内部随机遮挡 EOG，训练 EEG→EOG 生成器与质量引导融合。",
        fontsize=11.5,
        color="#4B5563",
        ha="left",
        va="center",
    )

    # Input boxes.
    add_box(ax, (0.04, 0.70), 0.14, 0.085, "输入 EEG\n[B, 1, 3000]", fc="#E0F7F2", ec="#42D6C1", fontsize=10)
    add_box(ax, (0.04, 0.46), 0.14, 0.085, "输入 EOG\nclean_eog", fc="#F0EAFE", ec="#A793F2", fontsize=10)

    # Encoders and teacher branch.
    add_box(ax, (0.24, 0.70), 0.16, 0.085, "EEG Encoder\nDeepFeatureNet", fc="#F8FAFC", fontsize=10)
    add_box(ax, (0.24, 0.46), 0.16, 0.085, "EOG Encoder\nclean teacher", fc="#F8FAFC", fontsize=10)
    add_box(ax, (0.44, 0.46), 0.15, 0.085, "H_eog_clean\n.detach()", fc="#FFF7ED", ec="#FDBA74", fontsize=10)

    add_arrow(ax, (0.18, 0.742), (0.24, 0.742))
    add_arrow(ax, (0.18, 0.502), (0.24, 0.502))
    add_arrow(ax, (0.40, 0.502), (0.44, 0.502))

    add_box(ax, (0.44, 0.70), 0.15, 0.085, "H_eeg", fc="#E0F7F2", ec="#42D6C1", fontsize=11, weight="bold")
    add_arrow(ax, (0.40, 0.742), (0.44, 0.742))

    # Generator path.
    add_box(ax, (0.64, 0.70), 0.16, 0.085, "EEG→EOG 生成器\nLinear-ReLU-Linear", fc="#EEF2FF", ec="#A793F2", fontsize=10)
    add_box(ax, (0.84, 0.70), 0.12, 0.085, "H_eog_fake", fc="#EEF2FF", ec="#A793F2", fontsize=10, weight="bold")
    add_arrow(ax, (0.59, 0.742), (0.64, 0.742))
    add_arrow(ax, (0.80, 0.742), (0.84, 0.742))

    # Corruption and observed EOG path.
    add_box(ax, (0.24, 0.29), 0.18, 0.085, "训练时内部遮挡\np=0.5", fc="#FEE2E2", ec="#FCA5A5", fontsize=10)
    add_box(ax, (0.47, 0.29), 0.16, 0.085, "observed_eog", fc="#FEE2E2", ec="#FCA5A5", fontsize=10)
    add_box(ax, (0.68, 0.29), 0.15, 0.085, "EOG Encoder\nobserved", fc="#F8FAFC", fontsize=10)
    add_box(ax, (0.86, 0.29), 0.10, 0.085, "H_eog_real", fc="#F0EAFE", ec="#A793F2", fontsize=10, weight="bold")

    add_arrow(ax, (0.18, 0.49), (0.24, 0.335))
    add_arrow(ax, (0.42, 0.335), (0.47, 0.335))
    add_arrow(ax, (0.63, 0.335), (0.68, 0.335))
    add_arrow(ax, (0.83, 0.335), (0.86, 0.335))

    # Quality sensor.
    add_box(ax, (0.47, 0.13), 0.16, 0.085, "质量感知器\n方差阈值", fc="#F8FAFC", fontsize=10)
    add_box(ax, (0.68, 0.13), 0.15, 0.085, "c ∈ {0,1}\n可靠性标记", fc="#DCFCE7", ec="#86EFAC", fontsize=10, weight="bold")
    add_arrow(ax, (0.55, 0.29), (0.55, 0.215))
    add_arrow(ax, (0.63, 0.172), (0.68, 0.172))

    # Fusion and downstream.
    add_box(
        ax,
        (0.50, 0.565),
        0.27,
        0.10,
        "质量引导融合\nH_final = c·H_real + (1-c)·H_fake",
        fc="#F8FAFC",
        ec="#CBD5E1",
        fontsize=10,
    )
    add_box(
        ax,
        (0.50, 0.405),
        0.27,
        0.10,
        "v2 残差注入\nH_fusion = H_eeg + α·H_final\nα = sigmoid(learnable logit)",
        fc="#ECFEFF",
        ec="#67E8F9",
        fontsize=9.5,
    )
    add_box(ax, (0.82, 0.475), 0.14, 0.085, "DeepSleepNet\nBiLSTM + classifier", fc="#F8FAFC", fontsize=10)
    add_box(ax, (0.82, 0.13), 0.14, 0.085, "Loss\nCE + 0.01·MSE", fc="#FFF7ED", ec="#FDBA74", fontsize=10, weight="bold")

    add_arrow(ax, (0.90, 0.70), (0.77, 0.62))
    add_arrow(ax, (0.91, 0.375), (0.77, 0.61))
    add_arrow(ax, (0.755, 0.172), (0.67, 0.565))
    add_arrow(ax, (0.575, 0.70), (0.58, 0.665))
    add_arrow(ax, (0.635, 0.565), (0.635, 0.505))
    add_arrow(ax, (0.77, 0.455), (0.82, 0.515))

    # Loss arrows.
    add_arrow(ax, (0.90, 0.70), (0.90, 0.215), color="#F97316", lw=1.4)
    add_arrow(ax, (0.515, 0.46), (0.82, 0.172), color="#F97316", lw=1.4)
    ax.text(0.91, 0.43, "MSE(H_fake, H_clean.detach())", fontsize=8.5, color="#9A3412", rotation=90, va="center")

    # Notes.
    add_box(
        ax,
        (0.04, 0.08),
        0.35,
        0.12,
        "推理/评估时：\n不再随机遮挡；若测试 EOG=0，质量感知器给 c=0，模型更多使用 H_eog_fake。",
        fc="#F8FAFC",
        ec="#CBD5E1",
        fontsize=9.5,
    )
    add_box(
        ax,
        (0.04, 0.235),
        0.35,
        0.12,
        "为什么 v2 有效：\nα 控制 EOG/伪 EOG 对主特征的注入强度，缓解 v1 直接相加造成的 normal 性能干扰。",
        fc="#F8FAFC",
        ec="#CBD5E1",
        fontsize=9.5,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
