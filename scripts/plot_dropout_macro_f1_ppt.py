from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"
OUT_PATH = OUT_DIR / "dropout_macro_f1_ppt.png"

FOLDS = ["fold_00", "fold_03", "fold_07", "fold_13", "fold_17"]
SETTINGS = [
    ("普通融合模型", ROOT / "results" / "deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog"),
    ("EOG Dropout模型", ROOT / "results" / "deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog_dropout_p05"),
]
SCENARIOS = [
    ("正常场景", "eval_test", "#42D6C1"),
    ("EOG缺失场景", "eval_test_eog_zero", "#A793F2"),
]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def load_macro_f1(root: Path, subdir: str) -> float:
    values = []
    for fold in FOLDS:
        path = root / fold / subdir / "evaluation_test.json"
        if not path.exists():
            raise FileNotFoundError(path)
        values.append(float(read_json(path)["macro_f1"]))
    return mean(values)


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    values_by_setting = []
    for _, root in SETTINGS:
        values_by_setting.append([load_macro_f1(root, subdir) for _, subdir, _ in SCENARIOS])

    baseline_missing = values_by_setting[0][1]
    dropout_missing = values_by_setting[1][1]
    relative_gain = (dropout_missing - baseline_missing) / baseline_missing * 100.0

    fig = plt.figure(figsize=(8.6, 6.2), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")

    title_ax = fig.add_axes([0.08, 0.78, 0.84, 0.16])
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.80,
        "实验验证：性能对比分析",
        fontsize=16,
        fontweight="bold",
        color="#9A7CF4",
        va="top",
    )
    title_ax.text(
        0.0,
        0.30,
        "基于5折交叉验证实验数据，对比普通融合模型与引入EOG Dropout的模型在“正常场景”与“EOG缺失场景”下的Macro-F1表现。",
        fontsize=10.5,
        color="#4B5563",
        va="top",
        wrap=True,
    )

    ax = fig.add_axes([0.18, 0.36, 0.64, 0.34])
    ax.set_facecolor("#FFFFFF")

    x = list(range(len(SETTINGS)))
    width = 0.28
    offsets = [-width / 1.8, width / 1.8]

    for scenario_idx, (label, _, color) in enumerate(SCENARIOS):
        positions = [item + offsets[scenario_idx] for item in x]
        values = [setting_values[scenario_idx] for setting_values in values_by_setting]
        bars = ax.bar(
            positions,
            values,
            width=width,
            label=label,
            color=color,
            edgecolor="#FFFFFF",
            linewidth=1.2,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#374151",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in SETTINGS], fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"], fontsize=10)
    ax.grid(axis="y", color="#E5E7EB", linewidth=1.1, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0, pad=10, colors="#333333")
    ax.tick_params(axis="y", length=0, pad=8, colors="#333333")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=False,
        fontsize=10,
        handlelength=1.2,
        columnspacing=1.6,
    )

    note_ax = fig.add_axes([0.08, 0.08, 0.84, 0.18])
    note_ax.axis("off")
    note_ax.add_patch(
        plt.Rectangle((0, 0), 1, 1, transform=note_ax.transAxes, color="#DCF8F3", ec="none")
    )
    note_ax.text(
        0.04,
        0.58,
        "关键发现：",
        fontsize=11,
        fontweight="bold",
        color="#263244",
        va="center",
    )
    note_ax.text(
        0.20,
        0.58,
        f"在EOG缺失场景下，Macro-F1由{baseline_missing:.3f}提升至{dropout_missing:.3f}，相对提升约{relative_gain:.1f}%。",
        fontsize=11,
        color="#263244",
        va="center",
    )
    note_ax.text(
        0.04,
        0.25,
        "说明EOG Dropout是一个具有竞争力的鲁棒性强基线，后续模块需要以此作为严格对照。",
        fontsize=10.5,
        color="#4B5563",
        va="center",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
