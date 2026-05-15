from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"
OUT_PATH = OUT_DIR / "eog_missing_decline_ppt.png"

FOLDS = [f"fold_{idx:02d}" for idx in range(20)]
SETTINGS = [
    ("单通道EEG", ROOT / "results" / "deepsleepnet_sleep_edf_paper_fpz_cz_v2", "eval_test"),
    ("EEG+EOG(正常)", ROOT / "results" / "deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog", "eval_test"),
    ("EEG+EOG(EOG置零)", ROOT / "results" / "deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog", "eval_test_eog_zero"),
]
METRICS = [
    ("Acc", "accuracy", "#42D6C1"),
    ("Macro-F1", "macro_f1", "#9A7CF4"),
    ("Kappa", "cohen_kappa", "#A2A9B3"),
]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def load_mean(root: Path, subdir: str, key: str) -> float:
    values = []
    for fold in FOLDS:
        path = root / fold / subdir / "evaluation_test.json"
        if not path.exists():
            raise FileNotFoundError(path)
        values.append(float(read_json(path)[key]))
    return mean(values)


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    values_by_setting = []
    for _, root, subdir in SETTINGS:
        values_by_setting.append([load_mean(root, subdir, key) for _, key, _ in METRICS])

    fig, ax = plt.subplots(figsize=(9.6, 4.8), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x = list(range(len(SETTINGS)))
    width = 0.22
    offsets = [-width, 0, width]

    for metric_idx, (label, _, color) in enumerate(METRICS):
        positions = [item + offsets[metric_idx] for item in x]
        values = [setting_values[metric_idx] for setting_values in values_by_setting]
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
                value + 0.018,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#374151",
            )

    ax.set_title("性能衰减趋势：EOG缺失的量化分析", loc="left", fontsize=16, fontweight="bold", color="#263244", pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _, _ in SETTINGS], fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"], fontsize=10)
    ax.grid(axis="y", color="#E5E7EB", linewidth=1.2, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0, pad=10, colors="#333333")
    ax.tick_params(axis="y", length=0, pad=8, colors="#333333")

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=1.4,
    )

    fig.tight_layout(pad=2.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
