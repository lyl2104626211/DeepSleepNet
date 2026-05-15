from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SINGLE_ROOT = ROOT / "results" / "deepsleepnet_sleep_edf_paper_fpz_cz_v2"
DUAL_ROOT = ROOT / "results" / "deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog"
OUT_DIR = ROOT / "docs" / "figures"
OUT_PATH = OUT_DIR / "eeg_eog_robustness_20fold.png"

FOLDS = [f"fold_{idx:02d}" for idx in range(20)]
SETTINGS = [
    ("EEG only", SINGLE_ROOT, "eval_test"),
    ("EEG+EOG", DUAL_ROOT, "eval_test"),
    ("EOG=0", DUAL_ROOT, "eval_test_eog_zero"),
]
METRICS = [
    ("Accuracy", "accuracy"),
    ("Macro-F1", "macro_f1"),
    ("Kappa", "cohen_kappa"),
]
CLASS_LABELS = ["W", "N1", "N2", "N3", "REM"]


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_eval_values(root: Path, subdir: str, metric_key: str) -> list[float]:
    values = []
    for fold in FOLDS:
        path = root / fold / subdir / "evaluation_test.json"
        if not path.exists():
            raise FileNotFoundError(path)
        values.append(float(read_json(path)[metric_key]))
    return values


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    mu = mean(values)
    return (sum((value - mu) ** 2 for value in values) / len(values)) ** 0.5


def sum_confusion_matrices(root: Path, subdir: str) -> tuple[list[str], list[list[int]]]:
    labels = None
    matrix_sum = None

    for fold in FOLDS:
        path = root / fold / subdir / "confusion_matrix_test.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = read_json(path)

        if labels is None:
            labels = list(payload["labels"])
            matrix_sum = [[0 for _ in labels] for _ in labels]

        for row_idx, row in enumerate(payload["matrix"]):
            for col_idx, value in enumerate(row):
                matrix_sum[row_idx][col_idx] += int(value)

    if labels is None or matrix_sum is None:
        raise RuntimeError("no confusion matrices loaded")
    return labels, matrix_sum


def class_f1_from_matrix(labels: list[str], matrix: list[list[int]]) -> dict[str, float]:
    scores = {}
    for idx, label in enumerate(labels):
        true_positive = matrix[idx][idx]
        support = sum(matrix[idx])
        predicted = sum(row[idx] for row in matrix)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        scores[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return scores


def plot_main_metrics(ax) -> None:
    x = list(range(len(METRICS)))
    width = 0.24
    offsets = [-width, 0, width]
    colors = ["#4C78A8", "#59A14F", "#E15759"]

    for setting_idx, (setting_name, root, subdir) in enumerate(SETTINGS):
        means = []
        stds = []
        for _, metric_key in METRICS:
            values = load_eval_values(root, subdir, metric_key)
            means.append(mean(values))
            stds.append(std(values))

        positions = [item + offsets[setting_idx] for item in x]
        ax.bar(
            positions,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=setting_name,
            color=colors[setting_idx],
            edgecolor="#222222",
            linewidth=0.5,
        )
        for position, value in zip(positions, means):
            ax.text(position, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x, [label for label, _ in METRICS])
    ax.set_ylim(0.48, 0.90)
    ax.set_ylabel("Score")
    ax.set_title("20-fold Overall Metrics")
    ax.legend(frameon=True, fontsize=9)


def plot_class_f1(ax) -> None:
    x = list(range(len(CLASS_LABELS)))
    width = 0.24
    offsets = [-width, 0, width]
    colors = ["#4C78A8", "#59A14F", "#E15759"]

    for setting_idx, (setting_name, root, subdir) in enumerate(SETTINGS):
        labels, matrix = sum_confusion_matrices(root, subdir)
        f1_scores = class_f1_from_matrix(labels, matrix)
        values = [f1_scores[label] for label in CLASS_LABELS]
        positions = [item + offsets[setting_idx] for item in x]
        ax.bar(
            positions,
            values,
            width=width,
            label=setting_name,
            color=colors[setting_idx],
            edgecolor="#222222",
            linewidth=0.5,
        )

    ax.set_xticks(x, CLASS_LABELS)
    ax.set_ylim(0.20, 1.00)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 from Aggregated Confusion Matrices")
    ax.legend(frameon=True, fontsize=9)


def main() -> None:
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=180)

    plot_main_metrics(axes[0])
    plot_class_f1(axes[1])

    fig.suptitle("DeepSleepNet EEG/EOG Robustness on Sleep-EDF", fontsize=15, fontweight="bold")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
