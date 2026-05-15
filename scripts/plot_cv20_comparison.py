from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "deepsleepnet_sleep_edf_paper_fpz_cz" / "cv20_summary.csv"
OUT_DIR = ROOT / "docs" / "figures"
OUT_PATH = OUT_DIR / "deepsleepnet_cv20_comparison.png"


def load_rows(csv_path: Path) -> list[dict[str, float | str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(
                {
                    "fold": row["fold"],
                    "accuracy": float(row["accuracy"]),
                    "macro_f1": float(row["macro_f1"]),
                    "cohen_kappa": float(row["cohen_kappa"]),
                    "loss": float(row["loss"]),
                }
            )
    return rows


def main() -> None:
    rows = load_rows(CSV_PATH)
    if not rows:
        raise RuntimeError(f"No rows found in {CSV_PATH}")

    acc = np.array([row["accuracy"] for row in rows], dtype=float)
    f1 = np.array([row["macro_f1"] for row in rows], dtype=float)
    kappa = np.array([row["cohen_kappa"] for row in rows], dtype=float)
    folds = [str(row["fold"]) for row in rows]

    repro_mean = np.array([acc.mean(), f1.mean(), kappa.mean()], dtype=float)
    repro_std = np.array([acc.std(), f1.std(), kappa.std()], dtype=float)

    # Reference values noted in the repo docs for the original DeepSleepNet paper.
    paper = np.array([0.8200, 0.7690, 0.7600], dtype=float)
    labels = ["Accuracy", "Macro-F1", "Kappa"]

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), dpi=160)

    ax = axes[0]
    x = np.arange(len(labels))
    width = 0.34
    paper_color = "#355C7D"
    repro_color = "#C06C84"

    ax.bar(x - width / 2, paper, width=width, color=paper_color, label="Paper")
    ax.bar(
        x + width / 2,
        repro_mean,
        width=width,
        color=repro_color,
        yerr=repro_std,
        capsize=5,
        label="Reproduction",
    )

    for idx, value in enumerate(paper):
        ax.text(idx - width / 2, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(repro_mean):
        ax.text(idx + width / 2, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x, labels)
    ax.set_ylim(0.65, 0.92)
    ax.set_ylabel("Score")
    ax.set_title("Paper vs Reproduction")
    ax.legend(frameon=True)

    ax2 = axes[1]
    ax2.plot(folds, acc, color="#2A9D8F", marker="o", linewidth=2, markersize=4, label="Fold accuracy")
    ax2.axhline(acc.mean(), color="#E76F51", linestyle="--", linewidth=2, label=f"Mean = {acc.mean():.3f}")
    ax2.fill_between(
        range(len(folds)),
        acc.mean() - acc.std(),
        acc.mean() + acc.std(),
        color="#F4A261",
        alpha=0.22,
        label=f"Std = {acc.std():.3f}",
    )
    ax2.set_xticks(range(len(folds)))
    ax2.set_xticklabels(folds, rotation=45, ha="right")
    ax2.set_ylim(0.72, 0.92)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("20-Fold Accuracy Distribution")
    ax2.legend(frameon=True)

    fig.suptitle("DeepSleepNet on Sleep-EDF Fpz-Cz", fontsize=15, fontweight="bold")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
