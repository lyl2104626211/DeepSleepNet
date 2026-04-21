from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricResult:
    accuracy: float
    macro_f1: float
    cohen_kappa: float


def _require_sklearn_metrics():
    try:
        from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少 scikit-learn，请先安装依赖：pip install scikit-learn") from exc

    return accuracy_score, cohen_kappa_score, confusion_matrix, f1_score


def compute_classification_metrics(y_true: list[int], y_pred: list[int]) -> MetricResult:
    """分类任务只保留当前项目真正会用到的 3 个指标。"""

    accuracy_score, cohen_kappa_score, _, f1_score = _require_sklearn_metrics()
    return MetricResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        cohen_kappa=float(cohen_kappa_score(y_true, y_pred)),
    )


def compute_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    labels: list[int] | None = None,
) -> list[list[int]]:
    _, _, confusion_matrix, _ = _require_sklearn_metrics()
    return confusion_matrix(y_true, y_pred, labels=labels).tolist()
