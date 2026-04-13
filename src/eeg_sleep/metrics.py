from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricResult:
    accuracy: float
    macro_f1: float
    cohen_kappa: float


def compute_classification_metrics(y_true: list[int], y_pred: list[int]) -> MetricResult:
    try:
        from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少 scikit-learn，请先安装依赖：pip install scikit-learn"
        ) from exc

    return MetricResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        # 小子集调试时验证集可能缺少某些类别，zero_division=0 可以避免不必要的告警。
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        cohen_kappa=float(cohen_kappa_score(y_true, y_pred)),
    )
