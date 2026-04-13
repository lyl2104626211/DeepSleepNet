from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DatasetConfig:
    name: str
    raw_dir: str
    processed_dir: str
    label_set: list[str]
    epoch_seconds: int
    split_mode: str


@dataclass
class ModelConfig:
    name: str


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    seed: int
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class EvaluationConfig:
    metrics: list[str]
    save_confusion_matrix: bool


@dataclass
class OutputConfig:
    result_dir: str


@dataclass
class ExperimentConfig:
    experiment_name: str
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    output: OutputConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少 PyYAML，请先安装依赖后再读取配置：uv sync"
        ) from exc

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式错误：{path}")
    return data


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = _read_yaml(config_path)
    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        dataset=DatasetConfig(**raw["dataset"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        output=OutputConfig(**raw["output"]),
    )
