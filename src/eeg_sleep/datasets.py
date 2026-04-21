from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DatasetConfig


@dataclass
class DatasetSummary:
    dataset_name: str
    raw_dir: Path
    processed_dir: Path
    label_set: list[str]
    epoch_seconds: int
    split_mode: str


class SleepEDFDatasetBuilder:
    """这里只做最基本的数据目录检查和流程说明。"""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.raw_dir = Path(config.raw_dir)
        self.processed_dir = Path(config.processed_dir)

    def summarize(self) -> DatasetSummary:
        return DatasetSummary(
            dataset_name=self.config.name,
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
            label_set=self.config.label_set,
            epoch_seconds=self.config.epoch_seconds,
            split_mode=self.config.split_mode,
        )

    def validate_layout(self) -> list[str]:
        issues: list[str] = []

        if not self.raw_dir.exists():
            issues.append(f"原始数据目录不存在：{self.raw_dir}")
        if not self.processed_dir.exists():
            issues.append(f"处理后数据目录不存在：{self.processed_dir}")
        if self.config.epoch_seconds != 30:
            issues.append("当前 epoch_seconds 不是 30，请确认是否符合 DeepSleepNet 设置")
        if self.config.split_mode != "cross_subject":
            issues.append("当前 split_mode 不是 cross_subject，请确认是否符合实验目标")

        return issues

    def planned_steps(self) -> list[str]:
        return [
            "准备 Sleep-EDF 原始 PSG 和标注文件",
            "选择单通道 EEG",
            "按 30 秒切成 epoch",
            "把原始标注映射到 W/N1/N2/N3/REM",
            "按被试切 train/val/test",
            "保存 manifest 和 split 文件",
        ]
