from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


LABEL_TO_ID = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
}

ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class SampleRecord:
    """manifest.json 中一条样本记录的结构。"""

    subject_id: str
    epoch_index: int
    start_second: float
    label: str
    n_samples: int
    channel_name: str
    relative_data_path: str


@dataclass(frozen=True)
class SubjectSplit:
    """按被试划分的数据集切分结果。"""

    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]


def _require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "使用 Dataset 前请先安装依赖：uv sync"
        ) from exc
    return np


def load_manifest(manifest_path: str | Path) -> list[SampleRecord]:
    """读取预处理生成的 manifest 文件。"""

    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as file:
        raw_records = json.load(file)

    return [SampleRecord(**record) for record in raw_records]


def group_records_by_subject(records: Iterable[SampleRecord]) -> dict[str, list[SampleRecord]]:
    """按被试整理样本，并按 epoch 索引排序。"""

    grouped: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        grouped[record.subject_id].append(record)

    return {
        subject_id: sorted(subject_records, key=lambda item: item.epoch_index)
        for subject_id, subject_records in sorted(grouped.items())
    }


def infer_sampling_rate(records: Iterable[SampleRecord], epoch_seconds: int = 30) -> int:
    """根据样本长度估计采样率。"""

    first_record = next(iter(records), None)
    if first_record is None:
        raise ValueError("records 为空，无法估计采样率")
    return int(round(first_record.n_samples / epoch_seconds))


def split_subjects(
    records: Iterable[SampleRecord],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> SubjectSplit:
    """按被试划分 train / val / test。

    这里的切分单位是被试而不是 epoch，避免同一被试出现在不同集合。
    """

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1")

    subject_ids = sorted({record.subject_id for record in records})
    if not subject_ids:
        raise ValueError("records 为空，无法进行按被试切分")

    rng = random.Random(seed)
    rng.shuffle(subject_ids)

    # 子集调试时被试可能非常少。
    # 为了先跑通第一阶段训练，这里优先保证：
    # - 至少有 train
    # - 如果被试数 >= 2，则尽量保留一个 val
    # - 如果被试数 >= 3，则再保留一个 test
    if len(subject_ids) == 1:
        return SubjectSplit(
            train_subjects=sorted(subject_ids),
            val_subjects=[],
            test_subjects=[],
        )
    if len(subject_ids) == 2:
        return SubjectSplit(
            train_subjects=sorted(subject_ids[:1]),
            val_subjects=sorted(subject_ids[1:]),
            test_subjects=[],
        )
    if len(subject_ids) == 3:
        return SubjectSplit(
            train_subjects=sorted(subject_ids[:1]),
            val_subjects=sorted(subject_ids[1:2]),
            test_subjects=sorted(subject_ids[2:]),
        )

    counts = [
        int(len(subject_ids) * train_ratio),
        int(len(subject_ids) * val_ratio),
        int(len(subject_ids) * test_ratio),
    ]
    assigned = sum(counts)
    remainders = [
        (len(subject_ids) * train_ratio) - counts[0],
        (len(subject_ids) * val_ratio) - counts[1],
        (len(subject_ids) * test_ratio) - counts[2],
    ]
    while assigned < len(subject_ids):
        bucket_idx = max(range(3), key=lambda idx: remainders[idx])
        counts[bucket_idx] += 1
        remainders[bucket_idx] = 0.0
        assigned += 1

    train_end = counts[0]
    val_end = counts[0] + counts[1]
    return SubjectSplit(
        train_subjects=sorted(subject_ids[:train_end]),
        val_subjects=sorted(subject_ids[train_end:val_end]),
        test_subjects=sorted(subject_ids[val_end:]),
    )


def save_subject_split(split: SubjectSplit, output_path: str | Path) -> Path:
    """保存按被试划分结果，方便后续训练直接读取。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(asdict(split), file, ensure_ascii=False, indent=2)
    return path


def load_subject_split(path: str | Path) -> SubjectSplit:
    """读取已经保存的按被试划分结果。"""

    split_path = Path(path)
    with split_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    return SubjectSplit(**raw)


class SleepEDFEpochDataset:
    """最小可用的 Sleep-EDF 单 epoch 数据集封装。"""

    def __init__(
        self,
        manifest_path: str | Path,
        subject_ids: Iterable[str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.data_root = self.manifest_path.parent.parent
        self.np = _require_numpy()

        records = load_manifest(self.manifest_path)
        if subject_ids is not None:
            allowed_subjects = set(subject_ids)
            records = [record for record in records if record.subject_id in allowed_subjects]
        self.records = records

        if not self.records:
            raise ValueError("筛选后的 epoch 数据集为空，请检查 manifest 或 subject_ids")

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_data_path(self, relative_data_path: str) -> Path:
        """兼容 Windows 和 Linux 两种路径分隔符。"""

        normalized_parts = relative_data_path.replace("\\", "/").split("/")
        return self.data_root.joinpath(*normalized_parts)

    def _load_signal(self, record: SampleRecord):
        data_path = self._resolve_data_path(record.relative_data_path)
        return self.np.load(data_path).astype(self.np.float32)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        signal = self._load_signal(record)
        label_id = LABEL_TO_ID[record.label]

        return {
            "signal": signal,
            "label": label_id,
            "label_name": record.label,
            "subject_id": record.subject_id,
            "epoch_index": record.epoch_index,
        }


class SleepEDFSequenceDataset:
    """按被试和时间顺序组织的序列数据集。"""

    def __init__(
        self,
        manifest_path: str | Path,
        sequence_length: int = 25,
        stride: int | None = None,
        subject_ids: Iterable[str] | None = None,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length 必须大于 0")

        self.manifest_path = Path(manifest_path)
        self.data_root = self.manifest_path.parent.parent
        self.sequence_length = sequence_length
        self.stride = stride if stride is not None else sequence_length
        self.np = _require_numpy()

        records = load_manifest(self.manifest_path)
        if subject_ids is not None:
            allowed_subjects = set(subject_ids)
            records = [record for record in records if record.subject_id in allowed_subjects]
        if not records:
            raise ValueError("筛选后的序列数据集为空，请检查 manifest 或 subject_ids")

        self.subject_records = group_records_by_subject(records)
        self.sequences: list[list[SampleRecord]] = []

        for subject_id, subject_sequence in self.subject_records.items():
            if len(subject_sequence) < self.sequence_length:
                continue

            for start_idx in range(0, len(subject_sequence) - self.sequence_length + 1, self.stride):
                window = subject_sequence[start_idx:start_idx + self.sequence_length]
                if self._is_contiguous(window):
                    self.sequences.append(window)

        if not self.sequences:
            raise ValueError("没有可用的序列窗口，请减小 sequence_length 或检查数据完整性")

    @staticmethod
    def _is_contiguous(records: list[SampleRecord]) -> bool:
        return all(
            current.epoch_index + 1 == next_record.epoch_index
            for current, next_record in zip(records[:-1], records[1:])
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def _resolve_data_path(self, relative_data_path: str) -> Path:
        """兼容 Windows 和 Linux 两种路径分隔符。"""

        normalized_parts = relative_data_path.replace("\\", "/").split("/")
        return self.data_root.joinpath(*normalized_parts)

    def _load_signal(self, record: SampleRecord):
        data_path = self._resolve_data_path(record.relative_data_path)
        return self.np.load(data_path).astype(self.np.float32)

    def __getitem__(self, index: int) -> dict:
        window = self.sequences[index]
        signals = self.np.stack([self._load_signal(record) for record in window], axis=0)
        labels = self.np.asarray([LABEL_TO_ID[record.label] for record in window], dtype=self.np.int64)

        return {
            "signals": signals,
            "labels": labels,
            "label_names": [record.label for record in window],
            "subject_id": window[0].subject_id,
            "epoch_indices": [record.epoch_index for record in window],
        }


def create_dataloader(
    dataset: SleepEDFEpochDataset,
    batch_size: int = 8,
    shuffle: bool = False,
    sampler=None,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """构建单 epoch DataLoader。"""

    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "构建 DataLoader 前请先安装依赖：uv sync"
        ) from exc

    def collate_fn(batch: list[dict]) -> dict:
        signals = [torch.tensor(item["signal"]) for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        return {
            "signals": torch.stack(signals, dim=0),
            "labels": labels,
            "subject_ids": [item["subject_id"] for item in batch],
            "epoch_indices": [item["epoch_index"] for item in batch],
            "label_names": [item["label_name"] for item in batch],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        # sampler 和 shuffle 不能同时生效；若传入 sampler，则以 sampler 为准。
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )


def create_sequence_dataloader(
    dataset: SleepEDFSequenceDataset,
    batch_size: int = 2,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """构建序列 DataLoader。"""

    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "构建序列 DataLoader 前请先安装依赖：uv sync"
        ) from exc

    def collate_fn(batch: list[dict]) -> dict:
        signals = torch.stack([torch.tensor(item["signals"]) for item in batch], dim=0)
        labels = torch.stack([torch.tensor(item["labels"], dtype=torch.long) for item in batch], dim=0)

        return {
            "signals": signals,
            "labels": labels,
            "subject_ids": [item["subject_id"] for item in batch],
            "epoch_indices": [item["epoch_indices"] for item in batch],
            "label_names": [item["label_names"] for item in batch],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )
