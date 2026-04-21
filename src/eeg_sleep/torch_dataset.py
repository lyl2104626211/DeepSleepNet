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

ID_TO_LABEL = {label_id: label_name for label_name, label_id in LABEL_TO_ID.items()}


def infer_participant_id(subject_id: str) -> str:
    """Sleep-EDF 里通常同一受试者的两晚记录只在最后一位不同。"""

    if len(subject_id) >= 2 and subject_id[:2] in {"SC", "ST"} and subject_id[-1].isdigit():
        return subject_id[:-1]
    return subject_id

@dataclass(frozen=True)
class SampleRecord:
    """manifest.json 中一条样本记录。"""

    subject_id: str
    epoch_index: int
    start_second: float
    label: str
    n_samples: int
    channel_name: str
    relative_data_path: str
    participant_id: str | None = None


@dataclass(frozen=True)
class SubjectSplit:
    """按被试划分后的 train/val/test。"""

    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]


def _require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("使用数据集前请先安装依赖：uv sync") from exc
    return np


def _require_torch_data():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise RuntimeError("构建 DataLoader 前请先安装依赖：uv sync") from exc
    return torch, DataLoader


def _read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _filter_records(records: list[SampleRecord], subject_ids: Iterable[str] | None) -> list[SampleRecord]:
 
    if subject_ids is None:
        return records
    allowed_subjects = set(subject_ids)
    return [record for record in records if record.subject_id in allowed_subjects]


def _resolve_data_path(data_root: Path, relative_data_path: str) -> Path:
    normalized_parts = relative_data_path.replace("\\", "/").split("/")
    return data_root.joinpath(*normalized_parts)


def _load_signal(np_module, data_root: Path, record: SampleRecord):
    data_path = _resolve_data_path(data_root, record.relative_data_path)
    return np_module.load(data_path).astype(np_module.float32)


def load_manifest(manifest_path: str | Path) -> list[SampleRecord]:
    """读取预处理生成的 manifest。"""

    records: list[SampleRecord] = []
    for record in _read_json(manifest_path):
        if "participant_id" not in record or record["participant_id"] is None:
            record["participant_id"] = infer_participant_id(record["subject_id"])
        records.append(SampleRecord(**record))
    return records


def group_records_by_subject(records: Iterable[SampleRecord]) -> dict[str, list[SampleRecord]]:
    """按被试分组，并按 epoch 顺序排好。"""

    grouped: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        grouped[record.subject_id].append(record)

    return {
        subject_id: sorted(subject_records, key=lambda item: item.epoch_index)
        for subject_id, subject_records in sorted(grouped.items())
    }


def infer_sampling_rate(records: Iterable[SampleRecord], epoch_seconds: int = 30) -> int:
    first_record = next(iter(records), None)
    if first_record is None:
        raise ValueError("records 为空，无法估计采样率")
    return int(round(first_record.n_samples / epoch_seconds))


def _build_subject_groups(records: Iterable[SampleRecord], group_by: str) -> dict[str, list[str]]:
    if group_by not in {"subject", "participant"}:
        raise ValueError("group_by 只支持 subject 或 participant")

    grouped_subjects: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_id = record.subject_id if group_by == "subject" else (record.participant_id or infer_participant_id(record.subject_id))
        grouped_subjects[group_id].add(record.subject_id)

    return {
        group_id: sorted(subject_ids)
        for group_id, subject_ids in sorted(grouped_subjects.items())
    }


def _expand_group_ids(subject_groups: dict[str, list[str]], group_ids: list[str]) -> list[str]:
    subject_ids: list[str] = []
    for group_id in group_ids:
        subject_ids.extend(subject_groups[group_id])
    return sorted(subject_ids)


def split_subjects(
    records: Iterable[SampleRecord],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    group_by: str = "subject",
) -> SubjectSplit:
    """按被试或参与者切分，避免同一个人同时出现在不同集合。"""

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1")

    subject_groups = _build_subject_groups(records, group_by)
    group_ids = sorted(subject_groups)
    if not group_ids:
        raise ValueError("records 为空，无法按被试切分")

    rng = random.Random(seed)
    rng.shuffle(group_ids)

    # 子集调试时被试很少，优先保证 train 一定存在。
    if len(group_ids) == 1:
        return SubjectSplit(train_subjects=_expand_group_ids(subject_groups, group_ids), val_subjects=[], test_subjects=[])
    if len(group_ids) == 2:
        return SubjectSplit(
            train_subjects=_expand_group_ids(subject_groups, group_ids[:1]),
            val_subjects=_expand_group_ids(subject_groups, group_ids[1:]),
            test_subjects=[],
        )
    if len(group_ids) == 3:
        return SubjectSplit(
            train_subjects=_expand_group_ids(subject_groups, group_ids[:1]),
            val_subjects=_expand_group_ids(subject_groups, group_ids[1:2]),
            test_subjects=_expand_group_ids(subject_groups, group_ids[2:]),
        )

    counts = [
        int(len(group_ids) * train_ratio),
        int(len(group_ids) * val_ratio),
        int(len(group_ids) * test_ratio),
    ]
    remainders = [
        len(group_ids) * train_ratio - counts[0],
        len(group_ids) * val_ratio - counts[1],
        len(group_ids) * test_ratio - counts[2],
    ]

    while sum(counts) < len(group_ids):
        bucket_idx = max(range(3), key=lambda index: remainders[index])
        counts[bucket_idx] += 1
        remainders[bucket_idx] = 0.0

    train_end = counts[0]
    val_end = counts[0] + counts[1]
    return SubjectSplit(
        train_subjects=_expand_group_ids(subject_groups, group_ids[:train_end]),
        val_subjects=_expand_group_ids(subject_groups, group_ids[train_end:val_end]),
        test_subjects=_expand_group_ids(subject_groups, group_ids[val_end:]),
    )


def build_kfold_split(
    records: Iterable[SampleRecord],
    n_folds: int,
    fold_index: int,
    seed: int = 42,
    group_by: str = "subject",
) -> SubjectSplit:
    """构建最直接的 k-fold 划分：当前 fold 做 test，其余全做 train。"""

    if n_folds < 2:
        raise ValueError("n_folds 必须大于等于 2")
    if not 0 <= fold_index < n_folds:
        raise ValueError("fold_index 超出范围")

    subject_groups = _build_subject_groups(records, group_by)
    group_ids = sorted(subject_groups)
    if len(group_ids) < n_folds:
        raise ValueError("分组数量少于 n_folds，无法构建 k-fold 划分")

    rng = random.Random(seed)
    rng.shuffle(group_ids)

    folds = [[] for _ in range(n_folds)]
    for index, group_id in enumerate(group_ids):
        folds[index % n_folds].append(group_id)

    test_group_ids = sorted(folds[fold_index])
    train_group_ids = sorted(
        group_id
        for current_fold_index, fold_group_ids in enumerate(folds)
        if current_fold_index != fold_index
        for group_id in fold_group_ids
    )
    return SubjectSplit(
        train_subjects=_expand_group_ids(subject_groups, train_group_ids),
        val_subjects=[],
        test_subjects=_expand_group_ids(subject_groups, test_group_ids),
    )


def save_subject_split(split: SubjectSplit, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(asdict(split), file, ensure_ascii=False, indent=2)
    return path


def load_subject_split(path: str | Path) -> SubjectSplit:
    return SubjectSplit(**_read_json(path))


class SleepEDFEpochDataset:
    """最直接的单 epoch 数据集封装。"""

    def __init__(
        self,
        manifest_path: str | Path,
        subject_ids: Iterable[str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.data_root = self.manifest_path.parent.parent
        self.np = _require_numpy()
        self.records = _filter_records(load_manifest(self.manifest_path), subject_ids)

        if not self.records:
            raise ValueError("筛选后的 epoch 数据集为空，请检查 manifest 或 subject_ids")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        signal = _load_signal(self.np, self.data_root, record)
        return {
            "signal": signal,
            "label": LABEL_TO_ID[record.label],
            "label_name": record.label,
            "subject_id": record.subject_id,
            "epoch_index": record.epoch_index,
        }


class SleepEDFSequenceDataset:
    """把连续 epoch 组织成固定长度窗口。"""

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

        records = _filter_records(load_manifest(self.manifest_path), subject_ids)
        if not records:
            raise ValueError("筛选后的序列数据集为空，请检查 manifest 或 subject_ids")

        self.subject_records = group_records_by_subject(records)
        self.sequences: list[list[SampleRecord]] = []

        for subject_records in self.subject_records.values():
            if len(subject_records) < self.sequence_length:
                continue

            last_start = len(subject_records) - self.sequence_length + 1
            for start_idx in range(0, last_start, self.stride):
                window = subject_records[start_idx:start_idx + self.sequence_length]
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

    def __getitem__(self, index: int) -> dict:
        window = self.sequences[index]
        signals = self.np.stack(
            [_load_signal(self.np, self.data_root, record) for record in window],
            axis=0,
        )
        labels = self.np.asarray(
            [LABEL_TO_ID[record.label] for record in window],
            dtype=self.np.int64,
        )
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
    """单 epoch DataLoader。"""

    torch, DataLoader = _require_torch_data()

    def collate_fn(batch: list[dict]) -> dict:
        return {
            "signals": torch.stack([torch.from_numpy(item["signal"]) for item in batch], dim=0),
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "subject_ids": [item["subject_id"] for item in batch],
            "epoch_indices": [item["epoch_index"] for item in batch],
            "label_names": [item["label_name"] for item in batch],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
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
    """序列 DataLoader。"""

    torch, DataLoader = _require_torch_data()

    def collate_fn(batch: list[dict]) -> dict:
        return {
            "signals": torch.stack([torch.from_numpy(item["signals"]) for item in batch], dim=0),
            "labels": torch.stack([torch.from_numpy(item["labels"]) for item in batch], dim=0),
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
