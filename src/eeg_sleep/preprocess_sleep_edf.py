from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


# Sleep-EDF 常见 5 分类映射。
STAGE_LABEL_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}

IGNORED_LABELS = {
    "Sleep stage ?",
    "Movement time",
}

DEFAULT_EPOCH_SECONDS = 30


@dataclass
class RecordPair:
    """一组 PSG 和 Hypnogram 文件。"""

    subject_id: str
    psg_path: Path
    hypnogram_path: Path


@dataclass
class EpochSample:
    """一个 epoch 的元信息。"""

    subject_id: str
    epoch_index: int
    start_second: float
    label: str
    n_samples: int
    channel_name: str
    relative_data_path: str
    participant_id: str | None = None


def _lazy_import_dependencies():
    try:
        import mne
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("运行预处理前请先安装依赖：uv sync") from exc

    return mne, np


def find_sleep_edf_pairs(input_dir: Path) -> list[RecordPair]:
    """自动匹配同一条记录下的 PSG 和 Hypnogram。"""

    pairs: list[RecordPair] = []
    for psg_path in sorted(input_dir.glob("*-PSG.edf")):
        match = re.match(r"(?P<record_id>[A-Z]{2}\d{4})[A-Z0-9]{2}-PSG\.edf", psg_path.name)
        if match is None:
            continue

        subject_id = match.group("record_id")
        hypnogram_candidates = sorted(input_dir.glob(f"{subject_id}??-Hypnogram.edf"))
        if not hypnogram_candidates:
            continue

        pairs.append(
            RecordPair(
                subject_id=subject_id,
                psg_path=psg_path,
                hypnogram_path=hypnogram_candidates[0],
            )
        )

    return pairs


def infer_participant_id(subject_id: str) -> str:
    """Sleep-EDF 里同一受试者的两晚记录通常只在最后一位不同。"""

    if len(subject_id) >= 2 and subject_id[:2] in {"SC", "ST"} and subject_id[-1].isdigit():
        return subject_id[:-1]
    return subject_id


def _normalize_channel_name(channel_name: str) -> str:
    return channel_name if channel_name.startswith("EEG ") else f"EEG {channel_name}"


def choose_eeg_channel(channel_names: list[str], preferred_channel: str | None = None) -> str:
    """优先选择论文和官方数据里最常见的两个 EEG 通道。"""

    if preferred_channel is not None:
        normalized_channel = _normalize_channel_name(preferred_channel)
        if normalized_channel not in channel_names:
            raise ValueError("未找到指定 EEG 通道：" + normalized_channel)
        return normalized_channel

    for channel_name in ["EEG Fpz-Cz", "EEG Pz-Oz"]:
        if channel_name in channel_names:
            return channel_name

    raise ValueError("未找到常见单通道 EEG。当前通道列表：" + ", ".join(channel_names))


def build_epoch_labels(
    annotations,
    epoch_seconds: int,
    max_second: float | None = None,
) -> list[tuple[float, str]]:
    """把区间标注展开成固定长度 epoch 标注。"""

    epoch_labels: list[tuple[float, str]] = []

    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
    ):
        if description in IGNORED_LABELS or description not in STAGE_LABEL_MAP:
            continue

        label = STAGE_LABEL_MAP[description]
        epoch_count = int(duration // epoch_seconds)
        for offset_idx in range(epoch_count):
            epoch_start = float(onset + offset_idx * epoch_seconds)
            epoch_end = epoch_start + epoch_seconds
            if max_second is not None and epoch_end > max_second:
                continue
            epoch_labels.append((epoch_start, label))

    return epoch_labels


def trim_edge_wake_epochs(epoch_labels: list[tuple[float, str]], wake_minutes: int) -> list[tuple[float, str]]:
    """只保留睡眠段前后指定分钟数的清醒 W，内部的 W 不受影响。"""

    if wake_minutes <= 0 or not epoch_labels:
        return epoch_labels

    first_sleep_index = next((index for index, (_, label) in enumerate(epoch_labels) if label != "W"), None)
    if first_sleep_index is None:
        return epoch_labels

    last_sleep_index = max(index for index, (_, label) in enumerate(epoch_labels) if label != "W")
    wake_epochs = max(0, wake_minutes * 60 // DEFAULT_EPOCH_SECONDS)
    start_index = max(0, first_sleep_index - wake_epochs)
    end_index = min(len(epoch_labels), last_sleep_index + wake_epochs + 1)
    return epoch_labels[start_index:end_index]


def extract_epoch_array(raw, channel_name: str, start_second: float, epoch_seconds: int):
    """从原始 PSG 中裁出单个 epoch 的单通道 EEG。"""

    segment = raw.copy().pick([channel_name]).crop(
        tmin=start_second,
        tmax=start_second + epoch_seconds - 1.0 / raw.info["sfreq"],
    )
    return segment.get_data()[0]


def process_record_pair(
    pair: RecordPair,
    output_dir: Path,
    epoch_seconds: int,
    preferred_channel: str | None = None,
    trim_wake_minutes: int = 0,
) -> tuple[list[EpochSample], Counter]:
    """处理一组 PSG/Hypnogram，保存 epoch 并返回索引信息。"""

    mne, np = _lazy_import_dependencies()

    raw = mne.io.read_raw_edf(pair.psg_path, preload=False, verbose="ERROR")
    annotations = mne.read_annotations(pair.hypnogram_path)
    channel_name = choose_eeg_channel(raw.ch_names, preferred_channel=preferred_channel)
    max_second = float(raw.times[-1] + 1.0 / raw.info["sfreq"])
    participant_id = infer_participant_id(pair.subject_id)

    subject_output_dir = output_dir / pair.subject_id
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    epoch_samples: list[EpochSample] = []
    label_counter: Counter = Counter()

    epoch_labels = build_epoch_labels(annotations, epoch_seconds, max_second=max_second)
    epoch_labels = trim_edge_wake_epochs(epoch_labels, trim_wake_minutes)

    for epoch_index, (start_second, label) in enumerate(epoch_labels):
        data = extract_epoch_array(raw, channel_name, start_second, epoch_seconds)
        save_path = subject_output_dir / f"epoch_{epoch_index:05d}_{label}.npy"
        np.save(save_path, data.astype(np.float32))

        epoch_samples.append(
            EpochSample(
                subject_id=pair.subject_id,
                epoch_index=epoch_index,
                start_second=start_second,
                label=label,
                n_samples=int(data.shape[0]),
                channel_name=channel_name,
                # manifest 里统一使用 /，避免跨平台路径问题。
                relative_data_path=save_path.relative_to(output_dir.parent).as_posix(),
                participant_id=participant_id,
            )
        )
        label_counter[label] += 1

    return epoch_samples, label_counter


def save_manifest(samples: list[EpochSample], output_dir: Path) -> Path:
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump([asdict(sample) for sample in samples], file, ensure_ascii=False, indent=2)
    return manifest_path


def save_summary(all_label_counts: Counter, output_dir: Path) -> Path:
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(dict(all_label_counts), file, ensure_ascii=False, indent=2)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sleep-EDF 预处理脚本")
    parser.add_argument("--input-dir", default="data/raw/sleep_edf_subset", help="原始 Sleep-EDF 目录")
    parser.add_argument("--output-dir", default="data/processed/sleep_edf_subset", help="预处理输出目录")
    parser.add_argument("--epoch-seconds", type=int, default=DEFAULT_EPOCH_SECONDS, help="epoch 长度，默认 30 秒")
    parser.add_argument("--channel", choices=["Fpz-Cz", "Pz-Oz"], default=None, help="显式指定 EEG 通道")
    parser.add_argument("--trim-wake-minutes", type=int, default=0, help="仅保留睡眠前后多少分钟的清醒 W")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    record_pairs = find_sleep_edf_pairs(input_dir)
    if not record_pairs:
        raise FileNotFoundError(f"未在目录中找到 PSG/Hypnogram 配对文件：{input_dir}")

    all_samples: list[EpochSample] = []
    all_label_counts: Counter = Counter()

    print(f"找到 {len(record_pairs)} 组记录，开始预处理")
    for pair in record_pairs:
        print(f"处理被试：{pair.subject_id}")
        samples, label_counts = process_record_pair(
            pair,
            output_dir,
            args.epoch_seconds,
            preferred_channel=args.channel,
            trim_wake_minutes=args.trim_wake_minutes,
        )
        all_samples.extend(samples)
        all_label_counts.update(label_counts)
        print(f"- 生成 epoch 数：{len(samples)}")
        print(f"- 标签分布：{dict(label_counts)}")

    manifest_path = save_manifest(all_samples, output_dir)
    summary_path = save_summary(all_label_counts, output_dir)

    print("预处理完成")
    print(f"- manifest: {manifest_path}")
    print(f"- summary: {summary_path}")
    print(f"- 总 epoch 数: {len(all_samples)}")


if __name__ == "__main__":
    main()
