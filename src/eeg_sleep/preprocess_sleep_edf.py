from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path


# Sleep-EDF 常见的睡眠标签描述。
# 这里把原始标注统一映射到 5 分类任务：
# W / N1 / N2 / N3 / REM
STAGE_LABEL_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}

# 这些标注通常不参与 5 分类训练。
IGNORED_LABELS = {
    "Sleep stage ?",
    "Movement time",
}

# DeepSleepNet 常见设置是固定 30 秒一个 epoch。
DEFAULT_EPOCH_SECONDS = 30


@dataclass
class RecordPair:
    """一组 PSG 和 Hypnogram 文件。"""

    subject_id: str
    psg_path: Path
    hypnogram_path: Path


@dataclass
class EpochSample:
    """单个 epoch 的元信息。"""

    subject_id: str
    epoch_index: int
    start_second: float
    label: str
    n_samples: int
    channel_name: str
    relative_data_path: str


def _lazy_import_dependencies():
    """延迟导入重依赖。

    这样做的好处是：
    1. 没装依赖时，用户仍然能读代码和查看帮助信息；
    2. 报错点会更明确，而不是一导入模块就直接失败。
    """

    try:
        import mne
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "运行预处理前请先安装依赖：uv sync"
        ) from exc

    return mne, np


def find_sleep_edf_pairs(input_dir: Path) -> list[RecordPair]:
    """在目录中自动查找 PSG / Hypnogram 配对文件。

    兼容两类常见命名：
    - 子集调试时的文件：
      - SC4201E0-PSG.edf
      - SC4201EC-Hypnogram.edf
    - 完整 sleep-cassette 中的文件：
      - SC4001E0-PSG.edf
      - SC4001EC-Hypnogram.edf
      - SC4501E0-PSG.edf
      - SC4501EW-Hypnogram.edf

    关键点是：
    - PSG 和 Hypnogram 并不是简单固定成 E0 / EC、F0 / FC
    - 但一组记录通常共享前 6 个字符，例如 `SC4201`
    """

    psg_files = sorted(input_dir.glob("*-PSG.edf"))
    pairs: list[RecordPair] = []

    for psg_path in psg_files:
        match = re.match(r"(?P<record_id>[A-Z]{2}\d{4})[A-Z0-9]{2}-PSG\.edf", psg_path.name)
        if match is None:
            continue

        subject_id = match.group("record_id")
        hypnogram_candidates = sorted(input_dir.glob(f"{subject_id}??-Hypnogram.edf"))
        if not hypnogram_candidates:
            continue

        # sleep-cassette 中同一组记录通常只对应 1 个 hypnogram 文件。
        # 若后续遇到多个候选，这里优先取排序后的第一个，至少不会直接中断。
        hypnogram_path = hypnogram_candidates[0]
        pairs.append(
            RecordPair(
                subject_id=subject_id,
                psg_path=psg_path,
                hypnogram_path=hypnogram_path,
            )
        )

    return pairs


def choose_eeg_channel(channel_names: list[str]) -> str:
    """优先选择 Sleep-EDF 中最常见的单通道 EEG。

    常见候选：
    - EEG Fpz-Cz
    - EEG Pz-Oz
    """

    preferred_channels = [
        "EEG Fpz-Cz",
        "EEG Pz-Oz",
    ]
    for channel in preferred_channels:
        if channel in channel_names:
            return channel

    raise ValueError(
        "未找到常见的单通道 EEG 通道。当前通道列表为："
        + ", ".join(channel_names)
    )


def build_epoch_labels(
    annotations,
    epoch_seconds: int,
    max_second: float | None = None,
) -> list[tuple[float, str]]:
    """把原始区间标注展开成 epoch 级标签。

    返回值中的每一项为：
    - epoch 起始秒数
    - 对应的 5 分类标签

    这里的思路很重要：
    原始 Hypnogram 标注通常是一段时间对应一个睡眠阶段，
    而模型训练需要的是固定长度 epoch 的标签，所以需要把区间展开。

    max_second 用来限制 epoch 不能超过 PSG 的真实结束时间。
    这样可以避免标注文件末尾略长于信号时，crop 直接报错。
    """

    epoch_labels: list[tuple[float, str]] = []

    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
    ):
        if description in IGNORED_LABELS:
            continue

        if description not in STAGE_LABEL_MAP:
            continue

        mapped_label = STAGE_LABEL_MAP[description]
        epoch_count = int(duration // epoch_seconds)

        for offset_idx in range(epoch_count):
            epoch_start = float(onset + offset_idx * epoch_seconds)
            epoch_end = epoch_start + epoch_seconds
            if max_second is not None and epoch_end > max_second:
                # 末尾越界的 epoch 直接丢弃，避免超过真实信号范围。
                continue
            epoch_labels.append((epoch_start, mapped_label))

    return epoch_labels


def extract_epoch_array(raw, channel_name: str, start_second: float, epoch_seconds: int):
    """从指定时间范围裁剪出一个 epoch 的单通道 EEG 数组。"""

    segment = raw.copy().pick([channel_name]).crop(
        tmin=start_second,
        tmax=start_second + epoch_seconds - 1.0 / raw.info["sfreq"],
    )
    data = segment.get_data()[0]
    return data


def process_record_pair(
    pair: RecordPair,
    output_dir: Path,
    epoch_seconds: int,
) -> tuple[list[EpochSample], Counter]:
    """处理单组 PSG/Hypnogram，输出 epoch 数据和元信息。"""

    mne, np = _lazy_import_dependencies()

    raw = mne.io.read_raw_edf(pair.psg_path, preload=False, verbose="ERROR")
    annotations = mne.read_annotations(pair.hypnogram_path)
    channel_name = choose_eeg_channel(raw.ch_names)
    max_second = float(raw.times[-1] + 1.0 / raw.info["sfreq"])

    subject_output_dir = output_dir / pair.subject_id
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    epoch_labels = build_epoch_labels(
        annotations=annotations,
        epoch_seconds=epoch_seconds,
        max_second=max_second,
    )
    label_counter: Counter = Counter()
    epoch_samples: list[EpochSample] = []

    for epoch_index, (start_second, label) in enumerate(epoch_labels):
        data = extract_epoch_array(
            raw=raw,
            channel_name=channel_name,
            start_second=start_second,
            epoch_seconds=epoch_seconds,
        )

        file_name = f"epoch_{epoch_index:05d}_{label}.npy"
        save_path = subject_output_dir / file_name
        np.save(save_path, data.astype(np.float32))

        epoch_samples.append(
            EpochSample(
                subject_id=pair.subject_id,
                epoch_index=epoch_index,
                start_second=start_second,
                label=label,
                n_samples=int(data.shape[0]),
                channel_name=channel_name,
                # manifest 中统一使用 /，避免 Windows 生成的数据到 Linux 上无法直接读取。
                relative_data_path=save_path.relative_to(output_dir.parent).as_posix(),
            )
        )
        label_counter[label] += 1

    return epoch_samples, label_counter


def save_manifest(samples: list[EpochSample], output_dir: Path) -> Path:
    """保存所有 epoch 的索引文件。

    后续写 PyTorch Dataset 时，可以直接读这个 manifest，
    不需要重新扫描目录。
    """

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump([asdict(sample) for sample in samples], file, ensure_ascii=False, indent=2)
    return manifest_path


def save_summary(all_label_counts: Counter, output_dir: Path) -> Path:
    """保存标签统计信息，方便先检查类别分布是否合理。"""

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(dict(all_label_counts), file, ensure_ascii=False, indent=2)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sleep-EDF 子集预处理脚本")
    parser.add_argument(
        "--input-dir",
        default="data/raw/sleep_edf_subset",
        help="原始 Sleep-EDF 子集目录",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/sleep_edf_subset",
        help="处理后数据输出目录",
    )
    parser.add_argument(
        "--epoch-seconds",
        type=int,
        default=DEFAULT_EPOCH_SECONDS,
        help="epoch 时长，默认 30 秒",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    record_pairs = find_sleep_edf_pairs(input_dir)
    if not record_pairs:
        raise FileNotFoundError(f"未在目录中找到 PSG/Hypnogram 配对文件：{input_dir}")

    all_samples: list[EpochSample] = []
    all_label_counts: Counter = Counter()

    print(f"找到 {len(record_pairs)} 组记录，开始预处理。")

    for pair in record_pairs:
        print(f"处理被试：{pair.subject_id}")
        samples, label_counts = process_record_pair(
            pair=pair,
            output_dir=output_dir,
            epoch_seconds=args.epoch_seconds,
        )
        all_samples.extend(samples)
        all_label_counts.update(label_counts)
        print(f"- 生成 {len(samples)} 个 epoch")
        print(f"- 标签分布：{dict(label_counts)}")

    manifest_path = save_manifest(all_samples, output_dir)
    summary_path = save_summary(all_label_counts, output_dir)

    print("预处理完成。")
    print(f"- manifest：{manifest_path}")
    print(f"- summary：{summary_path}")
    print(f"- 总 epoch 数：{len(all_samples)}")


if __name__ == "__main__":
    main()
