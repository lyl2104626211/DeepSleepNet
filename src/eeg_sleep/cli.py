from __future__ import annotations

import argparse

from .config import load_experiment_config
from .trainer import create_training_plan, save_training_plan, train_stage1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EEG 睡眠分期复现入口")
    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser("plan", help="生成当前实验的训练计划")
    plan_parser.add_argument(
        "--config",
        default="configs/base_experiment.yaml",
        help="实验配置文件路径",
    )

    preprocess_parser = subparsers.add_parser("preprocess", help="运行 Sleep-EDF 预处理")
    preprocess_parser.add_argument(
        "--input-dir",
        default="data/raw/sleep_edf_subset",
        help="原始 Sleep-EDF 子集目录",
    )
    preprocess_parser.add_argument(
        "--output-dir",
        default="data/processed/sleep_edf_subset",
        help="处理后数据输出目录",
    )
    preprocess_parser.add_argument(
        "--epoch-seconds",
        type=int,
        default=30,
        help="epoch 时长，默认 30 秒",
    )

    download_parser = subparsers.add_parser("download-sleep-edf", help="从 PhysioNet 下载 Sleep-EDF sleep-cassette")
    download_parser.add_argument(
        "--output-dir",
        default="data/raw/sleep_edf_sleep_cassette",
        help="原始 EDF 下载目录",
    )
    download_parser.add_argument(
        "--base-url",
        default="https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/",
        help="PhysioNet 目录地址，用于抓取文件列表",
    )
    download_parser.add_argument(
        "--download-base-url",
        default="https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/sleep-cassette/",
        help="实际文件下载地址，默认使用官方 AWS 桶",
    )
    download_parser.add_argument(
        "--record-prefix",
        default="SC",
        help="记录前缀，默认下载 sleep-cassette 的 SC 记录",
    )
    download_parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="最多下载多少组记录；0 表示下载全部",
    )
    download_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若文件已存在，是否覆盖下载",
    )
    download_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印下载计划，不真正下载",
    )

    inspect_parser = subparsers.add_parser("inspect-dataset", help="检查预处理后的数据集")
    inspect_parser.add_argument(
        "--manifest",
        default="data/processed/sleep_edf_subset/manifest.json",
        help="manifest.json 路径",
    )
    inspect_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="DataLoader 的 batch size",
    )

    split_parser = subparsers.add_parser("split-subjects", help="按被试划分 train / val / test")
    split_parser.add_argument(
        "--manifest",
        default="data/processed/sleep_edf_subset/manifest.json",
        help="manifest.json 路径",
    )
    split_parser.add_argument(
        "--output",
        default="data/processed/sleep_edf_subset/split.json",
        help="划分结果保存路径",
    )
    split_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集被试比例",
    )
    split_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="验证集被试比例",
    )
    split_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="测试集被试比例",
    )
    split_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="按被试划分的随机种子",
    )

    train_stage1_parser = subparsers.add_parser("train-stage1", help="训练第一阶段 DeepFeatureNet")
    train_stage1_parser.add_argument(
        "--config",
        default="configs/base_experiment.yaml",
        help="实验配置文件路径",
    )
    train_stage1_parser.add_argument(
        "--manifest",
        default="data/processed/sleep_edf_subset/manifest.json",
        help="manifest.json 路径",
    )
    train_stage1_parser.add_argument(
        "--split",
        default="data/processed/sleep_edf_subset/split.json",
        help="按被试划分结果路径",
    )
    train_stage1_parser.add_argument(
        "--output-dir",
        default="results/deepsleepnet_baseline/stage1",
        help="第一阶段训练结果目录",
    )
    train_stage1_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="可选：覆盖配置文件中的 epochs",
    )

    inspect_model_parser = subparsers.add_parser("inspect-model", help="检查模型前向和序列输入")
    inspect_model_parser.add_argument(
        "--manifest",
        default="data/processed/sleep_edf_subset/manifest.json",
        help="manifest.json 路径",
    )
    inspect_model_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="单 epoch DataLoader 的 batch size",
    )
    inspect_model_parser.add_argument(
        "--sequence-batch-size",
        type=int,
        default=2,
        help="序列 DataLoader 的 batch size",
    )
    inspect_model_parser.add_argument(
        "--sequence-length",
        type=int,
        default=25,
        help="序列长度，默认 25",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in (None, "plan"):
        config = load_experiment_config(args.config)
        plan = create_training_plan(config)
        plan_file = save_training_plan(plan, config.output.result_dir)

        print(f"实验名称：{plan.experiment_name}")
        print(f"计划文件：{plan_file}")
        print("数据检查：")
        if plan.dataset_checks:
            for issue in plan.dataset_checks:
                print(f"- {issue}")
        else:
            print("- 目录检查通过")

        print("后续步骤：")
        for step in plan.next_actions:
            print(f"- {step}")
        return

    if args.command == "preprocess":
        from .preprocess_sleep_edf import main as preprocess_main
        import sys

        sys.argv = [
            "preprocess",
            "--input-dir",
            args.input_dir,
            "--output-dir",
            args.output_dir,
            "--epoch-seconds",
            str(args.epoch_seconds),
        ]
        preprocess_main()
        return

    if args.command == "download-sleep-edf":
        from .download_sleep_edf import download_sleep_cassette

        plan = download_sleep_cassette(
            output_dir=args.output_dir,
            base_url=args.base_url,
            download_base_url=args.download_base_url,
            record_prefix=args.record_prefix,
            max_records=args.max_records,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

        print(f"计划记录数：{len({item.record_id for item in plan})}")
        print(f"计划文件数：{len(plan)}")
        if plan:
            print("前几个文件：")
            for item in plan[: min(6, len(plan))]:
                print(f"- {item.file_name}")
        if args.dry_run:
            print("当前为 dry-run，没有实际下载文件。")
        else:
            print(f"下载目录：{args.output_dir}")
            print(f"文件下载源：{args.download_base_url}")
        return

    if args.command == "inspect-dataset":
        from .torch_dataset import SleepEDFEpochDataset, create_dataloader

        dataset = SleepEDFEpochDataset(args.manifest)
        print(f"样本总数：{len(dataset)}")

        first_sample = dataset[0]
        print("第一个样本：")
        print(f"- 被试：{first_sample['subject_id']}")
        print(f"- epoch 索引：{first_sample['epoch_index']}")
        print(f"- 标签：{first_sample['label_name']} ({first_sample['label']})")
        print(f"- 信号长度：{len(first_sample['signal'])}")

        dataloader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
        first_batch = next(iter(dataloader))
        print("第一个 batch：")
        print(f"- signals shape：{tuple(first_batch['signals'].shape)}")
        print(f"- labels shape：{tuple(first_batch['labels'].shape)}")
        print(f"- labels：{first_batch['label_names']}")
        return

    if args.command == "split-subjects":
        from .torch_dataset import load_manifest, save_subject_split, split_subjects

        manifest_records = load_manifest(args.manifest)
        split = split_subjects(
            manifest_records,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

        split_path = save_subject_split(split, args.output)

        print(f"划分文件：{split_path}")
        print(f"- train subjects ({len(split.train_subjects)}): {split.train_subjects}")
        print(f"- val subjects ({len(split.val_subjects)}): {split.val_subjects}")
        print(f"- test subjects ({len(split.test_subjects)}): {split.test_subjects}")

        if len(split.train_subjects) == 1 and len(split.val_subjects) == 1 and len(split.test_subjects) == 0:
            print("说明：当前只有 2 个被试，已自动采用 smoke test 划分：1 个 train、1 个 val、0 个 test。")
        return

    if args.command == "train-stage1":
        config = load_experiment_config(args.config)
        summary, summary_path = train_stage1(
            config=config,
            manifest_path=args.manifest,
            split_path=args.split,
            output_dir=args.output_dir,
            epochs_override=args.epochs,
        )

        print("第一阶段训练完成：")
        print(f"- device：{summary.device}")
        print(f"- train subjects：{summary.train_subjects}")
        print(f"- val subjects：{summary.val_subjects}")
        print(f"- train size：{summary.train_size}")
        print(f"- val size：{summary.val_size}")
        print(f"- best epoch：{summary.best_epoch}")
        print(f"- best val macro_f1：{summary.best_val_macro_f1}")
        print(f"- checkpoint：{summary.checkpoint_path}")
        print(f"- summary：{summary_path}")

        if summary.epochs:
            last_epoch = summary.epochs[-1]
            print("最后一轮：")
            print(f"- epoch：{last_epoch.epoch}")
            print(f"- train loss：{last_epoch.train_loss:.6f}")
            if last_epoch.val_loss is not None:
                print(f"- val loss：{last_epoch.val_loss:.6f}")
                print(f"- val accuracy：{last_epoch.accuracy:.6f}")
                print(f"- val macro_f1：{last_epoch.macro_f1:.6f}")
                print(f"- val cohen_kappa：{last_epoch.cohen_kappa:.6f}")
        return

    if args.command == "inspect-model":
        from .models import DeepFeatureNet, DeepSleepNet
        from .torch_dataset import (
            SleepEDFEpochDataset,
            SleepEDFSequenceDataset,
            create_dataloader,
            create_sequence_dataloader,
            group_records_by_subject,
            load_manifest,
        )

        import torch

        manifest_records = load_manifest(args.manifest)
        subject_map = group_records_by_subject(manifest_records)
        print(f"被试数量：{len(subject_map)}")
        print(f"被试列表：{list(subject_map)}")

        epoch_dataset = SleepEDFEpochDataset(args.manifest)
        epoch_dataloader = create_dataloader(epoch_dataset, batch_size=args.batch_size, shuffle=False)
        epoch_batch = next(iter(epoch_dataloader))

        feature_model = DeepFeatureNet(
            input_size=epoch_batch["signals"].shape[-1],
            n_classes=5,
        )
        feature_model.eval()
        with torch.no_grad():
            feature_logits = feature_model(epoch_batch["signals"])
            feature_repr = feature_model.extract_features(epoch_batch["signals"])

        print("DeepFeatureNet 检查：")
        print(f"- 输入 shape：{tuple(epoch_batch['signals'].shape)}")
        print(f"- 表征 shape：{tuple(feature_repr.shape)}")
        print(f"- logits shape：{tuple(feature_logits.shape)}")

        sequence_dataset = SleepEDFSequenceDataset(
            args.manifest,
            sequence_length=args.sequence_length,
        )
        sequence_dataloader = create_sequence_dataloader(
            sequence_dataset,
            batch_size=args.sequence_batch_size,
            shuffle=False,
        )
        sequence_batch = next(iter(sequence_dataloader))

        sequence_model = DeepSleepNet(
            input_size=sequence_batch["signals"].shape[-1],
            n_classes=5,
            seq_length=args.sequence_length,
            n_rnn_layers=2,
            return_last=False,
        )
        sequence_model.eval()
        with torch.no_grad():
            sequence_logits = sequence_model(sequence_batch["signals"])

        print("DeepSleepNet 检查：")
        print(f"- 序列输入 shape：{tuple(sequence_batch['signals'].shape)}")
        print(f"- 序列标签 shape：{tuple(sequence_batch['labels'].shape)}")
        print(f"- 序列 logits shape：{tuple(sequence_logits.shape)}")
        print(f"- 第一个序列被试：{sequence_batch['subject_ids'][0]}")
        print(f"- 第一个序列 epoch 范围：{sequence_batch['epoch_indices'][0][0]} -> {sequence_batch['epoch_indices'][0][-1]}")
        return

    parser.error("未知命令")


if __name__ == "__main__":
    main()
