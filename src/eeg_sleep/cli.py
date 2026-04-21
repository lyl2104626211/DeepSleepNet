from __future__ import annotations

import argparse
import sys

from .config import load_experiment_config
from .download_sleep_edf import DEFAULT_DOMESTIC_ZIP_URL, download_sleep_cassette
from .trainer import create_training_plan, evaluate_stage2, save_training_plan, train_stage1, train_stage2


def build_parser() -> argparse.ArgumentParser:
    """CLI 保持朴素：直接把每个命令需要的参数列出来。"""

    parser = argparse.ArgumentParser(description="EEG sleep staging entrypoint")
    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser("plan", help="generate a training plan")
    # --config: 实验配置文件路径，决定数据、模型和训练超参数。
    plan_parser.add_argument("--config", default="configs/base_experiment.yaml", help="experiment config path")

    preprocess_parser = subparsers.add_parser("preprocess", help="run Sleep-EDF preprocessing")
    # --input-dir: 原始 Sleep-EDF 数据目录，里面放 PSG/Hypnogram 文件。
    preprocess_parser.add_argument("--input-dir", default="data/raw/sleep_edf_subset", help="raw Sleep-EDF directory")
    # --output-dir: 预处理后的输出目录，里面会生成 epoch、manifest、summary。
    preprocess_parser.add_argument("--output-dir", default="data/processed/sleep_edf_subset", help="processed output directory")
    # --epoch-seconds: 每个 epoch 的长度，DeepSleepNet 默认按 30 秒切分。
    preprocess_parser.add_argument("--epoch-seconds", type=int, default=30, help="epoch length in seconds")
    # --channel: 显式指定要抽取的 EEG 通道，用于对齐论文里的 Fpz-Cz 或 Pz-Oz。
    preprocess_parser.add_argument("--channel", choices=["Fpz-Cz", "Pz-Oz"], default=None, help="optional EEG channel")
    # --trim-wake-minutes: 仅保留睡眠段前后多少分钟的清醒 W；论文 Sleep-EDF 常用 30。
    preprocess_parser.add_argument("--trim-wake-minutes", type=int, default=0, help="keep only edge wake minutes")

    download_parser = subparsers.add_parser("download-sleep-edf", help="download Sleep-EDF sleep-cassette")
    # --output-dir: 下载后的原始 EDF 保存目录。
    download_parser.add_argument("--output-dir", default="data/raw/sleep_edf_sleep_cassette", help="raw EDF output directory")
    download_parser.add_argument(
        "--base-url",
        default="https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/",
        help="directory listing URL used to build the file plan",
    )
    # --base-url: 用来抓取目录列表的网页地址，一般保持默认即可。
    download_parser.add_argument(
        "--download-base-url",
        default="https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/sleep-cassette/",
        help="foreign fallback base URL for direct EDF downloads",
    )
    # --download-base-url: 国外逐文件下载时真正使用的基础地址。
    download_parser.add_argument(
        "--domestic-mirror-url",
        default=DEFAULT_DOMESTIC_ZIP_URL,
        help="optional domestic mirror zip URL",
    )
    # --domestic-mirror-url: 国内镜像 zip 地址，能用时会优先走这个。
    # --record-prefix: 下载哪类记录，Sleep-EDF sleep-cassette 通常是 SC。
    download_parser.add_argument("--record-prefix", default="SC", help="record prefix, SC for sleep-cassette")
    # --max-records: 最多下载多少组记录，0 表示全部下载。
    download_parser.add_argument("--max-records", type=int, default=0, help="maximum number of record pairs to download")
    # --overwrite: 如果目标文件已存在，是否覆盖。
    download_parser.add_argument("--overwrite", action="store_true", help="overwrite existing files")
    # --dry-run: 只打印下载计划，不实际下载。
    download_parser.add_argument("--dry-run", action="store_true", help="print the download plan without downloading")

    inspect_parser = subparsers.add_parser("inspect-dataset", help="inspect the processed dataset")
    # --manifest: 预处理后生成的 manifest.json 路径。
    inspect_parser.add_argument("--manifest", default="data/processed/sleep_edf_subset/manifest.json", help="manifest path")
    # --batch-size: 检查 DataLoader 时使用的 batch 大小。
    inspect_parser.add_argument("--batch-size", type=int, default=4, help="epoch dataloader batch size")

    split_parser = subparsers.add_parser("split-subjects", help="split subjects into train/val/test")
    # --manifest: 样本索引文件路径，按它里面的被试信息做划分。
    split_parser.add_argument("--manifest", default="data/processed/sleep_edf_subset/manifest.json", help="manifest path")
    # --output: 被试划分结果保存路径。
    split_parser.add_argument("--output", default="data/processed/sleep_edf_subset/split.json", help="subject split output path")
    # --train-ratio: 训练集被试比例。
    split_parser.add_argument("--train-ratio", type=float, default=0.8, help="train subject ratio")
    # --val-ratio: 验证集被试比例。
    split_parser.add_argument("--val-ratio", type=float, default=0.1, help="validation subject ratio")
    # --test-ratio: 测试集被试比例。
    split_parser.add_argument("--test-ratio", type=float, default=0.1, help="test subject ratio")
    # --seed: 划分被试时使用的随机种子。
    split_parser.add_argument("--seed", type=int, default=42, help="random seed used for subject split")
    # --group-by: 按单晚记录切分，还是按同一受试者的多晚记录一起切分。
    split_parser.add_argument("--group-by", choices=["subject", "participant"], default="subject", help="group unit used for split")
    # --n-folds: 如果大于 1，则改为构建 k-fold 划分，当前 fold 做 test，其余做 train。
    split_parser.add_argument("--n-folds", type=int, default=0, help="optional k-fold split count")
    # --fold-index: 指定当前使用哪一个 fold 作为测试折，范围是 0 到 n-folds-1。
    split_parser.add_argument("--fold-index", type=int, default=0, help="k-fold test fold index")

    train_stage1_parser = subparsers.add_parser("train-stage1", help="train the stage-1 DeepFeatureNet")
    # --config: stage1 训练配置文件路径。
    train_stage1_parser.add_argument("--config", default="configs/base_experiment.yaml", help="experiment config path")
    # --manifest: 预处理后的样本索引文件。
    train_stage1_parser.add_argument("--manifest", default="data/processed/sleep_edf_subset/manifest.json", help="manifest path")
    # --split: 按被试划分好的 train/val/test 文件。
    train_stage1_parser.add_argument("--split", default="data/processed/sleep_edf_subset/split.json", help="subject split path")
    # --output-dir: stage1 训练输出目录，会保存 checkpoint 和 summary。
    train_stage1_parser.add_argument("--output-dir", default="results/deepsleepnet_baseline/stage1", help="stage-1 output directory")
    # --epochs: 可选覆盖配置文件中的训练轮数。
    train_stage1_parser.add_argument("--epochs", type=int, default=None, help="optional override for epochs")

    train_stage2_parser = subparsers.add_parser("train-stage2", help="fine-tune the stage-2 DeepSleepNet")
    # --config: stage2 训练配置文件路径。
    train_stage2_parser.add_argument("--config", default="configs/base_experiment.yaml", help="experiment config path")
    # --manifest: 预处理后的样本索引文件。
    train_stage2_parser.add_argument("--manifest", default="data/processed/sleep_edf_subset/manifest.json", help="manifest path")
    # --split: 按被试划分好的 train/val/test 文件。
    train_stage2_parser.add_argument("--split", default="data/processed/sleep_edf_subset/split.json", help="subject split path")
    train_stage2_parser.add_argument(
        "--stage1-checkpoint",
        default="results/deepsleepnet_baseline/stage1/best_model.pt",
        help="stage-1 checkpoint path",
    )
    # --stage1-checkpoint: stage1 训练得到的最优权重，stage2 会先加载它。
    # --output-dir: stage2 训练输出目录，会保存 checkpoint 和 summary。
    train_stage2_parser.add_argument("--output-dir", default="results/deepsleepnet_baseline/stage2", help="stage-2 output directory")
    # --epochs: 可选覆盖 stage2 的训练轮数。
    train_stage2_parser.add_argument("--epochs", type=int, default=None, help="optional override for stage-2 epochs")

    evaluate_parser = subparsers.add_parser("evaluate-stage2", help="evaluate a trained stage-2 model")
    # --config: 评估时使用的配置文件路径。
    evaluate_parser.add_argument("--config", default="configs/base_experiment.yaml", help="experiment config path")
    # --manifest: 预处理后的样本索引文件。
    evaluate_parser.add_argument("--manifest", default="data/processed/sleep_edf_subset/manifest.json", help="manifest path")
    # --split: 按被试划分好的 train/val/test 文件。
    evaluate_parser.add_argument("--split", default="data/processed/sleep_edf_subset/split.json", help="subject split path")
    # --checkpoint: 要评估的 stage2 模型权重路径。
    evaluate_parser.add_argument("--checkpoint", default="results/deepsleepnet_baseline/stage2/best_model.pt", help="stage-2 checkpoint path")
    # --subset: 指定评估 train/val/test 哪个子集。
    evaluate_parser.add_argument("--subset", choices=["train", "val", "test"], default="test", help="which subset to evaluate")
    # --output-dir: 评估结果保存目录，里面会写 summary 和混淆矩阵。
    evaluate_parser.add_argument("--output-dir", default=None, help="directory used to save evaluation artifacts")

    inspect_model_parser = subparsers.add_parser("inspect-model", help="inspect model forward passes")
    # --manifest: 预处理后的样本索引文件。
    inspect_model_parser.add_argument("--manifest", default="data/processed/sleep_edf_subset/manifest.json", help="manifest path")
    # --batch-size: 检查 DeepFeatureNet 时单 epoch batch 大小。
    inspect_model_parser.add_argument("--batch-size", type=int, default=4, help="epoch dataloader batch size")
    # --sequence-batch-size: 检查 DeepSleepNet 时序列 batch 大小。
    inspect_model_parser.add_argument("--sequence-batch-size", type=int, default=2, help="sequence dataloader batch size")
    # --sequence-length: 检查 DeepSleepNet 时使用的序列长度。
    inspect_model_parser.add_argument("--sequence-length", type=int, default=25, help="sequence length")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        None: _handle_plan,
        "plan": _handle_plan,
        "preprocess": _handle_preprocess,
        "download-sleep-edf": _handle_download,
        "inspect-dataset": _handle_inspect_dataset,
        "split-subjects": _handle_split_subjects,
        "train-stage1": _handle_train_stage1,
        "train-stage2": _handle_train_stage2,
        "evaluate-stage2": _handle_evaluate_stage2,
        "inspect-model": _handle_inspect_model,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.error("unknown command")
    handler(args)


def _handle_plan(args) -> None:
    config = load_experiment_config(args.config)
    plan = create_training_plan(config)
    plan_file = save_training_plan(plan, config.output.result_dir)

    print(f"experiment: {plan['experiment_name']}")
    print(f"plan file: {plan_file}")
    print("dataset checks:")
    if plan["dataset_checks"]:
        for issue in plan["dataset_checks"]:
            print(f"- {issue}")
    else:
        print("- directory checks passed")

    print("next actions:")
    for step in plan["next_actions"]:
        print(f"- {step}")


def _handle_preprocess(args) -> None:
    from .preprocess_sleep_edf import main as preprocess_main

    cli_args = [
        "preprocess",
        "--input-dir",
        args.input_dir,
        "--output-dir",
        args.output_dir,
        "--epoch-seconds",
        str(args.epoch_seconds),
    ]
    if args.channel is not None:
        cli_args.extend(["--channel", args.channel])
    if args.trim_wake_minutes > 0:
        cli_args.extend(["--trim-wake-minutes", str(args.trim_wake_minutes)])
    sys.argv = cli_args
    preprocess_main()


def _handle_download(args) -> None:
    plan = download_sleep_cassette(
        output_dir=args.output_dir,
        base_url=args.base_url,
        download_base_url=args.download_base_url,
        record_prefix=args.record_prefix,
        max_records=args.max_records,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        domestic_mirror_url=args.domestic_mirror_url,
    )

    print(f"planned records: {len({item.record_id for item in plan})}")
    print(f"planned files: {len(plan)}")
    if plan:
        print("first files:")
        for item in plan[: min(6, len(plan))]:
            print(f"- {item.file_name}")
    if args.dry_run:
        print("dry-run only, no files were downloaded")
    else:
        print(f"output dir: {args.output_dir}")


def _handle_inspect_dataset(args) -> None:
    from .torch_dataset import SleepEDFEpochDataset, create_dataloader

    dataset = SleepEDFEpochDataset(args.manifest)
    print(f"num samples: {len(dataset)}")

    sample = dataset[0]
    print("first sample:")
    print(f"- subject: {sample['subject_id']}")
    print(f"- epoch index: {sample['epoch_index']}")
    print(f"- label: {sample['label_name']} ({sample['label']})")
    print(f"- signal length: {len(sample['signal'])}")

    batch = next(iter(create_dataloader(dataset, batch_size=args.batch_size)))
    print("first batch:")
    print(f"- signals shape: {tuple(batch['signals'].shape)}")
    print(f"- labels shape: {tuple(batch['labels'].shape)}")


def _handle_split_subjects(args) -> None:
    from .torch_dataset import build_kfold_split, load_manifest, save_subject_split, split_subjects

    records = load_manifest(args.manifest)
    if args.n_folds > 1:
        split = build_kfold_split(
            records,
            n_folds=args.n_folds,
            fold_index=args.fold_index,
            seed=args.seed,
            group_by=args.group_by,
        )
    else:
        split = split_subjects(
            records,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            group_by=args.group_by,
        )
    split_path = save_subject_split(split, args.output)

    print(f"split file: {split_path}")
    if args.n_folds > 1:
        print(f"- k-fold: {args.n_folds}")
        print(f"- fold index: {args.fold_index}")
    print(f"- group by: {args.group_by}")
    print(f"- train subjects ({len(split.train_subjects)}): {split.train_subjects}")
    print(f"- val subjects ({len(split.val_subjects)}): {split.val_subjects}")
    print(f"- test subjects ({len(split.test_subjects)}): {split.test_subjects}")


def _handle_train_stage1(args) -> None:
    summary, summary_path = train_stage1(
        config=load_experiment_config(args.config),
        manifest_path=args.manifest,
        split_path=args.split,
        output_dir=args.output_dir,
        epochs_override=args.epochs,
    )
    _print_training_summary("stage-1 training finished", summary, summary_path)


def _handle_train_stage2(args) -> None:
    summary, summary_path = train_stage2(
        config=load_experiment_config(args.config),
        manifest_path=args.manifest,
        split_path=args.split,
        stage1_checkpoint_path=args.stage1_checkpoint,
        output_dir=args.output_dir,
        epochs_override=args.epochs,
    )
    _print_training_summary("stage-2 training finished", summary, summary_path)


def _handle_evaluate_stage2(args) -> None:
    summary, summary_path = evaluate_stage2(
        config=load_experiment_config(args.config),
        manifest_path=args.manifest,
        split_path=args.split,
        checkpoint_path=args.checkpoint,
        subset=args.subset,
        output_dir=args.output_dir,
    )

    print("stage-2 evaluation finished")
    print(f"- subset: {summary['subset']}")
    print(f"- device: {summary['device']}")
    print(f"- checkpoint: {summary['checkpoint_path']}")
    print(f"- subject ids: {summary['subject_ids']}")
    print(f"- sequence length: {summary['sequence_length']}")
    print(f"- eval stride: {summary['eval_sequence_stride']}")
    print(f"- sequences: {summary['num_sequences']}")
    print(f"- covered epochs: {summary['covered_epochs']}")
    print(f"- loss: {summary['loss']:.6f}")
    print(f"- accuracy: {summary['accuracy']:.6f}")
    print(f"- macro_f1: {summary['macro_f1']:.6f}")
    print(f"- cohen_kappa: {summary['cohen_kappa']:.6f}")
    print(f"- summary: {summary_path}")
    print(f"- confusion matrix: {summary['confusion_matrix_path']}")


def _handle_inspect_model(args) -> None:
    from .models import DeepFeatureNet, DeepSleepNet
    from .torch_dataset import SleepEDFEpochDataset, SleepEDFSequenceDataset, create_dataloader, create_sequence_dataloader

    import torch

    epoch_batch = next(iter(create_dataloader(SleepEDFEpochDataset(args.manifest), batch_size=args.batch_size)))
    feature_model = DeepFeatureNet(input_size=epoch_batch["signals"].shape[-1], n_classes=5)
    feature_model.eval()
    with torch.no_grad():
        feature_logits = feature_model(epoch_batch["signals"])
        feature_repr = feature_model.extract_features(epoch_batch["signals"])

    print("DeepFeatureNet:")
    print(f"- input shape: {tuple(epoch_batch['signals'].shape)}")
    print(f"- feature shape: {tuple(feature_repr.shape)}")
    print(f"- logits shape: {tuple(feature_logits.shape)}")

    sequence_dataset = SleepEDFSequenceDataset(args.manifest, sequence_length=args.sequence_length)
    sequence_batch = next(
        iter(create_sequence_dataloader(sequence_dataset, batch_size=args.sequence_batch_size))
    )
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

    print("DeepSleepNet:")
    print(f"- sequence input shape: {tuple(sequence_batch['signals'].shape)}")
    print(f"- sequence labels shape: {tuple(sequence_batch['labels'].shape)}")
    print(f"- sequence logits shape: {tuple(sequence_logits.shape)}")


def _print_training_summary(title: str, summary: dict, summary_path) -> None:
    print(title)
    print(f"- device: {summary['device']}")
    if "stage1_checkpoint_path" in summary:
        print(f"- stage1 checkpoint: {summary['stage1_checkpoint_path']}")
        print(f"- sequence length: {summary['sequence_length']}")
        print(f"- train stride: {summary['train_sequence_stride']}")
        print(f"- eval stride: {summary['eval_sequence_stride']}")
        print(f"- train sequences: {summary['train_size']}")
        print(f"- val sequences: {summary['val_size']}")
    else:
        print(f"- train size: {summary['train_size']}")
        print(f"- val size: {summary['val_size']}")
    print(f"- train subjects: {summary['train_subjects']}")
    print(f"- val subjects: {summary['val_subjects']}")
    print(f"- best epoch: {summary['best_epoch']}")
    print(f"- best val macro_f1: {summary['best_val_macro_f1']}")
    print(f"- checkpoint: {summary['checkpoint_path']}")
    print(f"- summary: {summary_path}")

    if summary["epochs"]:
        last_epoch = summary["epochs"][-1]
        print("last epoch:")
        print(f"- epoch: {last_epoch['epoch']}")
        print(f"- train loss: {last_epoch['train_loss']:.6f}")
        if last_epoch["val_loss"] is not None:
            print(f"- val loss: {last_epoch['val_loss']:.6f}")
            print(f"- val accuracy: {last_epoch['accuracy']:.6f}")
            print(f"- val macro_f1: {last_epoch['macro_f1']:.6f}")
            print(f"- val cohen_kappa: {last_epoch['cohen_kappa']:.6f}")
            if "covered_epochs" in last_epoch and last_epoch["covered_epochs"] is not None:
                print(f"- covered epochs: {last_epoch['covered_epochs']}")


if __name__ == "__main__":
    main()
