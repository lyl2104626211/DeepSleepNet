from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random

from .config import ExperimentConfig
from .datasets import SleepEDFDatasetBuilder
from .metrics import compute_classification_metrics
from .models import build_model_summary
from .torch_dataset import (
    LABEL_TO_ID,
    SleepEDFEpochDataset,
    create_dataloader,
    load_subject_split,
)


@dataclass
class TrainingPlan:
    experiment_name: str
    dataset_checks: list[str]
    dataset_steps: list[str]
    model_description: str
    next_actions: list[str]


@dataclass
class Stage1EpochLog:
    epoch: int
    train_loss: float
    val_loss: float | None
    accuracy: float | None
    macro_f1: float | None
    cohen_kappa: float | None


@dataclass
class Stage1TrainingSummary:
    experiment_name: str
    stage: str
    device: str
    train_subjects: list[str]
    val_subjects: list[str]
    train_size: int
    val_size: int
    best_epoch: int | None
    best_val_macro_f1: float | None
    checkpoint_path: str | None
    epochs: list[Stage1EpochLog]


def create_training_plan(config: ExperimentConfig) -> TrainingPlan:
    dataset_builder = SleepEDFDatasetBuilder(config.dataset)
    model_summary = build_model_summary(config.model)

    checks = dataset_builder.validate_layout()
    next_actions = [
        "按被试划分 train / val / test，并固定随机种子",
        "实现 DeepFeatureNet 预训练循环（类别均衡 oversampling）",
        "实现 DeepSleepNet 序列微调循环（双学习率 Adam）",
        "接入 Accuracy、Macro-F1、Cohen's Kappa 和混淆矩阵",
        "保存权重、日志和配置，形成可复现实验记录",
    ]

    return TrainingPlan(
        experiment_name=config.experiment_name,
        dataset_checks=checks,
        dataset_steps=dataset_builder.planned_steps(),
        model_description=model_summary.description,
        next_actions=next_actions,
    )


def save_training_plan(plan: TrainingPlan, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target_file = output_path / "training_plan.json"
    with target_file.open("w", encoding="utf-8") as file:
        json.dump(asdict(plan), file, ensure_ascii=False, indent=2)
    return target_file


def _set_random_seed(seed: int) -> None:
    """固定随机种子，尽量保证同一配置下结果可复现。"""

    random.seed(seed)

    try:
        import numpy as np
    except ModuleNotFoundError:
        np = None
    if np is not None:
        np.random.seed(seed)

    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _build_weighted_sampler(dataset: SleepEDFEpochDataset):
    """按类别频次构造采样器，减轻类别不平衡。"""

    try:
        import torch
        from torch.utils.data import WeightedRandomSampler
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少 torch，请先安装依赖后再训练：uv sync"
        ) from exc

    label_ids = [LABEL_TO_ID[record.label] for record in dataset.records]
    class_counts: dict[int, int] = {}
    for label_id in label_ids:
        class_counts[label_id] = class_counts.get(label_id, 0) + 1

    sample_weights = [1.0 / class_counts[label_id] for label_id in label_ids]
    weight_tensor = torch.tensor(sample_weights, dtype=torch.double)

    return WeightedRandomSampler(
        weights=weight_tensor,
        num_samples=len(sample_weights),
        replacement=True,
    )


def _create_progress(iterable, total: int | None, description: str):
    """优先使用 tqdm 显示进度条；若环境中没有 tqdm，则退回普通迭代。"""

    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return iterable

    return tqdm(
        iterable,
        total=total,
        desc=description,
        leave=False,
        dynamic_ncols=True,
    )


def _run_stage1_eval(model, dataloader, criterion, device, epoch_idx: int):
    """验证阶段：不更新参数，只计算 loss 和分类指标。"""

    import torch

    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        progress = _create_progress(
            dataloader,
            total=len(dataloader),
            description=f"Epoch {epoch_idx} [val]",
        )
        for batch in progress:
            signals = batch["signals"].to(device)
            labels = batch["labels"].to(device)

            logits = model(signals)
            loss = criterion(logits, labels)

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            predictions = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

            if hasattr(progress, "set_postfix"):
                average_loss = total_loss / max(total_samples, 1)
                progress.set_postfix(loss=f"{average_loss:.4f}")

    average_loss = total_loss / max(total_samples, 1)
    metrics = compute_classification_metrics(y_true, y_pred)
    return average_loss, metrics


def train_stage1(
    config: ExperimentConfig,
    manifest_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path | None = None,
    epochs_override: int | None = None,
):
    """训练第一阶段 DeepFeatureNet 单 epoch 分类器。"""

    try:
        import torch
        from torch import nn
        from torch.optim import Adam
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少 torch，请先安装依赖后再训练：uv sync"
        ) from exc

    from .models import DeepFeatureNet

    _set_random_seed(config.training.seed)

    split = load_subject_split(split_path)
    if not split.train_subjects:
        raise ValueError("train_subjects 为空，无法开始第一阶段训练")

    train_dataset = SleepEDFEpochDataset(manifest_path, subject_ids=split.train_subjects)
    val_dataset = (
        SleepEDFEpochDataset(manifest_path, subject_ids=split.val_subjects)
        if split.val_subjects
        else None
    )

    train_sampler = _build_weighted_sampler(train_dataset)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    val_loader = (
        create_dataloader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
        )
        if val_dataset is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signal_length = train_dataset[0]["signal"].shape[-1]

    model = DeepFeatureNet(
        input_size=signal_length,
        n_classes=len(config.dataset.label_set),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.training.learning_rate)

    result_root = Path(output_dir) if output_dir is not None else Path(config.output.result_dir) / "stage1"
    result_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_root / "best_model.pt"
    summary_path = result_root / "training_summary.json"

    n_epochs = epochs_override if epochs_override is not None else config.training.epochs
    history: list[Stage1EpochLog] = []
    best_epoch: int | None = None
    best_macro_f1 = float("-inf")
    best_checkpoint_saved = False

    for epoch_idx in range(1, n_epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        print(f"开始第 {epoch_idx}/{n_epochs} 轮训练")
        progress = _create_progress(
            train_loader,
            total=len(train_loader),
            description=f"Epoch {epoch_idx} [train]",
        )
        for batch in progress:
            signals = batch["signals"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.shape[0]
            total_train_loss += float(loss.item()) * batch_size
            total_train_samples += batch_size

            if hasattr(progress, "set_postfix"):
                average_loss = total_train_loss / max(total_train_samples, 1)
                progress.set_postfix(loss=f"{average_loss:.4f}")

        train_loss = total_train_loss / max(total_train_samples, 1)

        if val_loader is not None:
            val_loss, val_metrics = _run_stage1_eval(
                model,
                val_loader,
                criterion,
                device,
                epoch_idx=epoch_idx,
            )
            current_macro_f1 = val_metrics.macro_f1
            if current_macro_f1 > best_macro_f1:
                best_macro_f1 = current_macro_f1
                best_epoch = epoch_idx
                torch.save(model.state_dict(), checkpoint_path)
                best_checkpoint_saved = True
        else:
            val_loss = None
            val_metrics = None
            if not best_checkpoint_saved:
                best_epoch = epoch_idx
                best_macro_f1 = float("nan")
                torch.save(model.state_dict(), checkpoint_path)
                best_checkpoint_saved = True

        if val_metrics is not None:
            print(
                f"第 {epoch_idx} 轮完成："
                f" train_loss={train_loss:.6f},"
                f" val_loss={val_loss:.6f},"
                f" val_acc={val_metrics.accuracy:.6f},"
                f" val_macro_f1={val_metrics.macro_f1:.6f},"
                f" val_kappa={val_metrics.cohen_kappa:.6f}"
            )
        else:
            print(f"第 {epoch_idx} 轮完成：train_loss={train_loss:.6f}")

        history.append(
            Stage1EpochLog(
                epoch=epoch_idx,
                train_loss=train_loss,
                val_loss=val_loss,
                accuracy=None if val_metrics is None else val_metrics.accuracy,
                macro_f1=None if val_metrics is None else val_metrics.macro_f1,
                cohen_kappa=None if val_metrics is None else val_metrics.cohen_kappa,
            )
        )

    summary = Stage1TrainingSummary(
        experiment_name=config.experiment_name,
        stage="stage1_deepfeaturenet",
        device=str(device),
        train_subjects=split.train_subjects,
        val_subjects=split.val_subjects,
        train_size=len(train_dataset),
        val_size=0 if val_dataset is None else len(val_dataset),
        best_epoch=best_epoch,
        best_val_macro_f1=None if best_macro_f1 == float("-inf") else best_macro_f1,
        checkpoint_path=str(checkpoint_path) if best_checkpoint_saved else None,
        epochs=history,
    )

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, ensure_ascii=False, indent=2)

    return summary, summary_path
