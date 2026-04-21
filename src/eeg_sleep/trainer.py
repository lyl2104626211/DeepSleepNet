from __future__ import annotations

import json
import random
from pathlib import Path

from .config import ExperimentConfig
from .datasets import SleepEDFDatasetBuilder
from .metrics import compute_classification_metrics, compute_confusion_matrix
from .models import build_model_summary
from .torch_dataset import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    SleepEDFEpochDataset,
    SleepEDFSequenceDataset,
    create_dataloader,
    create_sequence_dataloader,
    load_subject_split,
)


def create_training_plan(config: ExperimentConfig) -> dict:
    """根据实验配置生成一份最小训练计划说明。"""

    dataset_builder = SleepEDFDatasetBuilder(config.dataset)
    model_summary = build_model_summary(config.model)
    return {
        "experiment_name": config.experiment_name,
        "dataset_checks": dataset_builder.validate_layout(),
        "dataset_steps": dataset_builder.planned_steps(),
        "model_description": model_summary.description,
        "next_actions": [
            "先完成 stage1 训练",
            "再完成 stage2 序列微调",
            "最后在 test 集导出指标和混淆矩阵",
        ],
    }


def save_training_plan(plan: dict, output_dir: str | Path) -> Path:
    """把训练计划保存到输出目录。"""

    return _save_json(plan, Path(output_dir) / "training_plan.json")


def train_stage1(
    config: ExperimentConfig,
    manifest_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path | None = None,
    epochs_override: int | None = None,
):
    """训练 stage1 的单 epoch 分类器。"""

    torch, nn, Adam = _require_torch()
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

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=_build_weighted_sampler(train_dataset),
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    val_loader = (
        create_dataloader(
            val_dataset,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
        )
        if val_dataset is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFeatureNet(
        input_size=train_dataset[0]["signal"].shape[-1],
        n_classes=len(config.dataset.label_set),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.training.learning_rate)

    result_dir = _resolve_result_dir(config, output_dir, "stage1")
    checkpoint_path = result_dir / "best_model.pt"
    summary_path = result_dir / "training_summary.json"

    total_epochs = epochs_override or config.training.epochs
    history: list[dict] = []
    best_epoch = None
    best_macro_f1 = float("-inf")

    for epoch in range(1, total_epochs + 1):
        train_loss = _train_stage1_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": None,
            "accuracy": None,
            "macro_f1": None,
            "cohen_kappa": None,
        }

        if val_loader is not None:
            val_result = _evaluate_epoch_model(model, val_loader, criterion, device, f"Epoch {epoch} [val]")
            epoch_log.update(val_result)
            best_epoch, best_macro_f1 = _update_best_checkpoint(
                torch=torch,
                model=model,
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                candidate_score=val_result["macro_f1"],
                best_epoch=best_epoch,
                best_score=best_macro_f1,
            )
        else:
            best_epoch, best_macro_f1 = _save_checkpoint_without_validation(
                torch=torch,
                model=model,
                checkpoint_path=checkpoint_path,
                epoch=epoch,
            )

        history.append(epoch_log)
        _print_stage1_epoch(epoch_log, epoch, total_epochs)

    summary = {
        "experiment_name": config.experiment_name,
        "stage": "stage1_deepfeaturenet",
        "device": str(device),
        "train_subjects": split.train_subjects,
        "val_subjects": split.val_subjects,
        "train_size": len(train_dataset),
        "val_size": 0 if val_dataset is None else len(val_dataset),
        "best_epoch": best_epoch,
        "best_val_macro_f1": _finalize_best_score(best_macro_f1),
        "checkpoint_path": str(checkpoint_path),
        "epochs": history,
    }
    _save_json(summary, summary_path)
    return summary, summary_path


def train_stage2(
    config: ExperimentConfig,
    manifest_path: str | Path,
    split_path: str | Path,
    stage1_checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    epochs_override: int | None = None,
):
    """训练 stage2 的序列模型，并加载 stage1 的特征提取器权重。"""

    torch, nn, _ = _require_torch()
    from .models import DeepSleepNet

    _set_random_seed(config.training.seed)

    split = load_subject_split(split_path)
    if not split.train_subjects:
        raise ValueError("train_subjects 为空，无法开始第二阶段训练")

    sequence_length = config.training.stage2_sequence_length
    train_stride = config.training.stage2_sequence_stride or sequence_length
    eval_stride = config.training.stage2_eval_stride
    batch_size = _resolve_stage2_batch_size(config)
    total_epochs = epochs_override or config.training.stage2_epochs or config.training.epochs

    train_dataset = SleepEDFSequenceDataset(
        manifest_path,
        sequence_length=sequence_length,
        stride=train_stride,
        subject_ids=split.train_subjects,
    )
    val_dataset = (
        SleepEDFSequenceDataset(
            manifest_path,
            sequence_length=sequence_length,
            stride=eval_stride,
            subject_ids=split.val_subjects,
        )
        if split.val_subjects
        else None
    )

    train_loader = create_sequence_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    val_loader = (
        create_sequence_dataloader(
            val_dataset,
            batch_size=batch_size,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
        )
        if val_dataset is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSleepNet(
        input_size=train_dataset[0]["signals"].shape[-1],
        n_classes=len(config.dataset.label_set),
        seq_length=sequence_length,
        n_rnn_layers=2,
        return_last=False,
    ).to(device)
    _load_stage1_weights(model, stage1_checkpoint_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = _build_stage2_optimizer(model, config)
    clip_norm = config.training.stage2_gradient_clip_norm

    result_dir = _resolve_result_dir(config, output_dir, "stage2")
    checkpoint_path = result_dir / "best_model.pt"
    summary_path = result_dir / "training_summary.json"

    history: list[dict] = []
    best_epoch = None
    best_macro_f1 = float("-inf")

    for epoch in range(1, total_epochs + 1):
        train_loss = _train_stage2_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            clip_norm=clip_norm,
        )
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": None,
            "accuracy": None,
            "macro_f1": None,
            "cohen_kappa": None,
            "covered_epochs": None,
        }

        if val_loader is not None:
            val_result = _evaluate_sequence_model(
                model,
                val_loader,
                criterion,
                device,
                f"Epoch {epoch} [stage2-val]",
            )
            epoch_log.update(val_result)
            best_epoch, best_macro_f1 = _update_best_checkpoint(
                torch=torch,
                model=model,
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                candidate_score=val_result["macro_f1"],
                best_epoch=best_epoch,
                best_score=best_macro_f1,
            )
        else:
            best_epoch, best_macro_f1 = _save_checkpoint_without_validation(
                torch=torch,
                model=model,
                checkpoint_path=checkpoint_path,
                epoch=epoch,
            )

        history.append(epoch_log)
        _print_stage2_epoch(epoch_log, epoch, total_epochs)

    summary = {
        "experiment_name": config.experiment_name,
        "stage": "stage2_deepsleepnet",
        "device": str(device),
        "stage1_checkpoint_path": str(stage1_checkpoint_path),
        "sequence_length": sequence_length,
        "train_sequence_stride": train_stride,
        "eval_sequence_stride": eval_stride,
        "train_subjects": split.train_subjects,
        "val_subjects": split.val_subjects,
        "train_size": len(train_dataset),
        "val_size": 0 if val_dataset is None else len(val_dataset),
        "best_epoch": best_epoch,
        "best_val_macro_f1": _finalize_best_score(best_macro_f1),
        "checkpoint_path": str(checkpoint_path),
        "epochs": history,
    }
    _save_json(summary, summary_path)
    return summary, summary_path


def evaluate_stage2(
    config: ExperimentConfig,
    manifest_path: str | Path,
    split_path: str | Path,
    checkpoint_path: str | Path,
    subset: str = "test",
    output_dir: str | Path | None = None,
):
    """评估 stage2 模型，并按需导出混淆矩阵。"""

    torch, nn, _ = _require_torch()
    from .models import DeepSleepNet

    split = load_subject_split(split_path)
    subject_ids = {
        "train": split.train_subjects,
        "val": split.val_subjects,
        "test": split.test_subjects,
    }.get(subset)
    if subject_ids is None:
        raise ValueError(f"不支持的 subset：{subset}")
    if not subject_ids:
        raise ValueError(f"{subset}_subjects 为空，无法评估")

    sequence_length = config.training.stage2_sequence_length
    eval_stride = config.training.stage2_eval_stride
    batch_size = _resolve_stage2_batch_size(config)

    dataset = SleepEDFSequenceDataset(
        manifest_path,
        sequence_length=sequence_length,
        stride=eval_stride,
        subject_ids=subject_ids,
    )
    dataloader = create_sequence_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSleepNet(
        input_size=dataset[0]["signals"].shape[-1],
        n_classes=len(config.dataset.label_set),
        seq_length=sequence_length,
        n_rnn_layers=2,
        return_last=False,
    ).to(device)
    _load_checkpoint(model, checkpoint_path)

    result = _evaluate_sequence_model(
        model,
        dataloader,
        nn.CrossEntropyLoss(),
        device,
        f"Evaluate [{subset}]",
    )

    result_dir = _resolve_result_dir(config, output_dir, "stage2")
    summary_path = result_dir / f"evaluation_{subset}.json"
    confusion_matrix_path = _save_confusion_matrix_if_needed(config, result, result_dir, subset)

    summary = {
        "experiment_name": config.experiment_name,
        "stage": "stage2_deepsleepnet",
        "subset": subset,
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "subject_ids": subject_ids,
        "sequence_length": sequence_length,
        "eval_sequence_stride": eval_stride,
        "num_sequences": result["num_sequences"],
        "covered_epochs": result["covered_epochs"],
        "loss": result["val_loss"],
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "cohen_kappa": result["cohen_kappa"],
        "summary_path": str(summary_path),
        "confusion_matrix_path": None if confusion_matrix_path is None else str(confusion_matrix_path),
    }
    _save_json(summary, summary_path)
    return summary, summary_path


def _require_torch():
    """按需导入 torch，并给出一致的缺失依赖提示。"""

    try:
        import torch
        from torch import nn
        from torch.optim import Adam
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少 torch，请先安装依赖：uv sync") from exc
    return torch, nn, Adam


def _save_json(payload: dict, path: str | Path) -> Path:
    """把字典按 UTF-8 JSON 形式保存到指定路径。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return path


def _set_random_seed(seed: int) -> None:
    """固定随机种子，避免同一配置下结果波动过大。"""

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
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_progress(iterable, total: int, description: str):
    """有 tqdm 时显示进度条，没有时就直接遍历。"""

    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return iterable
    return tqdm(iterable, total=total, desc=description, leave=False, dynamic_ncols=True)


def _resolve_result_dir(
    config: ExperimentConfig,
    output_dir: str | Path | None,
    stage_name: str,
) -> Path:
    """统一处理输出目录，兼容显式覆盖和默认目录。"""

    result_dir = Path(output_dir) if output_dir is not None else Path(config.output.result_dir) / stage_name
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def _resolve_stage2_batch_size(config: ExperimentConfig) -> int:
    """stage2 默认 batch 更小，避免直接复用 stage1 的较大 batch。"""

    return config.training.stage2_batch_size or max(1, min(config.training.batch_size, 8))


def _update_best_checkpoint(
    torch,
    model,
    checkpoint_path: str | Path,
    epoch: int,
    candidate_score: float,
    best_epoch,
    best_score: float,
) -> tuple[int | None, float]:
    """按 macro_f1 维护最佳 checkpoint。"""

    if candidate_score > best_score:
        torch.save(model.state_dict(), checkpoint_path)
        return epoch, candidate_score
    return best_epoch, best_score


def _save_checkpoint_without_validation(
    torch,
    model,
    checkpoint_path: str | Path,
    epoch: int,
) -> tuple[int, float]:
    """无验证集时始终保存当前权重，最后保留的就是最新一轮。"""

    torch.save(model.state_dict(), checkpoint_path)
    return epoch, float("nan")


def _finalize_best_score(best_score: float) -> float | None:
    """把内部的哨兵值转换成对外 summary 中的最终值。"""

    return None if best_score == float("-inf") else best_score


def _save_confusion_matrix_if_needed(
    config: ExperimentConfig,
    result: dict,
    result_dir: Path,
    subset: str,
) -> Path | None:
    """按配置决定是否导出混淆矩阵文件。"""

    if not config.evaluation.save_confusion_matrix:
        return None

    confusion_matrix_path = result_dir / f"confusion_matrix_{subset}.json"
    _save_json(
        {
            "labels": [ID_TO_LABEL[label_id] for label_id in sorted(ID_TO_LABEL)],
            "matrix": compute_confusion_matrix(
                result["y_true"],
                result["y_pred"],
                labels=sorted(ID_TO_LABEL),
            ),
        },
        confusion_matrix_path,
    )
    return confusion_matrix_path


def _build_weighted_sampler(dataset: SleepEDFEpochDataset):
    """stage1 保留类别均衡采样，缓解标签分布不均。"""

    torch, _, _ = _require_torch()
    from torch.utils.data import WeightedRandomSampler

    label_ids = [LABEL_TO_ID[record.label] for record in dataset.records]
    class_counts: dict[int, int] = {}
    for label_id in label_ids:
        class_counts[label_id] = class_counts.get(label_id, 0) + 1

    weights = [1.0 / class_counts[label_id] for label_id in label_ids]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def _load_checkpoint(model, checkpoint_path: str | Path) -> None:
    """加载完整模型权重，兼容常见 checkpoint 封装格式。"""

    state_dict = _load_state_dict_from_checkpoint(checkpoint_path)
    model.load_state_dict(state_dict, strict=True)


def _load_stage1_weights(model, checkpoint_path: str | Path) -> None:
    """只把 stage1 的特征提取器权重加载到 stage2 中。"""

    state_dict = _load_state_dict_from_checkpoint(checkpoint_path)

    # 如果 checkpoint 来自完整 DeepSleepNet，需要先取出 feature_extractor 前缀。
    if any(str(key).startswith("feature_extractor.") for key in state_dict):
        state_dict = {
            str(key).removeprefix("feature_extractor."): value
            for key, value in state_dict.items()
            if str(key).startswith("feature_extractor.")
        }

    model.feature_extractor.load_state_dict(state_dict, strict=True)


def _load_state_dict_from_checkpoint(checkpoint_path: str | Path) -> dict:
    """统一解析 checkpoint，兼容裸 state_dict 与字典封装格式。"""

    torch, _, _ = _require_torch()
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError("checkpoint 格式不正确")

    if any(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key).removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def _build_stage2_optimizer(model, config: ExperimentConfig):
    """stage2 采用分组学习率：CNN 小一些，时序部分大一些。"""

    _, _, Adam = _require_torch()
    cnn_lr = config.training.stage2_cnn_learning_rate or config.training.learning_rate * 0.1
    seq_lr = config.training.stage2_sequence_learning_rate or config.training.learning_rate

    return Adam(
        [
            {"params": model.feature_extractor.parameters(), "lr": cnn_lr},
            {
                "params": list(model.shortcut_projection.parameters())
                + list(model.sequence_model.parameters())
                + list(model.classifier.parameters()),
                "lr": seq_lr,
            },
        ]
    )


def _train_stage1_one_epoch(model, dataloader, criterion, optimizer, device, epoch: int) -> float:
    """执行一轮 stage1 训练，并返回样本级平均 loss。"""

    model.train()
    total_loss = 0.0
    total_samples = 0

    progress = _make_progress(dataloader, len(dataloader), f"Epoch {epoch} [train]")
    for batch in progress:
        signals = batch["signals"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        loss = criterion(model(signals), labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        if hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{total_loss / max(total_samples, 1):.4f}")

    return total_loss / max(total_samples, 1)


def _train_stage2_one_epoch(model, dataloader, criterion, optimizer, device, epoch: int, clip_norm: float | None) -> float:
    """执行一轮 stage2 训练，并返回 token 级平均 loss。"""

    torch, _, _ = _require_torch()
    model.train()
    total_loss = 0.0
    total_tokens = 0

    progress = _make_progress(dataloader, len(dataloader), f"Epoch {epoch} [stage2-train]")
    for batch in progress:
        signals = batch["signals"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss.backward()

        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        optimizer.step()

        token_count = int(labels.numel())
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count

        if hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{total_loss / max(total_tokens, 1):.4f}")

    return total_loss / max(total_tokens, 1)


def _evaluate_epoch_model(model, dataloader, criterion, device, description: str) -> dict:
    """评估 stage1 模型，直接在 batch 级别累计指标。"""

    torch, _, _ = _require_torch()
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        progress = _make_progress(dataloader, len(dataloader), description)
        for batch in progress:
            signals = batch["signals"].to(device)
            labels = batch["labels"].to(device)
            logits = model(signals)
            loss = criterion(logits, labels)

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

            if hasattr(progress, "set_postfix"):
                progress.set_postfix(loss=f"{total_loss / max(total_samples, 1):.4f}")

    metrics = compute_classification_metrics(y_true, y_pred)
    return {
        "val_loss": total_loss / max(total_samples, 1),
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "cohen_kappa": metrics.cohen_kappa,
    }


def _evaluate_sequence_model(model, dataloader, criterion, device, description: str) -> dict:
    """评估 stage2 模型，并把重叠窗口的预测重新聚合到 epoch 级。"""

    torch, _, _ = _require_torch()
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    epoch_logits: dict[tuple[str, int], torch.Tensor] = {}
    epoch_labels: dict[tuple[str, int], int] = {}

    with torch.no_grad():
        progress = _make_progress(dataloader, len(dataloader), description)
        for batch in progress:
            signals = batch["signals"].to(device)
            labels = batch["labels"].to(device)
            logits = model(signals)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

            token_count = int(labels.numel())
            total_loss += float(loss.item()) * token_count
            total_tokens += token_count

            logits_cpu = logits.cpu()
            labels_cpu = labels.cpu()
            for batch_index, subject_id in enumerate(batch["subject_ids"]):
                for step_index, epoch_index in enumerate(batch["epoch_indices"][batch_index]):
                    key = (subject_id, int(epoch_index))
                    current_logits = logits_cpu[batch_index, step_index]
                    current_label = int(labels_cpu[batch_index, step_index].item())

                    # 同一个 epoch 可能被多个滑窗覆盖，这里把各窗口 logits 累加后再做 argmax。
                    epoch_logits[key] = epoch_logits.get(key, torch.zeros_like(current_logits)) + current_logits
                    epoch_labels[key] = current_label

            if hasattr(progress, "set_postfix"):
                progress.set_postfix(loss=f"{total_loss / max(total_tokens, 1):.4f}")

    ordered_keys = sorted(epoch_labels, key=lambda item: (item[0], item[1]))
    y_true = [epoch_labels[key] for key in ordered_keys]
    y_pred = [int(epoch_logits[key].argmax().item()) for key in ordered_keys]
    metrics = compute_classification_metrics(y_true, y_pred)
    return {
        "val_loss": total_loss / max(total_tokens, 1),
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "cohen_kappa": metrics.cohen_kappa,
        "covered_epochs": len(ordered_keys),
        "num_sequences": len(dataloader.dataset),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _print_stage1_epoch(epoch_log: dict, epoch: int, total_epochs: int) -> None:
    """打印 stage1 每轮训练摘要。"""

    if epoch_log["val_loss"] is None:
        print(f"第 {epoch}/{total_epochs} 轮完成：train_loss={epoch_log['train_loss']:.6f}")
        return
    print(
        f"第 {epoch}/{total_epochs} 轮完成："
        f" train_loss={epoch_log['train_loss']:.6f},"
        f" val_loss={epoch_log['val_loss']:.6f},"
        f" val_acc={epoch_log['accuracy']:.6f},"
        f" val_macro_f1={epoch_log['macro_f1']:.6f},"
        f" val_kappa={epoch_log['cohen_kappa']:.6f}"
    )


def _print_stage2_epoch(epoch_log: dict, epoch: int, total_epochs: int) -> None:
    """打印 stage2 每轮训练摘要。"""

    if epoch_log["val_loss"] is None:
        print(f"第 {epoch}/{total_epochs} 轮第二阶段完成：train_loss={epoch_log['train_loss']:.6f}")
        return
    print(
        f"第 {epoch}/{total_epochs} 轮第二阶段完成："
        f" train_loss={epoch_log['train_loss']:.6f},"
        f" val_loss={epoch_log['val_loss']:.6f},"
        f" val_acc={epoch_log['accuracy']:.6f},"
        f" val_macro_f1={epoch_log['macro_f1']:.6f},"
        f" val_kappa={epoch_log['cohen_kappa']:.6f},"
        f" covered_epochs={epoch_log['covered_epochs']}"
    )
