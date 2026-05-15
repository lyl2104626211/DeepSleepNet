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
    eog_dropout_prob: float = 0.0,
    eog_channel_index: int = 1,
):
    """训练 stage1 的单 epoch 分类器。"""

    torch, nn, Adam = _require_torch()

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
    input_channels = _infer_input_channels(train_dataset[0]["signal"])
    model = _build_stage1_model(
        config=config,
        input_size=train_dataset[0]["signal"].shape[-1],
        input_channels=input_channels,
        n_classes=len(config.dataset.label_set),
    ).to(device)
    _configure_internal_eog_dropout(model, eog_dropout_prob, eog_channel_index)
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
        train_loss = _train_stage1_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            eog_dropout_prob=eog_dropout_prob,
            eog_channel_index=eog_channel_index,
        )
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
        "eog_dropout_prob": eog_dropout_prob,
        "eog_channel_index": eog_channel_index if eog_dropout_prob > 0 else None,
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
    eog_dropout_prob: float = 0.0,
    eog_channel_index: int = 1,
):
    """训练 stage2 的序列模型，并加载 stage1 的特征提取器权重。"""

    torch, nn, _ = _require_torch()

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
    input_channels = _infer_input_channels(train_dataset[0]["signals"][0])
    model = _build_stage2_model(
        config=config,
        input_size=train_dataset[0]["signals"].shape[-1],
        input_channels=input_channels,
        n_classes=len(config.dataset.label_set),
        sequence_length=sequence_length,
    ).to(device)
    _configure_internal_eog_dropout(model, eog_dropout_prob, eog_channel_index)
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
            eog_dropout_prob=eog_dropout_prob,
            eog_channel_index=eog_channel_index,
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
        "eog_dropout_prob": eog_dropout_prob,
        "eog_channel_index": eog_channel_index if eog_dropout_prob > 0 else None,
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
    mask_eog: bool = False,
    eog_channel_index: int = 1,
):
    """评估 stage2 模型，并按需导出混淆矩阵。"""

    torch, nn, _ = _require_torch()

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
    input_channels = _infer_input_channels(dataset[0]["signals"][0])
    model = _build_stage2_model(
        config=config,
        input_size=dataset[0]["signals"].shape[-1],
        input_channels=input_channels,
        n_classes=len(config.dataset.label_set),
        sequence_length=sequence_length,
    ).to(device)
    _configure_internal_eog_dropout(model, 0.0, eog_channel_index)
    _load_checkpoint(model, checkpoint_path)

    result = _evaluate_sequence_model(
        model,
        dataloader,
        nn.CrossEntropyLoss(),
        device,
        f"Evaluate [{subset}]",
        zero_channel_indices=[eog_channel_index] if mask_eog else None,
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
        "input_perturbation": {
            "mask_eog": mask_eog,
            "eog_channel_index": eog_channel_index if mask_eog else None,
        },
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


def _infer_input_channels(signal) -> int:
    """兼容旧的单通道 [L] 和新的多通道 [C, L] 样本格式。"""

    return int(signal.shape[0]) if getattr(signal, "ndim", 0) == 2 else 1


def _build_stage1_model(
    config: ExperimentConfig,
    input_size: int,
    input_channels: int,
    n_classes: int,
):
    """按配置构建 stage1 模型。"""

    if config.model.name == "deepsleepnet_baseline":
        from .models import DeepFeatureNet

        return DeepFeatureNet(
            input_size=input_size,
            input_channels=input_channels,
            n_classes=n_classes,
        )

    if config.model.name == "deepsleepnet_gated_fusion":
        if input_channels < 2:
            raise ValueError("deepsleepnet_gated_fusion 需要 EEG+EOG 双通道输入")
        from .robust_schemes.scheme_b_gated_fusion import GatedFusionFeatureNet

        return GatedFusionFeatureNet(
            input_size=input_size,
            n_classes=n_classes,
        )

    if config.model.name == "deepsleepnet_mixture_fusion":
        if input_channels < 2:
            raise ValueError("deepsleepnet_mixture_fusion 需要 EEG+EOG 双通道输入")
        from .robust_schemes.scheme_c_mixture_fusion import MixtureFusionFeatureNet

        return MixtureFusionFeatureNet(
            input_size=input_size,
            n_classes=n_classes,
        )

    if config.model.name == "deepsleepnet_quality_guided_generator":
        if input_channels < 2:
            raise ValueError("deepsleepnet_quality_guided_generator 需要 EEG+EOG 双通道输入")
        from .robust_schemes.scheme_d_quality_guided_generator import QualityGuidedGeneratorFeatureNet

        return QualityGuidedGeneratorFeatureNet(
            input_size=input_size,
            n_classes=n_classes,
        )

    if config.model.name == "deepsleepnet_quality_guided_generator_v2":
        if input_channels < 2:
            raise ValueError("deepsleepnet_quality_guided_generator_v2 需要 EEG+EOG 双通道输入")
        from .robust_schemes.scheme_d_v2_quality_guided_generator import QualityGuidedGeneratorV2FeatureNet

        return QualityGuidedGeneratorV2FeatureNet(
            input_size=input_size,
            n_classes=n_classes,
        )

    if config.model.name == "deepsleepnet_quality_guided_generator_v3":
        if input_channels < 2:
            raise ValueError("deepsleepnet_quality_guided_generator_v3 需要 EEG+EOG 双通道输入")
        from .robust_schemes.scheme_d_v3_quality_guided_generator import QualityGuidedGeneratorV3FeatureNet

        return QualityGuidedGeneratorV3FeatureNet(
            input_size=input_size,
            n_classes=n_classes,
        )

    raise ValueError(f"暂不支持的模型名称：{config.model.name}")


def _build_stage2_model(
    config: ExperimentConfig,
    input_size: int,
    input_channels: int,
    n_classes: int,
    sequence_length: int,
):
    """按配置构建 stage2 模型。"""

    if config.model.name == "deepsleepnet_baseline":
        from .models import DeepSleepNet

        return DeepSleepNet(
            input_size=input_size,
            input_channels=input_channels,
            n_classes=n_classes,
            seq_length=sequence_length,
            n_rnn_layers=2,
            return_last=False,
        )

    if config.model.name == "deepsleepnet_gated_fusion":
        if input_channels < 2:
            raise ValueError("deepsleepnet_gated_fusion 需要 EEG+EOG 双通道输入")
        from .robust_schemes.scheme_b_gated_fusion import GatedFusionDeepSleepNet

        return GatedFusionDeepSleepNet(
            input_size=input_size,
            n_classes=n_classes,
            seq_length=sequence_length,
            n_rnn_layers=2,
            return_last=False,
        )

    if config.model.name == "deepsleepnet_mixture_fusion":
        if input_channels < 2:
            raise ValueError("deepsleepnet_mixture_fusion 需要 EEG+EOG 双通道输入")
        from .robust_schemes.scheme_c_mixture_fusion import MixtureFusionDeepSleepNet

        return MixtureFusionDeepSleepNet(
            input_size=input_size,
            n_classes=n_classes,
            seq_length=sequence_length,
            n_rnn_layers=2,
            return_last=False,
        )

    if config.model.name == "deepsleepnet_quality_guided_generator":
        if input_channels < 2:
            raise ValueError("deepsleepnet_quality_guided_generator 需要 EEG+EOG 双通道输入")
        from .robust_schemes.scheme_d_quality_guided_generator import QualityGuidedGeneratorDeepSleepNet

        return QualityGuidedGeneratorDeepSleepNet(
            input_size=input_size,
            n_classes=n_classes,
            seq_length=sequence_length,
            n_rnn_layers=2,
            return_last=False,
        )

    if config.model.name == "deepsleepnet_quality_guided_generator_v2":
        if input_channels < 2:
            raise ValueError("deepsleepnet_quality_guided_generator_v2 需要 EEG+EOG 双通道输入")
        from .robust_schemes.scheme_d_v2_quality_guided_generator import QualityGuidedGeneratorV2DeepSleepNet

        return QualityGuidedGeneratorV2DeepSleepNet(
            input_size=input_size,
            n_classes=n_classes,
            seq_length=sequence_length,
            n_rnn_layers=2,
            return_last=False,
        )

    if config.model.name == "deepsleepnet_quality_guided_generator_v3":
        if input_channels < 2:
            raise ValueError("deepsleepnet_quality_guided_generator_v3 需要 EEG+EOG 双通道输入")
        from .robust_schemes.scheme_d_v3_quality_guided_generator import QualityGuidedGeneratorV3DeepSleepNet

        return QualityGuidedGeneratorV3DeepSleepNet(
            input_size=input_size,
            n_classes=n_classes,
            seq_length=sequence_length,
            n_rnn_layers=2,
            return_last=False,
        )

    raise ValueError(f"暂不支持的模型名称：{config.model.name}")


def _configure_internal_eog_dropout(model, dropout_prob: float, eog_channel_index: int) -> None:
    if hasattr(model, "set_eog_dropout"):
        model.set_eog_dropout(dropout_prob, eog_channel_index)


def _uses_internal_eog_dropout(model) -> bool:
    return bool(getattr(model, "uses_internal_eog_dropout", False))


def _unpack_model_output(output):
    if isinstance(output, tuple):
        logits = output[0]
        auxiliary_loss = output[1] if len(output) > 1 else None
        return logits, auxiliary_loss
    return output, None


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
    """stage2 采用分组学习率。

    已有 CNN encoder 用较小学习率微调；BiLSTM/classifier 用较大学习率。
    对鲁棒模块，generator/fusion 是 stage2 仍需充分学习的新模块，不能跟
    encoder 一起被 1e-6 级别学习率冻结，因此单独放到较大学习率组。
    """

    _, _, Adam = _require_torch()
    cnn_lr = config.training.stage2_cnn_learning_rate or config.training.learning_rate * 0.1
    seq_lr = config.training.stage2_sequence_learning_rate or config.training.learning_rate
    robust_lr = seq_lr

    feature_params = []
    robust_params = []
    for name, parameter in model.feature_extractor.named_parameters():
        if _is_robust_module_parameter(name):
            robust_params.append(parameter)
        else:
            feature_params.append(parameter)

    param_groups = []
    if feature_params:
        param_groups.append({"params": feature_params, "lr": cnn_lr})
    if robust_params:
        param_groups.append({"params": robust_params, "lr": robust_lr})
    param_groups.append(
        {
            "params": list(model.shortcut_projection.parameters())
            + list(model.sequence_model.parameters())
            + list(model.classifier.parameters()),
            "lr": seq_lr,
        }
    )

    return Adam(param_groups)


def _is_robust_module_parameter(name: str) -> bool:
    return (
        name.startswith("generator.")
        or name.startswith("fusion.")
        or name.startswith("quality_sensor.")
        or "residual_logit" in name
    )


def _train_stage1_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch: int,
    eog_dropout_prob: float = 0.0,
    eog_channel_index: int = 1,
) -> float:
    """执行一轮 stage1 训练，并返回样本级平均 loss。"""

    model.train()
    total_loss = 0.0
    total_samples = 0

    progress = _make_progress(dataloader, len(dataloader), f"Epoch {epoch} [train]")
    for batch in progress:
        signals = batch["signals"].to(device)
        if not _uses_internal_eog_dropout(model):
            signals = _apply_random_channel_dropout(signals, eog_channel_index, eog_dropout_prob)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits, auxiliary_loss = _unpack_model_output(model(signals))
        loss = criterion(logits, labels)
        if auxiliary_loss is not None:
            loss = loss + auxiliary_loss
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        if hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{total_loss / max(total_samples, 1):.4f}")

    return total_loss / max(total_samples, 1)


def _train_stage2_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch: int,
    clip_norm: float | None,
    eog_dropout_prob: float = 0.0,
    eog_channel_index: int = 1,
) -> float:
    """执行一轮 stage2 训练，并返回 token 级平均 loss。"""

    torch, _, _ = _require_torch()
    model.train()
    total_loss = 0.0
    total_tokens = 0

    progress = _make_progress(dataloader, len(dataloader), f"Epoch {epoch} [stage2-train]")
    for batch in progress:
        signals = batch["signals"].to(device)
        if not _uses_internal_eog_dropout(model):
            signals = _apply_random_channel_dropout(signals, eog_channel_index, eog_dropout_prob)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits, auxiliary_loss = _unpack_model_output(model(signals))
        loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        if auxiliary_loss is not None:
            loss = loss + auxiliary_loss
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
            logits, _ = _unpack_model_output(model(signals))
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


def _evaluate_sequence_model(
    model,
    dataloader,
    criterion,
    device,
    description: str,
    zero_channel_indices: list[int] | None = None,
) -> dict:
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
            if zero_channel_indices:
                signals = _zero_signal_channels(signals, zero_channel_indices)
            labels = batch["labels"].to(device)
            logits, _ = _unpack_model_output(model(signals))
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


def _zero_signal_channels(signals, channel_indices: list[int]):
    """测试时模拟某些通道脱落；当前序列输入格式应为 [B, S, C, L]。"""

    if signals.ndim != 4:
        raise ValueError("zero_channel_indices 只支持多通道序列输入 [B, S, C, L]")

    num_channels = int(signals.shape[2])
    normalized_indices = []
    for channel_index in channel_indices:
        if not 0 <= channel_index < num_channels:
            raise ValueError(f"channel index {channel_index} 超出范围，当前通道数为 {num_channels}")
        normalized_indices.append(channel_index)

    masked = signals.clone()
    masked[:, :, normalized_indices, :] = 0
    return masked


def _apply_random_channel_dropout(signals, channel_index: int, dropout_prob: float):
    """训练时随机遮住指定通道；stage1 输入为 [B, C, L]，stage2 输入为 [B, S, C, L]。"""

    if dropout_prob <= 0:
        return signals
    if dropout_prob > 1:
        raise ValueError("dropout_prob 必须在 [0, 1] 范围内")
    if signals.ndim not in {3, 4}:
        raise ValueError("EOG dropout 只支持多通道输入 [B, C, L] 或 [B, S, C, L]")

    channel_dim = 1 if signals.ndim == 3 else 2
    num_channels = int(signals.shape[channel_dim])
    if not 0 <= channel_index < num_channels:
        raise ValueError(f"channel index {channel_index} 超出范围，当前通道数为 {num_channels}")

    if signals.ndim == 3:
        drop_mask = signals.new_empty((signals.shape[0], 1, 1)).bernoulli_(dropout_prob).bool()
        masked = signals.clone()
        masked[:, channel_index:channel_index + 1, :] = masked[:, channel_index:channel_index + 1, :].masked_fill(drop_mask, 0)
        return masked

    drop_mask = signals.new_empty((signals.shape[0], 1, 1, 1)).bernoulli_(dropout_prob).bool()
    masked = signals.clone()
    masked[:, :, channel_index:channel_index + 1, :] = masked[:, :, channel_index:channel_index + 1, :].masked_fill(drop_mask, 0)
    return masked


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
