from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


@dataclass
class ModelSummary:
    name: str
    description: str


def _compute_same_padding(length: int, kernel_size: int, stride: int, dilation: int = 1) -> tuple[int, int]:
    """按 TensorFlow SAME padding 的规则计算左右补零。"""

    output_length = math.ceil(length / stride)
    effective_kernel = (kernel_size - 1) * dilation + 1
    total_padding = max((output_length - 1) * stride + effective_kernel - length, 0)
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding
    return left_padding, right_padding


class SamePadConv1d(nn.Module):
    """保持和原始 TensorFlow 实现一致的 SAME padding 卷积。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        left_padding, right_padding = _compute_same_padding(
            length=inputs.shape[-1],
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        return self.conv(F.pad(inputs, (left_padding, right_padding)))


class SamePadMaxPool1d(nn.Module):
    """保持和原始 TensorFlow 实现一致的 SAME padding 池化。"""

    def __init__(self, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        left_padding, right_padding = _compute_same_padding(
            length=inputs.shape[-1],
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        padded_inputs = F.pad(inputs, (left_padding, right_padding), value=float("-inf"))
        return self.pool(padded_inputs)


class ConvBnReluBlock(nn.Module):
    """卷积主干最基础的一个块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            SamePadConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class CNNBranch(nn.Module):
    """DeepSleepNet 的一个 CNN 分支。"""

    def __init__(
        self,
        first_kernel_size: int,
        first_stride: int,
        first_pool_size: int,
        first_pool_stride: int,
        later_kernel_size: int,
        later_pool_size: int,
        later_pool_stride: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBnReluBlock(1, 64, first_kernel_size, first_stride),
            SamePadMaxPool1d(first_pool_size, first_pool_stride),
            nn.Dropout(p=dropout),
            ConvBnReluBlock(64, 128, later_kernel_size, 1),
            ConvBnReluBlock(128, 128, later_kernel_size, 1),
            ConvBnReluBlock(128, 128, later_kernel_size, 1),
            SamePadMaxPool1d(later_pool_size, later_pool_stride),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.features(inputs)


class PeepholeLSTMCell(nn.Module):
    """最小实现版 peephole LSTMCell。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        forget_bias: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.input_linear = nn.Linear(input_size, hidden_size * 4)
        self.hidden_linear = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.peephole_i = nn.Parameter(torch.zeros(hidden_size))
        self.peephole_f = nn.Parameter(torch.zeros(hidden_size))
        self.peephole_o = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        inputs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_state, cell_state = state
        gates = self.input_linear(inputs) + self.hidden_linear(hidden_state)

        # 顺序保持和 TensorFlow v1 LSTMCell 一致：i, j, f, o。
        input_gate, candidate_gate, forget_gate, output_gate = gates.chunk(4, dim=-1)

        input_gate = torch.sigmoid(input_gate + self.peephole_i * cell_state)
        candidate_gate = torch.tanh(candidate_gate)
        forget_gate = torch.sigmoid(forget_gate + self.forget_bias + self.peephole_f * cell_state)

        next_cell_state = forget_gate * cell_state + input_gate * candidate_gate
        output_gate = torch.sigmoid(output_gate + self.peephole_o * next_cell_state)
        next_hidden_state = output_gate * torch.tanh(next_cell_state)
        return next_hidden_state, next_cell_state


class StackedBidirectionalPeepholeLSTM(nn.Module):
    """两层双向 peephole LSTM。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        output_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dropout = output_dropout
        self.forward_layers = nn.ModuleList()
        self.backward_layers = nn.ModuleList()

        layer_input_size = input_size
        for _ in range(num_layers):
            self.forward_layers.append(PeepholeLSTMCell(layer_input_size, hidden_size))
            self.backward_layers.append(PeepholeLSTMCell(layer_input_size, hidden_size))
            layer_input_size = hidden_size * 2

    def _init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return zeros, zeros.clone()

    def _run_direction(
        self,
        cell: PeepholeLSTMCell,
        inputs: torch.Tensor,
        time_indices: range,
    ) -> tuple[list[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        state = self._init_state(inputs.shape[0], inputs.device, inputs.dtype)
        outputs: list[torch.Tensor] = []

        for time_idx in time_indices:
            hidden_state, cell_state = cell(inputs[:, time_idx, :], state)
            state = (hidden_state, cell_state)
            outputs.append(F.dropout(hidden_state, p=self.output_dropout, training=self.training))

        return outputs, state

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, list[dict[str, tuple[torch.Tensor, torch.Tensor]]]]:
        _, sequence_length, _ = inputs.shape
        layer_input = inputs
        final_states: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] = []

        for forward_cell, backward_cell in zip(self.forward_layers, self.backward_layers):
            forward_outputs, forward_state = self._run_direction(
                forward_cell,
                layer_input,
                range(sequence_length),
            )
            backward_outputs, backward_state = self._run_direction(
                backward_cell,
                layer_input,
                range(sequence_length - 1, -1, -1),
            )

            forward_stack = torch.stack(forward_outputs, dim=1)
            backward_stack = torch.stack(list(reversed(backward_outputs)), dim=1)
            layer_input = torch.cat([forward_stack, backward_stack], dim=-1)
            final_states.append({"forward": forward_state, "backward": backward_state})

        return layer_input, final_states


class DeepFeatureNet(nn.Module):
    """第一阶段模型，只做单个 epoch 的特征提取和分类。
    输入张量形状(B*Sequence,dim)
    """

    def __init__(
        self,
        input_size: int = 3000,
        n_classes: int = 5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.dropout = nn.Dropout(p=dropout)

        # 一个分支偏向短时间尺度，一个分支偏向长时间尺度。
        self.small_cnn = CNNBranch(
            first_kernel_size=50,
            first_stride=6,
            first_pool_size=8,
            first_pool_stride=8,
            later_kernel_size=8,
            later_pool_size=4,
            later_pool_stride=4,
            dropout=dropout,
        )
        self.large_cnn = CNNBranch(
            first_kernel_size=400,
            first_stride=50,
            first_pool_size=4,
            first_pool_stride=4,
            later_kernel_size=6,
            later_pool_size=2,
            later_pool_stride=2,
            dropout=dropout,
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_size)
            small_dim = self.small_cnn(dummy_input).flatten(start_dim=1).shape[-1]
            large_dim = self.large_cnn(dummy_input).flatten(start_dim=1).shape[-1]

        self.representation_dim = small_dim + large_dim
        self.classifier = nn.Linear(self.representation_dim, n_classes)

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            return inputs.unsqueeze(1)
        if inputs.ndim == 3 and inputs.shape[1] == 1:
            return inputs
        raise ValueError(
            f"DeepFeatureNet 期望输入形状是 [B, L] 或 [B, 1, L]，实际收到 {tuple(inputs.shape)}"
        )

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        prepared_inputs = self._prepare_inputs(inputs)
        small_features = self.small_cnn(prepared_inputs).flatten(start_dim=1)
        large_features = self.large_cnn(prepared_inputs).flatten(start_dim=1)
        features = torch.cat([small_features, large_features], dim=1)
        return self.dropout(features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(inputs))


class DeepSleepNet(nn.Module):
    """完整的 DeepSleepNet，两阶段中的第二阶段模型。"""

    def __init__(
        self,
        input_size: int = 3000,
        n_classes: int = 5,
        seq_length: int = 25,
        n_rnn_layers: int = 2,
        return_last: bool = False,
        feature_dropout: float = 0.5,
        sequence_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.return_last = return_last

        self.feature_extractor = DeepFeatureNet(
            input_size=input_size,
            n_classes=n_classes,
            dropout=feature_dropout,
        )
        self.shortcut_projection = nn.Sequential(
            nn.Linear(self.feature_extractor.representation_dim, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        self.sequence_model = StackedBidirectionalPeepholeLSTM(
            input_size=self.feature_extractor.representation_dim,
            hidden_size=512,
            num_layers=n_rnn_layers,
            output_dropout=sequence_dropout,
        )
        self.output_dropout = nn.Dropout(p=sequence_dropout)
        self.classifier = nn.Linear(1024, n_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"DeepSleepNet 期望输入形状是 [B, S, L]，实际收到 {tuple(inputs.shape)}"
            )

        batch_size, sequence_length, signal_length = inputs.shape

        # 先对每个 epoch 共享同一个 CNN 特征提取器。
        flattened_inputs = inputs.reshape(batch_size * sequence_length, signal_length)
        epoch_features = self.feature_extractor.extract_features(flattened_inputs)
        shortcut_features = self.shortcut_projection(epoch_features)

        # 再把特征还原成序列，交给双向 LSTM 建模上下文。
        sequence_inputs = epoch_features.reshape(batch_size, sequence_length, -1)
        sequence_outputs, _ = self.sequence_model(sequence_inputs)

        shortcut_sequence = shortcut_features.reshape(batch_size, sequence_length, -1)
        combined_outputs = self.output_dropout(sequence_outputs + shortcut_sequence)

        if self.return_last:
            return self.classifier(combined_outputs[:, -1, :])

        logits = self.classifier(combined_outputs.reshape(batch_size * sequence_length, -1))
        return logits.reshape(batch_size, sequence_length, -1)


def build_model_summary(config: ModelConfig) -> ModelSummary:
    if config.name != "deepsleepnet_baseline":
        raise ValueError(f"暂不支持的模型名称：{config.name}")

    return ModelSummary(
        name=config.name,
        description=(
            "DeepSleepNet baseline：输入是 30 秒、100 Hz 的单通道 EEG（3000 点）；"
            "前端用双分支 CNN 提取短时和长时特征；"
            "后端用 2 层双向 peephole LSTM 建模时序上下文；"
            "最后把 shortcut 分支和时序输出相加，再做 5 分类。"
        ),
    )


def build_model(
    config: ModelConfig,
    input_size: int = 3000,
    n_classes: int = 5,
    seq_length: int = 25,
) -> nn.Module:
    if config.name != "deepsleepnet_baseline":
        raise ValueError(f"暂不支持的模型名称：{config.name}")

    return DeepSleepNet(
        input_size=input_size,
        n_classes=n_classes,
        seq_length=seq_length,
        n_rnn_layers=2,
        return_last=False,
    )
