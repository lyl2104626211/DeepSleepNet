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
    output_length = math.ceil(length / stride)
    effective_kernel = (kernel_size - 1) * dilation + 1
    total_padding = max((output_length - 1) * stride + effective_kernel - length, 0)
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding
    return left_padding, right_padding


class SamePadConv1d(nn.Module):
    """与原始 TensorFlow 实现一致的 SAME padding 1D 卷积。"""

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
        padded_inputs = F.pad(inputs, (left_padding, right_padding))
        return self.conv(padded_inputs)


class SamePadMaxPool1d(nn.Module):
    """与原始 TensorFlow 实现一致的 SAME padding 1D 最大池化。"""

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
    """卷积 -> BN -> ReLU，与论文和官方实现顺序一致。"""

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
    """DeepSleepNet 的单个 CNN 分支。"""

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
    """带 peephole 的 LSTMCell。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        forget_bias: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # TensorFlow 原版 LSTMCell 默认会给 forget gate 额外加 1.0，
        # 这样训练初期更倾向于“先记住已有状态”，更接近官方实现。
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
        # 这里按 TensorFlow v1 LSTMCell 的顺序拆门：i, j, f, o
        # 其中 j 对应候选记忆，不是 forget gate。
        input_gate, candidate_gate, forget_gate, output_gate = gates.chunk(4, dim=-1)

        input_gate = torch.sigmoid(input_gate + self.peephole_i * cell_state)
        candidate_gate = torch.tanh(candidate_gate)
        forget_gate = torch.sigmoid(
            forget_gate + self.forget_bias + self.peephole_f * cell_state
        )

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
        self.num_layers = num_layers
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
        return (
            torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
            torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, tuple[torch.Tensor, torch.Tensor]]]]:
        batch_size, sequence_length, _ = inputs.shape
        layer_input = inputs
        final_states: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] = []

        for forward_cell, backward_cell in zip(self.forward_layers, self.backward_layers):
            forward_state = self._init_state(batch_size, layer_input.device, layer_input.dtype)
            backward_state = self._init_state(batch_size, layer_input.device, layer_input.dtype)

            forward_outputs: list[torch.Tensor] = []
            backward_outputs: list[torch.Tensor | None] = [None] * sequence_length

            for time_idx in range(sequence_length):
                hidden_state, cell_state = forward_cell(layer_input[:, time_idx, :], forward_state)
                forward_state = (hidden_state, cell_state)
                forward_outputs.append(
                    F.dropout(hidden_state, p=self.output_dropout, training=self.training)
                )

            for time_idx in range(sequence_length - 1, -1, -1):
                hidden_state, cell_state = backward_cell(layer_input[:, time_idx, :], backward_state)
                backward_state = (hidden_state, cell_state)
                backward_outputs[time_idx] = F.dropout(
                    hidden_state,
                    p=self.output_dropout,
                    training=self.training,
                )

            forward_stack = torch.stack(forward_outputs, dim=1)
            backward_stack = torch.stack([output for output in backward_outputs if output is not None], dim=1)
            layer_input = torch.cat([forward_stack, backward_stack], dim=-1)
            final_states.append(
                {
                    "forward": forward_state,
                    "backward": backward_state,
                }
            )

        return layer_input, final_states


class DeepFeatureNet(nn.Module):
    """DeepSleepNet 第一阶段使用的特征提取模型。"""

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
        raise ValueError(f"DeepFeatureNet 期望输入形状为 [B, L] 或 [B, 1, L]，实际收到 {tuple(inputs.shape)}")

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        prepared_inputs = self._prepare_inputs(inputs)
        small_features = self.small_cnn(prepared_inputs).flatten(start_dim=1)
        large_features = self.large_cnn(prepared_inputs).flatten(start_dim=1)
        concatenated_features = torch.cat([small_features, large_features], dim=1)
        return self.dropout(concatenated_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(inputs)
        return self.classifier(features)


class DeepSleepNet(nn.Module):
    """DeepSleepNet 完整模型。"""

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

        # 第一部分：对单个 30 秒 epoch 做双分支 CNN 特征提取。
        self.feature_extractor = DeepFeatureNet(
            input_size=input_size,
            n_classes=n_classes,
            dropout=feature_dropout,
        )
        # shortcut 分支把 CNN 特征投影到 1024 维，后面和双向 LSTM 输出相加。
        self.shortcut_projection = nn.Sequential(
            nn.Linear(self.feature_extractor.representation_dim, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        # 第二部分：对连续 epoch 序列建模上下文。
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
            raise ValueError(f"DeepSleepNet 期望输入形状为 [B, S, L]，实际收到 {tuple(inputs.shape)}")

        batch_size, sequence_length, signal_length = inputs.shape
        # 先把序列拆平，让每个 epoch 共用同一个 CNN 特征提取器。
        flattened_inputs = inputs.reshape(batch_size * sequence_length, signal_length)

        epoch_features = self.feature_extractor.extract_features(flattened_inputs)
        shortcut_features = self.shortcut_projection(epoch_features)

        # 再恢复成 [B, S, F]，交给双向 LSTM 建模时间上下文。
        sequence_inputs = epoch_features.reshape(batch_size, sequence_length, -1)
        sequence_outputs, _ = self.sequence_model(sequence_inputs)

        shortcut_sequence = shortcut_features.reshape(batch_size, sequence_length, -1)
        # 论文里的 residual/shortcut 思路：时序输出 + 直接投影特征。
        combined_outputs = sequence_outputs + shortcut_sequence
        combined_outputs = self.output_dropout(combined_outputs)

        if self.return_last:
            last_logits = self.classifier(combined_outputs[:, -1, :])
            return last_logits

        logits = self.classifier(combined_outputs.reshape(batch_size * sequence_length, -1))
        return logits.reshape(batch_size, sequence_length, -1)


def build_model_summary(config: ModelConfig) -> ModelSummary:
    if config.name != "deepsleepnet_baseline":
        raise ValueError(f"暂不支持的模型名称：{config.name}")

    return ModelSummary(
        name=config.name,
        description=(
            "DeepSleepNet baseline：输入为 30 秒、100 Hz 的单通道 EEG（3000 点）；"
            "前端使用双分支 CNN，分别采用 conv50/stride6 和 conv400/stride50 的首层设置；"
            "后端为 2 层双向 peephole LSTM（每个方向 512 维），"
            "并带有 1024 维 shortcut 投影与残差相加，最后输出 5 类 softmax。"
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
