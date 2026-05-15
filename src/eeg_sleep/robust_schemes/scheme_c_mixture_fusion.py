from __future__ import annotations

import torch
from torch import nn

from eeg_sleep.models import DeepFeatureNet, StackedBidirectionalPeepholeLSTM


class MixtureFusionFeatureNet(nn.Module):
    """Softmax mixture of EEG and EOG features.

    Minimal idea:
    - Encode EEG and EOG separately.
    - Predict two modality weights with softmax.
    - Fuse as w_eeg * f_eeg + w_eog * f_eog.

    This is more flexible than EEG-main gated fusion, but slightly less aligned
    with the assumption that EEG is the stable primary modality.
    """

    def __init__(
        self,
        input_size: int = 3000,
        n_classes: int = 5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.eeg_encoder = DeepFeatureNet(input_size=input_size, input_channels=1, n_classes=n_classes, dropout=dropout)
        self.eog_encoder = DeepFeatureNet(input_size=input_size, input_channels=1, n_classes=n_classes, dropout=dropout)
        self.representation_dim = self.eeg_encoder.representation_dim
        self.weight_net = nn.Sequential(
            nn.Linear(self.representation_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )
        self.classifier = nn.Linear(self.representation_dim, n_classes)

    @staticmethod
    def _split_inputs(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[1] < 2:
            raise ValueError("MixtureFusionFeatureNet expects [B, 2, L] or more channels")
        return inputs[:, 0:1, :], inputs[:, 1:2, :]

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        eeg, eog = self._split_inputs(inputs)
        eeg_features = self.eeg_encoder.extract_features(eeg)
        eog_features = self.eog_encoder.extract_features(eog)
        weights = torch.softmax(self.weight_net(torch.cat([eeg_features, eog_features], dim=-1)), dim=-1)
        eeg_weight = weights[:, 0:1]
        eog_weight = weights[:, 1:2]
        return eeg_weight * eeg_features + eog_weight * eog_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(inputs))


class MixtureFusionDeepSleepNet(nn.Module):
    """Stage2 DeepSleepNet variant using MixtureFusionFeatureNet."""

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
        self.feature_extractor = MixtureFusionFeatureNet(
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
        if inputs.ndim != 4:
            raise ValueError(f"MixtureFusionDeepSleepNet expects [B, S, C, L], got {tuple(inputs.shape)}")

        batch_size, sequence_length, input_channels, signal_length = inputs.shape
        if input_channels < 2:
            raise ValueError("MixtureFusionDeepSleepNet requires EEG+EOG input")

        flattened_inputs = inputs.reshape(batch_size * sequence_length, input_channels, signal_length)
        epoch_features = self.feature_extractor.extract_features(flattened_inputs)
        shortcut_features = self.shortcut_projection(epoch_features)

        sequence_inputs = epoch_features.reshape(batch_size, sequence_length, -1)
        sequence_outputs, _ = self.sequence_model(sequence_inputs)

        shortcut_sequence = shortcut_features.reshape(batch_size, sequence_length, -1)
        combined_outputs = self.output_dropout(sequence_outputs + shortcut_sequence)

        if self.return_last:
            return self.classifier(combined_outputs[:, -1, :])

        logits = self.classifier(combined_outputs.reshape(batch_size * sequence_length, -1))
        return logits.reshape(batch_size, sequence_length, -1)
