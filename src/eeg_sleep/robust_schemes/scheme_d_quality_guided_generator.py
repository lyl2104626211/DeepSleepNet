from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from eeg_sleep.models import DeepFeatureNet, StackedBidirectionalPeepholeLSTM


class RuleBasedQualitySensor(nn.Module):
    """Rule-based EOG quality sensor.

    First version: detect missing/flatline EOG by variance.
    Output shape is [B, 1], where 1 means reliable and 0 means unreliable.
    """

    def __init__(self, min_variance: float = 1e-12) -> None:
        super().__init__()
        self.min_variance = min_variance

    def forward(self, eog: torch.Tensor) -> torch.Tensor:
        if eog.ndim != 3:
            raise ValueError(f"quality sensor expects [B, 1, L], got {tuple(eog.shape)}")
        variance = eog.var(dim=-1, unbiased=False)
        return (variance > self.min_variance).to(dtype=eog.dtype)


class QualityGuidedGeneratorFeatureNet(nn.Module):
    """Quality-guided EOG feature recovery.

    Minimal module:
    - EEG encoder extracts H_eeg.
    - EOG encoder extracts H_eog_real from observed EOG.
    - Generator predicts H_eog_fake from H_eeg.
    - Rule-based quality c blends real/fake EOG features.
    - Residual fusion keeps output dimension unchanged:

      H_eog_final = c * H_eog_real + (1 - c) * H_eog_fake
      H_fusion = H_eeg + H_eog_final

    During training, this module can internally zero EOG after computing the
    clean EOG teacher feature, so the generator is trained against clean EOG
    features while the classifier sees missing-EOG examples.
    """

    uses_internal_eog_dropout = True

    def __init__(
        self,
        input_size: int = 3000,
        n_classes: int = 5,
        dropout: float = 0.5,
        generator_loss_weight: float = 0.05,
        eog_dropout_prob: float = 0.0,
        eog_channel_index: int = 1,
    ) -> None:
        super().__init__()
        self.eog_dropout_prob = eog_dropout_prob
        self.eog_channel_index = eog_channel_index
        self.generator_loss_weight = generator_loss_weight

        self.eeg_encoder = DeepFeatureNet(input_size=input_size, input_channels=1, n_classes=n_classes, dropout=dropout)
        self.eog_encoder = DeepFeatureNet(input_size=input_size, input_channels=1, n_classes=n_classes, dropout=dropout)
        self.representation_dim = self.eeg_encoder.representation_dim
        self.quality_sensor = RuleBasedQualitySensor()
        self.generator = nn.Sequential(
            nn.Linear(self.representation_dim, self.representation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.representation_dim, self.representation_dim),
        )
        self.classifier = nn.Linear(self.representation_dim, n_classes)

    def set_eog_dropout(self, dropout_prob: float, eog_channel_index: int = 1) -> None:
        self.eog_dropout_prob = dropout_prob
        self.eog_channel_index = eog_channel_index

    @staticmethod
    def _split_inputs(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[1] < 2:
            raise ValueError("QualityGuidedGeneratorFeatureNet expects [B, 2, L] or more channels")
        return inputs[:, 0:1, :], inputs[:, 1:2, :]

    def _maybe_corrupt_eog_for_training(self, eog: torch.Tensor) -> torch.Tensor:
        if not self.training or self.eog_dropout_prob <= 0:
            return eog
        if self.eog_dropout_prob > 1:
            raise ValueError("eog_dropout_prob must be in [0, 1]")

        drop_mask = eog.new_empty((eog.shape[0], 1, 1)).bernoulli_(self.eog_dropout_prob).bool()
        corrupted = eog.clone()
        return corrupted.masked_fill(drop_mask, 0)

    def extract_features_with_aux(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eeg, clean_eog = self._split_inputs(inputs)

        eeg_features = self.eeg_encoder.extract_features(eeg)
        clean_eog_features = self.eog_encoder.extract_features(clean_eog)

        observed_eog = self._maybe_corrupt_eog_for_training(clean_eog)
        observed_eog_features = self.eog_encoder.extract_features(observed_eog)
        fake_eog_features = self.generator(eeg_features)

        quality = self.quality_sensor(observed_eog)
        eog_final = quality * observed_eog_features + (1.0 - quality) * fake_eog_features
        fused_features = eeg_features + eog_final

        generator_loss = F.mse_loss(fake_eog_features, clean_eog_features.detach()) * self.generator_loss_weight
        return fused_features, generator_loss

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features, _ = self.extract_features_with_aux(inputs)
        return features

    def forward(self, inputs: torch.Tensor):
        features, generator_loss = self.extract_features_with_aux(inputs)
        logits = self.classifier(features)
        return logits, generator_loss


class QualityGuidedGeneratorDeepSleepNet(nn.Module):
    """Stage2 DeepSleepNet variant with quality-guided EOG feature recovery."""

    uses_internal_eog_dropout = True

    def __init__(
        self,
        input_size: int = 3000,
        n_classes: int = 5,
        seq_length: int = 25,
        n_rnn_layers: int = 2,
        return_last: bool = False,
        feature_dropout: float = 0.5,
        sequence_dropout: float = 0.5,
        generator_loss_weight: float = 0.05,
        eog_dropout_prob: float = 0.0,
        eog_channel_index: int = 1,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.return_last = return_last
        self.feature_extractor = QualityGuidedGeneratorFeatureNet(
            input_size=input_size,
            n_classes=n_classes,
            dropout=feature_dropout,
            generator_loss_weight=generator_loss_weight,
            eog_dropout_prob=eog_dropout_prob,
            eog_channel_index=eog_channel_index,
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

    def set_eog_dropout(self, dropout_prob: float, eog_channel_index: int = 1) -> None:
        self.feature_extractor.set_eog_dropout(dropout_prob, eog_channel_index)

    def forward(self, inputs: torch.Tensor):
        if inputs.ndim != 4:
            raise ValueError(f"QualityGuidedGeneratorDeepSleepNet expects [B, S, C, L], got {tuple(inputs.shape)}")

        batch_size, sequence_length, input_channels, signal_length = inputs.shape
        if input_channels < 2:
            raise ValueError("QualityGuidedGeneratorDeepSleepNet requires EEG+EOG input")

        flattened_inputs = inputs.reshape(batch_size * sequence_length, input_channels, signal_length)
        epoch_features, generator_loss = self.feature_extractor.extract_features_with_aux(flattened_inputs)
        shortcut_features = self.shortcut_projection(epoch_features)

        sequence_inputs = epoch_features.reshape(batch_size, sequence_length, -1)
        sequence_outputs, _ = self.sequence_model(sequence_inputs)

        shortcut_sequence = shortcut_features.reshape(batch_size, sequence_length, -1)
        combined_outputs = self.output_dropout(sequence_outputs + shortcut_sequence)

        if self.return_last:
            return self.classifier(combined_outputs[:, -1, :]), generator_loss

        logits = self.classifier(combined_outputs.reshape(batch_size * sequence_length, -1))
        return logits.reshape(batch_size, sequence_length, -1), generator_loss
