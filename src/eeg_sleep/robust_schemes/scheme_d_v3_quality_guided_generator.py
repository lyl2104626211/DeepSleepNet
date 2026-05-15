from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from eeg_sleep.models import DeepFeatureNet, StackedBidirectionalPeepholeLSTM


class ContinuousRuleBasedQualitySensor(nn.Module):
    """Lightweight continuous EOG quality sensor.

    The first score detects flatline / zero EOG through low variance. The
    optional second score suppresses very high-variance artifacts. The output
    is in [0, 1], where 1 means reliable and 0 means unreliable.
    """

    def __init__(
        self,
        min_variance: float = 1e-8,
        max_variance: float | None = 1e4,
        sharpness: float = 2.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.min_variance = min_variance
        self.max_variance = max_variance
        self.sharpness = sharpness
        self.eps = eps

    def forward(self, eog: torch.Tensor) -> torch.Tensor:
        if eog.ndim != 3:
            raise ValueError(f"quality sensor expects [B, 1, L], got {tuple(eog.shape)}")

        variance = eog.var(dim=-1, unbiased=False).clamp_min(0.0)
        low_score = variance / (variance + self.min_variance)

        if self.max_variance is None:
            return low_score.clamp(0.0, 1.0)

        log_variance = torch.log(variance.clamp_min(self.eps))
        log_max = math.log(self.max_variance)
        high_score = torch.sigmoid((log_max - log_variance) * self.sharpness)
        return (low_score * high_score).clamp(0.0, 1.0)


class ResidualEOGGenerator(nn.Module):
    """Generate EOG-like features from detached EEG features.

    The generator loss should train this module, not reshape the EEG encoder
    away from the sleep-staging objective.
    """

    def __init__(self, feature_dim: int, dropout: float = 0.1, residual_logit_init: float = -1.0) -> None:
        super().__init__()
        self.base = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.delta = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, feature_dim),
        )
        self.residual_logit = nn.Parameter(torch.tensor(float(residual_logit_init)))

    def forward(self, eeg_features: torch.Tensor) -> torch.Tensor:
        beta = torch.sigmoid(self.residual_logit)
        return self.base(eeg_features) + beta * self.delta(self.norm(eeg_features))


class ResidualQualityFusion(nn.Module):
    """Lightweight residual fusion over EEG, final EOG, and quality score.

    The fusion MLP only receives the already gated EOG feature. This keeps the
    module smaller and avoids passing redundant real/fake EOG branches through
    a randomly initialized high-dimensional fusion layer.
    """

    def __init__(self, feature_dim: int, dropout: float = 0.1, residual_logit_init: float = -2.0) -> None:
        super().__init__()
        fusion_dim = feature_dim * 2 + 1
        self.delta = nn.Sequential(
            nn.Linear(fusion_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, feature_dim),
        )
        self.residual_logit = nn.Parameter(torch.tensor(float(residual_logit_init)))

    def forward(
        self,
        eeg_features: torch.Tensor,
        real_eog_features: torch.Tensor,
        fake_eog_features: torch.Tensor,
        quality: torch.Tensor,
    ) -> torch.Tensor:
        eog_final = quality * real_eog_features + (1.0 - quality) * fake_eog_features
        fusion_inputs = torch.cat([eeg_features, eog_final, quality], dim=-1)
        gamma = torch.sigmoid(self.residual_logit)
        return eeg_features + gamma * self.delta(fusion_inputs)


class QualityGuidedGeneratorV3FeatureNet(nn.Module):
    """Quality-guided EOG feature recovery v3.

    v3 keeps the same plug-in point as v1/v2, but upgrades the robust module:
    - continuous rule-based quality score c in [0, 1];
    - LayerNorm residual EEG->EOG feature generator;
    - lightweight residual fusion MLP conditioned on EEG, gated final EOG, and
      quality score.
    """

    uses_internal_eog_dropout = True

    def __init__(
        self,
        input_size: int = 3000,
        n_classes: int = 5,
        dropout: float = 0.5,
        generator_mse_loss_weight: float = 0.01,
        generator_cosine_loss_weight: float = 0.01,
        eog_dropout_prob: float = 0.0,
        eog_channel_index: int = 1,
    ) -> None:
        super().__init__()
        self.eog_dropout_prob = eog_dropout_prob
        self.eog_channel_index = eog_channel_index
        self.generator_mse_loss_weight = generator_mse_loss_weight
        self.generator_cosine_loss_weight = generator_cosine_loss_weight

        self.eeg_encoder = DeepFeatureNet(input_size=input_size, input_channels=1, n_classes=n_classes, dropout=dropout)
        self.eog_encoder = DeepFeatureNet(input_size=input_size, input_channels=1, n_classes=n_classes, dropout=dropout)
        self.representation_dim = self.eeg_encoder.representation_dim

        self.quality_sensor = ContinuousRuleBasedQualitySensor()
        self.generator = ResidualEOGGenerator(self.representation_dim)
        self.fusion = ResidualQualityFusion(self.representation_dim)
        self.classifier = nn.Linear(self.representation_dim, n_classes)

    def set_eog_dropout(self, dropout_prob: float, eog_channel_index: int = 1) -> None:
        self.eog_dropout_prob = dropout_prob
        self.eog_channel_index = eog_channel_index

    @staticmethod
    def _split_inputs(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[1] < 2:
            raise ValueError("QualityGuidedGeneratorV3FeatureNet expects [B, 2, L] or more channels")
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
        fake_eog_features = self.generator(eeg_features.detach())

        quality = self.quality_sensor(observed_eog)
        fused_features = self.fusion(eeg_features, observed_eog_features, fake_eog_features, quality)

        clean_eog_teacher = clean_eog_features.detach()
        generator_mse_loss = F.mse_loss(fake_eog_features, clean_eog_teacher) * self.generator_mse_loss_weight
        generator_cosine_loss = (
            1.0 - F.cosine_similarity(fake_eog_features, clean_eog_teacher, dim=-1).mean()
        ) * self.generator_cosine_loss_weight
        generator_loss = generator_mse_loss + generator_cosine_loss
        return fused_features, generator_loss

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features, _ = self.extract_features_with_aux(inputs)
        return features

    def forward(self, inputs: torch.Tensor):
        features, generator_loss = self.extract_features_with_aux(inputs)
        logits = self.classifier(features)
        return logits, generator_loss


class QualityGuidedGeneratorV3DeepSleepNet(nn.Module):
    """Stage2 DeepSleepNet variant with v3 quality-guided EOG recovery."""

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
        generator_mse_loss_weight: float = 0.01,
        generator_cosine_loss_weight: float = 0.01,
        eog_dropout_prob: float = 0.0,
        eog_channel_index: int = 1,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.return_last = return_last
        self.feature_extractor = QualityGuidedGeneratorV3FeatureNet(
            input_size=input_size,
            n_classes=n_classes,
            dropout=feature_dropout,
            generator_mse_loss_weight=generator_mse_loss_weight,
            generator_cosine_loss_weight=generator_cosine_loss_weight,
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
            raise ValueError(f"QualityGuidedGeneratorV3DeepSleepNet expects [B, S, C, L], got {tuple(inputs.shape)}")

        batch_size, sequence_length, input_channels, signal_length = inputs.shape
        if input_channels < 2:
            raise ValueError("QualityGuidedGeneratorV3DeepSleepNet requires EEG+EOG input")

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
