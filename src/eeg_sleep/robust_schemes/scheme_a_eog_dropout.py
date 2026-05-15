from __future__ import annotations

import torch


def apply_eog_dropout(
    signals: torch.Tensor,
    dropout_prob: float = 0.5,
    eog_channel_index: int = 1,
) -> torch.Tensor:
    """Randomly zero the EOG channel during training.

    Supported shapes:
    - stage1: [B, C, L]
    - stage2: [B, S, C, L]

    The dropout mask is sampled per sample/sequence, not per timestep. This
    simulates a whole EOG channel being unavailable for that training example.
    """

    if dropout_prob <= 0:
        return signals
    if dropout_prob > 1:
        raise ValueError("dropout_prob must be in [0, 1]")
    if signals.ndim not in {3, 4}:
        raise ValueError("expected [B, C, L] or [B, S, C, L]")

    channel_dim = 1 if signals.ndim == 3 else 2
    num_channels = int(signals.shape[channel_dim])
    if not 0 <= eog_channel_index < num_channels:
        raise ValueError(f"channel index {eog_channel_index} out of range for {num_channels} channels")

    masked = signals.clone()
    if signals.ndim == 3:
        drop_mask = signals.new_empty((signals.shape[0], 1, 1)).bernoulli_(dropout_prob).bool()
        eog = masked[:, eog_channel_index:eog_channel_index + 1, :]
        masked[:, eog_channel_index:eog_channel_index + 1, :] = eog.masked_fill(drop_mask, 0)
        return masked

    drop_mask = signals.new_empty((signals.shape[0], 1, 1, 1)).bernoulli_(dropout_prob).bool()
    eog = masked[:, :, eog_channel_index:eog_channel_index + 1, :]
    masked[:, :, eog_channel_index:eog_channel_index + 1, :] = eog.masked_fill(drop_mask, 0)
    return masked


def zero_eog(
    signals: torch.Tensor,
    eog_channel_index: int = 1,
) -> torch.Tensor:
    """Deterministically zero EOG for missing-modality evaluation."""

    if signals.ndim not in {3, 4}:
        raise ValueError("expected [B, C, L] or [B, S, C, L]")

    channel_dim = 1 if signals.ndim == 3 else 2
    num_channels = int(signals.shape[channel_dim])
    if not 0 <= eog_channel_index < num_channels:
        raise ValueError(f"channel index {eog_channel_index} out of range for {num_channels} channels")

    masked = signals.clone()
    if signals.ndim == 3:
        masked[:, eog_channel_index:eog_channel_index + 1, :] = 0
    else:
        masked[:, :, eog_channel_index:eog_channel_index + 1, :] = 0
    return masked
