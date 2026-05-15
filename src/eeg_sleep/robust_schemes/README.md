# Robust EOG Schemes

This folder keeps experimental robustness ideas separate from the current
DeepSleepNet baseline. The goal is to make each idea easy to inspect before it
is wired into `trainer.py` and config selection.

## scheme_a_eog_dropout.py

Minimal training-time robustness baseline.

- Randomly zero the EOG channel during training.
- Deterministically zero EOG during missing-EOG evaluation.
- This is already the strongest baseline to beat.

Current usage is implemented in `trainer.py` through:

```bash
--eog-dropout-prob 0.5
--mask-eog
```

## scheme_b_gated_fusion.py

Recommended first model module.

```text
fused = f_eeg + sigmoid(MLP([f_eeg, f_eog])) * f_eog
```

Rationale:

- EEG remains the stable main modality.
- EOG is injected only through a learned reliability gate.
- It is easy to explain and compare against EOG dropout training.

Classes:

- `GatedFusionFeatureNet`
- `GatedFusionDeepSleepNet`

## scheme_c_mixture_fusion.py

More flexible modality weighting.

```text
[w_eeg, w_eog] = softmax(MLP([f_eeg, f_eog]))
fused = w_eeg * f_eeg + w_eog * f_eog
```

Rationale:

- Lets the model dynamically choose between modalities.
- More flexible than gated EOG injection.
- Slightly less constrained than the EEG-main assumption.

Classes:

- `MixtureFusionFeatureNet`
- `MixtureFusionDeepSleepNet`

## scheme_d_quality_guided_generator.py

Full minimal robustness module.

```text
H_eeg = EEGEncoder(x_eeg)
H_eog_real = EOGEncoder(x_eog_observed)
H_eog_fake = Generator(H_eeg)
c = QualitySensor(x_eog_observed)

H_eog_final = c * H_eog_real + (1 - c) * H_eog_fake
H_fusion = H_eeg + H_eog_final
```

Rationale:

- If EOG is reliable, use real EOG features.
- If EOG is missing or flatline, replace EOG with EEG-generated EOG features.
- The residual fusion keeps the feature dimension unchanged, so the downstream BiLSTM does not need to change.

Training:

```text
loss = CE + lambda * MSE(H_eog_fake, H_eog_clean.detach())
```

The module handles EOG dropout internally. It computes clean EOG teacher features first, then corrupts the observed EOG for classification. This avoids training the generator against already-zeroed EOG features.

## Recommended Test Order

1. Keep `scheme_a` as baseline.
2. Test `scheme_b` first with training-time EOG dropout.
3. Test `scheme_d` if gate-only fusion is not enough.
4. Test `scheme_c` only if you need a more flexible modality-weighting comparison.

Use the same 5 pilot folds first:

```text
fold_00 fold_03 fold_07 fold_13 fold_17
```

Compare:

- normal Macro-F1
- EOG=0 Macro-F1
- REM F1 under EOG=0
