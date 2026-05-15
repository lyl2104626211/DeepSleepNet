# Paper Reading Notes For EOG-Missing Robust Sleep Staging

Date: 2026-05-15

Scope: local papers under `paper/`. Focus: what can improve the current robust sleep staging experiments under missing or corrupted EOG.

## Current Project Fit

Your current method direction is well aligned with the literature:

- EOG reliability / quality sensor: matches imperfect-modality and adaptive-fusion lines.
- EEG-to-EOG feature generator: matches missing-modality imagination / imputation / representation generation.
- Real/generated EOG dynamic fusion: matches coordinated representation, attention fusion, and modality-aware routing.
- Classification plus feature teacher loss: matches multi-task supervised loss, contrastive alignment, and representation distillation ideas.

The main weakness in current results is not the idea; it is experimental evidence. The literature suggests that the next gain should come from stronger evaluation and targeted training scenarios, not simply adding a larger module.

## Highest-Value Experimental Changes

### 1. Add a realistic EOG corruption benchmark

Current `EOG=0` is necessary but too narrow. Add the following test modes for every method:

| Scenario | Purpose | Suggested implementation |
| --- | --- | --- |
| `zero` | complete missing EOG | existing EOG=0 |
| `flatline` | detached sensor / constant value | replace EOG by per-record median or zero after normalization |
| `gaussian_noise` | noisy EOG | add noise at SNR levels such as 20, 10, 5, 0 dB |
| `high_amp_artifact` | motion / electrode artifact | inject random bursts with amplitude from high EOG percentiles |
| `drift` | slow baseline drift | add low-frequency sinusoid or random walk |
| `chunk_missing` | real clinical interruption | mask continuous chunks across multiple epochs |
| `partial_epoch_dropout` | short local signal loss | mask random sub-windows inside epochs |

Report normal and corrupted results side by side:

```text
normal
zero
flatline
noise_20db / noise_10db / noise_5db / noise_0db
artifact
drift
chunk_missing
partial_epoch_dropout
```

This is directly supported by CoRe-Sleep and CIMSleepNet. CoRe-Sleep explicitly evaluates noisy modality cases, while CIMSleepNet evaluates multiple missing rates and complete missing cases.

### 2. Add missing-rate curves, not only one missing point

CIMSleepNet reports robustness under increasing missing rates. Your current `p=0.5` is useful, but a curve is more convincing:

```text
EOG missing/corruption rate: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
```

For each rate, evaluate:

- EEG+EOG baseline
- EOG dropout p=0.5
- quality-guided generator v2
- v3-safe-lr if fold_00 recovers

This can turn a small average gain into a stronger robustness story if your method degrades more slowly as corruption increases.

### 3. Calibrate the EOG quality sensor from data

The current v3 quality sensor uses fixed variance thresholds. Literature suggests quality should be tied to actual sensor behavior.

Recommended minimal calibration:

- Compute per-epoch EOG variance, peak-to-peak amplitude, absolute mean, and 95th percentile amplitude on clean training data.
- Define low-quality by percentile thresholds instead of fixed constants:
  - flatline: variance below 1st percentile or near zero
  - artifact/noise: variance or peak-to-peak above 99th percentile
  - drift: large low-frequency trend or large epoch mean shift
- Save fold-specific thresholds from train split only.

CoRe-Sleep uses standard-deviation changes over long chunks to identify noisy patients. Use this as a validation diagnostic: when a chunk has abnormal EOG statistics, the learned/heuristic quality `c` should drop.

### 4. Add quality diagnostics to prove the sensor works

For each corruption mode, log and plot:

- mean and std of quality score `c`
- quality score by sleep stage
- quality score by corruption severity
- correlation between `c` and EOG corruption type/severity
- performance grouped by low/medium/high `c`

This is important because your method claim depends on EOG reliability estimation. Without these diagnostics, reviewers may treat the quality sensor as an unverified heuristic.

### 5. Preserve v2 as the stable method; use v3 only if fold_00 recovers

Current evidence says v2 is the best stable version. v3-safe-lr is worth testing, but do not keep increasing complexity if it does not clearly beat v2.

Decision rule:

```text
If v3-safe-lr fold_00 normal and EOG=0 are close to or better than v2:
    run fixed 5-fold v3-safe-lr
else:
    freeze v2 or make a v2.5 with only calibrated quality/corruption training
```

The literature supports lightweight improvements if experiments are strong. SalientSleepNet and CoRe-Sleep both show that good fusion/training design can matter more than very large model size.

### 6. Add stage-wise analysis focused on N1 and REM

SalientSleepNet explicitly notes that EOG is especially useful for distinguishing N1 and REM because eye movements differ while EEG can be similar. Your current results also show REM collapses under EOG=0 and N1 remains weak.

For every main table, include class-wise F1:

```text
W, N1, N2, N3, REM
```

Add a small REM/N1 confusion table or figure under:

- normal
- EOG=0
- noise/artifact
- best robust method

This gives a physiological reason for your method, not only a metric gain.

### 7. Use U-Sleep as the first cross-backbone validation

U-Sleep is the best next backbone because it is explicitly designed for arbitrary EEG/EOG channel combinations and is a strong clinical-style system.

Minimal cross-backbone target:

- Do not reproduce every U-Sleep training detail.
- Implement only the same robust module idea around EEG/EOG feature fusion if feasible.
- Evaluate on the same corruption modes used for DeepSleepNet.

If resources are limited, report a smaller validation:

```text
DeepSleepNet: full main experiments and ablations
U-Sleep: key corruption benchmark on best method vs EOG dropout
EEGPT: exploratory representation-backbone result or discussion
```

### 8. Add modality-dropout and corruption training variants

U-Sleep uses noise replacement augmentation on EEG/EOG segments or whole channels. Your current EOG dropout is strong, but it only teaches zero-missing behavior.

Add one training variant:

```text
EOG corruption augmentation p=0.5:
    sample one of zero / flatline / noise / drift / artifact / chunk_missing
```

Compare:

- EOG dropout p=0.5
- EOG corruption augmentation p=0.5
- quality-guided generator trained with the same corruption augmentation

This tests whether your quality sensor and generator are useful beyond plain modality dropout.

### 9. Add an ablation table that maps to the method claim

Use the following ablations:

| Variant | Meaning |
| --- | --- |
| full model | quality + generator + gated fusion + teacher loss |
| no quality | fixed fusion or no quality score |
| no generator | real EOG only with quality gate |
| no teacher loss | classification loss only |
| no anti-noise gating | allow raw real EOG features through fusion |
| calibrated quality only | v2/v3 with percentile thresholds |

This maps cleanly to the missing-modality survey taxonomy: representation generation, adaptive fusion, and distillation/alignment.

### 10. Add statistical reporting

For publication, small gains over EOG dropout need statistical support:

- fixed folds
- paired comparison across folds
- bootstrap confidence intervals over subjects/epochs if feasible
- report mean +/- std across folds
- use pooled confusion matrix for main metrics and fold-level table for significance

If the improvement remains tiny, emphasize robustness under multiple corruption scenarios rather than a single average gain.

## Paper-by-Paper Notes

### CIMSleepNet 2024

File: `paper/CIMSleepNet_2024.pdf`

Core idea:

- Handles arbitrary missing multimodal physiological signals.
- Uses a modal awareness imagination module to recover missing modalities.
- Uses semantic and modal calibration contrastive learning to align recovered data with real distributions while preserving modality-specific information.
- Adds multi-level temporal attention.

Useful for your experiment:

- Directly relevant competitor/related work.
- Their strongest experimental pattern is missing-rate curves plus ablation.
- They train with missing rate `rho=0.5` and test complete missing modality.
- Their ablation shows both imputation and contrastive calibration matter.

What to borrow:

- Add missing-rate curves from 0.0 to 0.5.
- Add t-SNE or feature visualization only if time allows.
- Add ablation proving generator and teacher/alignment loss matter.
- Distinguish your method by focusing on EOG quality/corruption, not arbitrary missing modality in general.

### CoRe-Sleep 2023

File: `paper/CoRe-Sleep_2023.pdf`

Core idea:

- Coordinated representation fusion for EEG/EOG.
- Separate unimodal and multimodal branches.
- Uses multi-supervised losses and an alignment loss.
- Robust to missing modalities and noisy modality segments.

Useful for your experiment:

- Very relevant to "imperfect modality" framing.
- They test missing EEG/EOG without retraining.
- They identify noisy patients by abnormal standard deviation over long chunks and evaluate on a noisy subset.
- They show multi-task unimodal losses improve robustness.

What to borrow:

- Add noisy EOG evaluation based on statistical abnormality.
- Add an auxiliary unimodal prediction loss if easy, or discuss it as related work.
- Add a diagnostic figure showing when EOG is corrupted, quality score falls and fusion relies more on generated EOG/EEG.

### U-Sleep 2021

File: `paper/U-Sleep_2021.pdf`

Core idea:

- Fully convolutional U-Net style sleep staging.
- Trained on large multi-study PSG data.
- Accepts arbitrary standard EEG/EOG channel pairs.
- Uses channel sampling and majority voting across channel combinations.
- Uses noise replacement augmentation for segments or channels.

Useful for your experiment:

- Strongest cross-backbone candidate.
- Supports your claim that clinical systems need robustness to channel/protocol variation.
- Its augmentation is directly useful: replace variable signal lengths or entire channels with Gaussian noise.

What to borrow:

- Use U-Sleep as cross-backbone validation before EEGPT.
- Add corruption augmentation to training, not only zero dropout.
- Use robust per-channel/per-subject scaling and clipping if your preprocessing allows.

### FlexSleepTransformer 2024

File: `paper/FlexSleepTransformer_2024.pdf`

Core idea:

- Transformer sleep staging model for flexible channel counts.
- Uses sequence-to-sequence input.
- Proposes multi-channel random fusion instead of simple concatenation.
- Joint training across datasets with different channel configurations keeps about 98% of single-dataset accuracy and outperforms direct transfer.

Useful for your experiment:

- Supports "variable channel configuration" motivation.
- Random fusion suggests a simple way to regularize multi-channel dependence.

What to borrow:

- Add a random channel/sub-epoch corruption or fusion augmentation.
- Use cross-dataset/channel-setting language in related work.
- Do not prioritize implementing the full model unless U-Sleep is done.

### XSleepNet 2021

File: `paper/XSleepNet_2021.pdf`

Core idea:

- Multi-view sequence-to-sequence sleep staging.
- Learns from raw signal and time-frequency views.
- Uses adaptive gradient blending based on whether each view is generalizing or overfitting.

Useful for your experiment:

- Supports dynamic weighting: different streams/modalities should not contribute equally at all times.
- Notes that some views help N1/REM more while others help N3.

What to borrow:

- Use learned quality/gating curves as evidence that modality contribution should be dynamic.
- If v3 remains unstable, avoid complex gradient blending; cite XSleepNet to justify your simpler quality-guided gate.

### SalientSleepNet 2021

File: `paper/SalientSleepNet_2021.pdf`

Core idea:

- Detects salient waves from EEG and EOG using two streams.
- Uses multi-scale transition modeling and multimodal attention.
- Explicitly states EOG helps distinguish N1 and REM.

Useful for your experiment:

- Strong physiological support for why EOG matters.
- Good justification for stage-wise REM/N1 analysis.

What to borrow:

- Add REM/N1-focused confusion analysis.
- Discuss EOG degradation as especially harmful for REM/N1.
- Use class-wise F1 as a first-class result, not an appendix-only detail.

### EEGPT 2024

File: `paper/EEGPT_2024.pdf`

Core idea:

- Pretrained EEG transformer with dual self-supervised learning.
- Combines masked reconstruction with spatio-temporal representation alignment.
- Uses local spatio-temporal/channel embeddings and linear probing.

Useful for your experiment:

- Strong candidate for future EEG-only or EEG-representation backbone.
- Supports your teacher feature loss idea: representation alignment can be more useful than raw reconstruction.

What to borrow:

- Keep feature-level teacher loss; do not switch to raw EOG generation unless necessary.
- If using EEGPT, start with frozen encoder + small robust head rather than full fine-tuning.

### SleepFM 2024

File: `paper/SleepFM_2024.pdf`

Core idea:

- Multimodal sleep foundation model over brain activity, ECG, and respiratory signals.
- Uses leave-one-out contrastive learning.
- Learned embeddings outperform supervised CNNs on sleep stage and sleep-disordered breathing tasks under limited labels.

Useful for your experiment:

- Supports cross-modal representation alignment and leave-one-modality-out training.
- Useful background for "foundation model" and self-supervised sleep representation.

What to borrow:

- If adding a contrastive auxiliary loss, use a leave-one-out idea: generated EOG representation should align with real EOG representation conditioned on EEG.
- Do not implement a full SleepFM-style system now; it is too broad for current scope.

### SleepGPT 2026

File: `paper/SleepGPT_2026.pdf`

Core idea:

- Time-frequency sleep foundation model.
- Channel-adaptive mechanism for variable PSG channels.
- Unified time-frequency transformer.
- Uses masked reconstruction and generation to repair artifacts and augment small datasets.

Useful for your experiment:

- Supports artifact repair and masked PSG reconstruction as a modern direction.
- Supports using generation for data augmentation under small-data conditions.

What to borrow:

- Use masked/corrupted EOG reconstruction as a teacher-style training signal.
- Add a small experiment where corruption-augmented training improves robustness.
- Cite as high-level future work, not as a required baseline.

### Arbitrary Sensor Sleep Staging 2024/2025

File: `paper/ArbitrarySensorSleepStaging_2024.pdf`

Core idea:

- Score-based diffusion model for sleep staging from arbitrary sensor sets.
- Trains sensor-specific score networks and combines them with Bayesian factorization.
- Provides information gain per sensor as an interpretability metric.
- Tests missing segments, zero replacement, and added Gaussian noise.

Useful for your experiment:

- Strong support for sensor reliability and information contribution analysis.
- Their information-gain idea maps naturally to your quality score.

What to borrow:

- Add a proxy for "EOG contribution": compare logits/probabilities with and without EOG or with generated EOG.
- Plot quality score vs performance or prediction change.
- Use missing segment / zero / Gaussian noise tests as corruption scenarios.

### SMIL 2021

File: `paper/SMIL_2021.pdf`

Core idea:

- General multimodal learning with severely missing modality.
- Handles missing modality in training, testing, or both.
- Uses feature reconstruction and meta-regularization.

Useful for your experiment:

- Useful related work if you later train with real incomplete EOG data.
- Supports feature-space reconstruction instead of raw-signal reconstruction.

What to borrow:

- Mention as general missing-modality background.
- Do not prioritize implementation for this project.

### Missing Modality Survey 2024/2026

File: `paper/Missing_Modality_Survey_2024.pdf`

Core idea:

- Organizes missing-modality methods by data processing and strategy design.
- Main categories include modality imputation, representation-focused models, attention-based models, distillation-based models, graph methods, ensembles, dedicated models, and schedulers.

Useful for your experiment:

- Gives the related-work structure.
- Your method fits representation generation + adaptive fusion + distillation/alignment.

What to borrow:

- Use the taxonomy in the introduction/related work.
- Clearly state your setting: EOG missing/corrupted at inference and simulated during training, with clean EOG available for teacher loss during training.

### DeepSleepNet 2017

File: `paper/DeepSleepNet_2017.pdf`

Core idea:

- CNN feature extractor plus BiLSTM sequence residual learning.
- Two-step training.
- Single-channel EEG backbone.

Useful for your experiment:

- Current validation backbone.
- Important limitation: it was originally EEG-only, so your EEG+EOG adaptation and robust module must be clearly described.

What to borrow:

- Keep DeepSleepNet as main controlled backbone.
- Avoid modifying core backbone when testing robust modules, so gains are attributable to the EOG robustness design.

## Recommended Next Execution Order

1. Finish `v3-safe-lr fold_00`.
2. If it recovers, run fixed 5-fold; if not, freeze v2 as main method.
3. Implement EOG corruption evaluation modes.
4. Add calibrated quality thresholds from train split.
5. Rerun best method and EOG dropout under all corruption modes.
6. Produce stage-wise F1 and REM/N1 confusion analysis.
7. Add ablations for quality, generator, teacher loss, and anti-noise gate.
8. Add U-Sleep cross-backbone validation.
9. Only then consider EEGPT exploratory validation.

## Likely Paper Claim After These Changes

If results hold, the strongest claim should be:

> We address robust EEG+EOG sleep staging when EOG is missing or corrupted. Unlike generic missing-modality methods, the proposed module explicitly estimates EOG reliability, reconstructs EOG features from EEG, and dynamically fuses real and generated EOG features. Experiments across zero-missing and realistic EOG corruption scenarios show improved robustness over modality dropout while preserving normal EEG+EOG performance.

This is stronger and more defensible than claiming a general foundation model or arbitrary-modality method.
