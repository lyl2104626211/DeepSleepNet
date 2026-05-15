# Project Agent Notes

This file records project-specific collaboration rules for future Codex sessions.
Read this file before making changes in this repository.

## User Preference: Minimal Implementation

When the user asks for a "minimal implementation" in this project, interpret it as:

- Minimize code changes.
- Keep the implementation runnable and testable.
- Prefer adding isolated files/classes over rewriting stable code.
- Avoid unnecessary refactors, broad abstractions, or unrelated cleanup.

Do not interpret "minimal implementation" as reducing the research idea or module design.

For the robust EOG-missing sleep staging work, preserve the intended module design when requested:

- quality sensor for EOG reliability;
- EEG-to-EOG feature generator;
- dynamic gating/fusion between real and generated EOG features;
- classification loss plus feature reconstruction/teacher loss when applicable.

In short:

```text
Minimal = code-level minimal changes.
Not minimal = shrinking the method idea into a weaker baseline.
```

## Current Research Context

The current project direction is robust sleep staging under EOG missing/corrupted conditions.
DeepSleepNet is currently used as the validation backbone, but the robust module should not be treated as conceptually limited to DeepSleepNet.

## Current Goal And Progress

Read this progress note when resuming the project:

- `docs/research_progress_2026-05-15.md`
- `docs/research_progress_2026-05-14.md`

Current publication/graduation goal:

- minimum target: SCI Q3;
- stretch target: SCI Q2;
- graduation requires one SCI paper.

Current research state:

- the EOG-missing robustness problem has been validated on DeepSleepNet;
- `EOG dropout p=0.5` is the current strong baseline;
- `quality-guided generator v2` slightly outperforms EOG dropout on 5-fold pooled Acc/Macro-F1/Kappa, but the gain is small;
- `quality-guided generator v3` has been implemented with continuous quality scoring, gated anti-noise fusion, residual generator/fusion, and MSE+cosine feature teacher loss;
- `quality-guided generator v3-safe` initially underperformed on fold_00, so stage2 optimizer was updated to train robust module parameters with `stage2_sequence_learning_rate` while keeping existing encoders at `stage2_cnn_learning_rate`;
- next steps are rerunning v3-safe-lr on fold_00/5-fold, testing realistic EOG corruption scenarios, and cross-backbone validation on U-Sleep and EEGPT.
