# 科研进度记录

日期：2026-05-07

## 本阶段目标

整理 `DeepSleepNet` 在 `Sleep-EDF / Fpz-Cz` 设置下的 `EEG+EOG` 双通道 baseline 20 折结果，并重新判断它与旧单通道 `EEG only` 结果之间哪些折可以公平比较。

## 当前进展

### 1. 双通道 20-fold baseline 已完成

双通道结果目录：

- `results/deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog`

已完成折：

- `fold_00` 到 `fold_19`

每折均已生成：

- `stage1/training_summary.json`
- `stage1/best_model.pt`
- `stage2/training_summary.json`
- `stage2/best_model.pt`
- `eval_test/evaluation_test.json`
- `eval_test/confusion_matrix_test.json`

### 2. 双通道 20 折平均结果

基于全部 20 个 `eval_test/evaluation_test.json`：

| Metric | EEG+EOG 20-fold mean |
| --- | ---: |
| Accuracy | 0.826007 |
| Macro-F1 | 0.770517 |
| Cohen-Kappa | 0.756617 |
| Loss | 0.958570 |

说明：

- 这是双通道自身的完整 20 折结果。
- 由于旧单通道 processed 数据存在历史版本差异，这个 20 折均值暂时不应直接与旧单通道 20 折均值作为最终公平结论。

### 3. 与旧单通道结果的公平性检查

旧单通道结果目录：

- `results/deepsleepnet_sleep_edf_paper_fpz_cz`

公平性判断依据：

- `subject_ids` 一致
- `covered_epochs` 一致
- `num_sequences` 一致

当前可公平比较的 14 折：

- `fold_02`
- `fold_03`
- `fold_04`
- `fold_05`
- `fold_06`
- `fold_07`
- `fold_09`
- `fold_11`
- `fold_12`
- `fold_13`
- `fold_16`
- `fold_17`
- `fold_18`
- `fold_19`

当前不可公平比较的 6 折：

- `fold_00`
- `fold_01`
- `fold_08`
- `fold_10`
- `fold_14`
- `fold_15`

这些不公平折的共同点是：两边 `subject_ids` 一致，但 `covered_epochs` 或 `num_sequences` 不一致，说明问题主要来自旧单通道 processed 数据与当前双通道 processed 数据的口径不同。

## 当前阶段性结果

仅基于可公平比较的 14 折：

| Metric | Single EEG | EEG+EOG | Delta |
| --- | ---: | ---: | ---: |
| Accuracy | 0.823544 | 0.834287 | +0.010743 |
| Macro-F1 | 0.763559 | 0.773891 | +0.010332 |
| Cohen-Kappa | 0.752462 | 0.766399 | +0.013937 |
| Loss | 0.778925 | 0.793103 | +0.014178 |

逐折观察：

- 双通道在多数可比折上提升，但不是每一折都提升。
- `fold_03`、`fold_06`、`fold_12`、`fold_18` 等折上双通道部分指标下降。
- `fold_11`、`fold_13` 的双通道提升较明显。
- 双通道 `fold_15` 的 loss 明显偏高，但该折当前不适合作为旧单通道公平对比结论。

## 当前解释

当前结果支持下面这个相对稳妥的叙述：

- `EOG` 对睡眠分期有增量信息。
- 简单 `EEG+EOG` 通道拼接能带来约 `1%` 左右的平均提升。
- 这个提升并不大，说明 naive fusion 不是最终答案。
- 后续研究 `EOG` 脱落、污染条件下的鲁棒融合仍然有必要。

## 下一步建议

### 1. 优先修复单通道公平对照

优先建议重新生成单通道 `v2`：

- 使用当前代码
- 使用当前原始数据
- 使用与双通道一致的预处理逻辑
- 重新跑 20 折 `EEG only`

这样后续主结果可以写成同口径 20 折对比，不需要把结论限制在 14 折。

如果算力不足，可以只补跑当前不公平的 6 折：

- `fold_00`
- `fold_01`
- `fold_08`
- `fold_10`
- `fold_14`
- `fold_15`

### 2. 整理正式结果表

建议至少整理：

- `EEG only` vs `EEG+EOG` 的 Acc / Macro-F1 / Kappa / Loss
- 每折结果表
- 均值和标准差
- per-class F1
- confusion matrix

per-class F1 需要重点关注：

- `N1`
- `REM`

因为这两个类别更可能从 `EOG` 中获益，也更适合支撑后续鲁棒性研究。

### 3. 进入 EOG 异常评估

在公平 baseline 明确后，先不急着上新模块，先验证问题是否成立：

- 测试时 `EOG = 0`
- 测试时 `EOG` flatline
- 测试时 `EOG` 加强噪声
- 测试时 `EOG` drift / saturation

如果普通双通道 baseline 在这些条件下明显下降，再进入：

- 训练期 `EOG dropout`
- reliability gate
- 可插拔鲁棒融合模块

## 当前结论

双通道 baseline 已经完成，结果没有否定课题，反而给出了一个比较健康的研究起点：

- 正常条件下，`EEG+EOG` 比 `EEG only` 略好。
- 提升幅度有限，说明简单融合还不够。
- 接下来应把重点放在同口径对照和 `EOG` 异常条件下的性能下降验证上。
