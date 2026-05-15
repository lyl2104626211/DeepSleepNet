# DeepSleepNet 在 Sleep-EDF Fpz-Cz 上的 20 折结果汇总

## 实验设置

- 数据集：`Sleep-EDF`
- 通道：`Fpz-Cz`
- 任务：`5` 类睡眠分期
- 模型：`DeepSleepNet`
- 评估方式：按被试划分的 `20-fold cross-validation`
- 结果目录：`results/deepsleepnet_sleep_edf_paper_fpz_cz`

## 20 折平均结果

基于 `fold_00` 到 `fold_19` 的测试集评估结果，得到：

| Metric | Mean | Std |
| --- | ---: | ---: |
| Accuracy | `81.88%` | `3.52%` |
| Macro-F1 | `76.23%` | `3.57%` |
| Cohen's Kappa | `0.7474` | `0.0481` |
| Loss | `0.8390` | `0.2522` |

## 对比图

结果对比图见：

- [deepsleepnet_cv20_comparison.png](C:/Users/21046/Desktop/EEG_SLEEP/docs/figures/deepsleepnet_cv20_comparison.png)

## 与论文结果对比

论文中 DeepSleepNet 在 `Sleep-EDF / Fpz-Cz / 20-fold CV` 上的代表性结果大致为：

| Setting | Accuracy | Macro-F1 | Cohen's Kappa |
| --- | ---: | ---: | ---: |
| Original paper | `82.0%` | `76.9%` | `0.76` |
| This reproduction | `81.88%` | `76.23%` | `0.7474` |
| Gap | `-0.12%` | `-0.67%` | `-0.0126` |

## 结果解读

这组结果与论文已经非常接近，可以认为主实验复现基本成功。

- `Accuracy` 仅低于论文约 `0.12` 个百分点，几乎重合。
- `Macro-F1` 低约 `0.67` 个百分点，说明整体类别平衡表现与论文接近。
- `Kappa` 低约 `0.013`，仍处于较小偏差范围内。

造成轻微差距的可能原因包括：

- 预处理细节与论文实现并非完全一致；
- 训练轮数、随机种子和优化超参数存在微小偏移；
- 不同折中的被试组成略有波动；
- 工程实现与原作者代码在采样、序列构造或评估细节上存在差异。

## 最好与最差折

| Fold | Accuracy | Macro-F1 | Cohen's Kappa |
| --- | ---: | ---: | ---: |
| Best: `fold_17` | `0.9000` | `0.8488` | `0.8623` |
| Worst: `fold_15` | `0.7604` | `0.7277` | `0.6837` |

不同折之间存在一定波动，但整体均值稳定，没有出现大面积失效折，说明模型在当前划分下具有较好的可重复性。

## 论文中可直接使用的描述

可在正文中写为：

> 在 Sleep-EDF Fpz-Cz 单通道设置下，我们基于按被试划分的 20 折交叉验证对 DeepSleepNet 进行了复现。最终在测试集上取得了 `81.88%` 的 Accuracy、`76.23%` 的 Macro-F1 和 `0.7474` 的 Cohen's Kappa。与原论文报告的 `82.0% / 76.9% / 0.76` 相比，结果高度接近，说明本文实现能够较好复现 DeepSleepNet 在该数据集上的核心性能。

## 对应汇总表

逐折结果见：

- `results/deepsleepnet_sleep_edf_paper_fpz_cz/cv20_summary.csv`
