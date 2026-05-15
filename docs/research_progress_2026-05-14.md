# 科研目标与进度记录

日期：2026-05-14

## 课题定位

当前课题主线：

```text
面向 EOG 缺失/损坏场景的鲁棒睡眠分期方法
```

核心问题：

- `EEG+EOG` 正常输入时，EOG 对睡眠分期，尤其 `REM`，有帮助。
- 普通双通道模型在测试时 `EOG` 缺失或损坏会明显退化。
- 目标是在 `EOG` 缺失/损坏时保持较高性能，同时不明显牺牲正常 `EEG+EOG` 场景性能。

当前主要 backbone：

- `DeepSleepNet`

计划用于跨模型验证的 baseline/backbone：

- `DeepSleepNet_2017`
- `U-Sleep_2021`
- `EEGPT_2024`

## 毕业与投稿目标

用户当前毕业条件：

```text
需要发表一篇 SCI。
最好 SCI 二区，SCI 三区也可以满足毕业条件。
```

现实判断：

- `SCI 三区`：当前路线有希望，是现实目标。
- `SCI 二区`：可以作为冲刺目标，但需要结果明显强于强 baseline，并且实验链完整。
- 当前阶段不能只靠方法想法，必须用实验结果支撑。

如果最后只做到：

```text
单一 DeepSleepNet
只测 EOG=0
鲁棒模块只比 EOG dropout 高 0.002 左右
normal 场景还下降
```

则 SCI 二三区都不稳。

如果最后做到：

```text
DeepSleepNet / U-Sleep / EEGPT 三个 backbone
EOG=0 / noise / flatline / drift 多异常场景
方法稳定优于 EOG dropout
normal 场景不明显下降
REM / N1 有明确改善
消融和统计分析完整
```

则 `SCI 三区` 比较有底气，`SCI 二区` 可以尝试。

## 当前进度判断

按完整毕业论文/投稿实验链估算：

```text
当前完成度约 55% - 65%
```

已经完成：

- 单通道 `EEG only v2` 同口径 baseline。
- 双通道 `EEG+EOG` 同口径 baseline。
- 测试时 `EOG=0` 缺失实验。
- 训练时 `EOG dropout p=0.5` 强 baseline。
- `gated fusion` baseline。
- `quality-guided generator` v1。
- `quality-guided generator` v2 独立实现。
- 结果汇总脚本和绘图脚本初步可用。

还需要完成：

- `quality-guided generator v2` 结果评估。
- 更贴近原始设计的 `v3` 鲁棒模块。
- `EOG` 异常类型扩展：`zero / flatline / noise / drift / artifact`。
- `U-Sleep` 和 `EEGPT` 的跨 backbone 对比。
- 消融实验：质量感知器、生成器、教师损失、动态融合。
- 统计显著性和最终论文图表。

## 已完成关键结果

### 20-fold 同口径主结果

| Setting | Acc | Macro-F1 | Kappa | Loss |
| --- | ---: | ---: | ---: | ---: |
| EEG only v2 | 0.813344 | 0.755996 | 0.740190 | 0.849549 |
| EEG+EOG | 0.826007 | 0.770517 | 0.756617 | 0.958570 |
| EEG+EOG, EOG=0 | 0.698359 | 0.576347 | 0.559210 | 1.787396 |

结论：

- `EEG+EOG` 相比 `EEG only` 有小幅提升。
- `EOG=0` 后性能明显低于 `EEG only`，说明普通双通道模型不能自动退回 EEG-only。
- 鲁棒性问题成立。

### 类别 F1

| Class | EEG only | EEG+EOG | EOG=0 |
| --- | ---: | ---: | ---: |
| W | 0.916158 | 0.915691 | 0.868880 |
| N1 | 0.474414 | 0.488356 | 0.271600 |
| N2 | 0.836173 | 0.844385 | 0.733175 |
| N3 | 0.780892 | 0.790639 | 0.777395 |
| REM | 0.807493 | 0.855115 | 0.302096 |

关键观察：

- `EOG` 正常时主要提升 `REM`。
- `EOG=0` 时 `REM` 崩塌最明显。
- 这直接支撑当前课题的必要性。

### 5-fold pilot 强 baseline 与鲁棒模块

固定 pilot folds：

```text
fold_00 fold_03 fold_07 fold_13 fold_17
```

| Setting | Acc | Macro-F1 | Kappa |
| --- | ---: | ---: | ---: |
| EEG+EOG baseline normal | 0.863862 | 0.815683 | 0.814167 |
| EEG+EOG baseline EOG=0 | 0.721717 | 0.623701 | 0.598963 |
| EOG dropout p=0.5 normal | 0.852605 | 0.807413 | 0.799706 |
| EOG dropout p=0.5 EOG=0 | 0.839766 | 0.788958 | 0.781630 |
| Gated fusion p=0.5 normal | 0.853964 | 0.801307 | 0.799818 |
| Gated fusion p=0.5 EOG=0 | 0.837900 | 0.783945 | 0.779338 |
| Quality generator p=0.5 normal | 0.852281 | 0.802510 | 0.799108 |
| Quality generator p=0.5 EOG=0 | 0.846257 | 0.791012 | 0.789937 |

当前判断：

- `EOG dropout p=0.5` 是当前强 baseline。
- `quality generator v1` 在 `EOG=0` 下略优于 dropout，但提升很小。
- `quality generator v1` normal 场景略低于 dropout。
- 现有结果不足以直接支撑高水平创新结论，需要继续优化和扩展异常场景。

## 当前方法设计要求

用户明确要求：

```text
“最小实现”指代码层面最小改动，不是方法设计缩水。
```

因此后续实现要遵守：

- 代码尽量少改。
- 尽量新增独立文件/类，不破坏已有稳定实验。
- 不要把用户原始模块设计简化成弱 baseline。
- 鲁棒模块设计应保留：
  - EOG 质量感知器；
  - EEG->EOG 特征生成器；
  - 真实 EOG / 生成 EOG 动态融合；
  - 分类损失 + 特征重建/教师损失。

## 当前风险点

最大风险：

```text
EOG dropout p=0.5 baseline 很强，当前鲁棒模块只小幅超过它。
```

因此不能只依赖 `EOG=0` 场景。后续必须扩展到更真实的 EOG 异常：

- flatline
- random noise
- high-amplitude artifact
- drift
- partial epoch dropout

这些场景更能体现质量感知器的价值。

## 下一步优先级

### 第一优先级：看 v2 结果

目标：

```text
验证 v1 表现不理想是否来自 EOG/伪 EOG 特征加得太硬。
```

v2 改动：

- `generator_loss_weight: 0.05 -> 0.01`
- `fused = H_eeg + alpha * H_eog_final`
- `alpha` 为可学习参数，限制在 `0~1`，初始值 `0.5`

判断标准：

- normal Macro-F1 是否回升；
- EOG=0 Macro-F1 是否不低于 v1/dropout；
- REM/N1 是否改善。

### 第二优先级：设计 v3

v3 应更贴近原始设计，而不是小修小补：

```text
质量感知 c
真实 EOG 特征 H_eog_real
生成 EOG 特征 H_eog_fake
EEG 特征 H_eeg
融合器综合 [H_eeg, H_eog_real, H_eog_fake, c]
分类损失 + 特征教师损失
```

代码要求：

- 尽量新建独立文件；
- 尽量复用现有 trainer 接入机制；
- 不修改原论文 backbone 主流程；
- 保持模块可插拔。

### 第三优先级：异常 EOG 场景

除 `EOG=0` 以外，必须补：

- `flatline`
- `noise`
- `drift`
- `artifact`

重点比较：

- 普通 `EEG+EOG`
- `EOG dropout`
- `quality-guided generator`
- 后续 `v3`

### 第四优先级：跨 backbone

目标不是重做所有细节，而是证明模块可迁移：

- DeepSleepNet：完整主实验和消融。
- U-Sleep：关键对比实验。
- EEGPT：关键对比实验或最小可运行 baseline。

### 第五优先级：论文材料

需要准备：

- 主结果表；
- 类别 F1 表，重点 `REM` 和 `N1`；
- EOG 异常鲁棒性图；
- 方法结构图；
- 消融表；
- 统计显著性；
- 暑假/寒假前可执行计划。

## 时间预期

如果不再大改方向：

```text
2-3 个月：有机会完成毕业论文级别实验闭环。
3-5 个月：更现实地完成一版投稿稿。
```

阶段安排建议：

- 5 月 - 6 月：定版 v2/v3，完成 DeepSleepNet 上完整实验。
- 7 月：补异常 EOG 场景和 U-Sleep。
- 8 月：补 EEGPT、消融、图表和初稿。
- 研二寒假前：力争完成可投稿版本。

## 给导师沟通口径

当前不应硬说“已经能发”，而应说：

```text
我已经完成 DeepSleepNet 上的问题验证，证明 EOG 正常时有帮助，
但普通双通道模型在 EOG 缺失时会严重退化。
在此基础上，我已经完成 EOG dropout 强 baseline 和质量感知跨模态恢复模块的初版。
下一步将围绕多异常 EOG 场景、跨 backbone 和消融实验，把该方向推进到 SCI 投稿级别。
```

