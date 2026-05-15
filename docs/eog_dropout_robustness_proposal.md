# 面向 EOG 脱落的鲁棒睡眠分期实验方案

## 1. 研究问题

### 1.1 背景

当前常见睡眠分期模型通常使用单通道 EEG，或者在多模态场景下使用 `EEG + EOG`。  
但在真实采集中，`EOG` 电极存在脱落、接触不良、漂移、噪声污染等问题。  
如果模型过度依赖 `EOG`，一旦眼电异常，尤其是 `REM`、`N1` 相关判别能力可能明显下降。

### 1.2 核心问题

本研究希望回答下面这个问题：

**在 `EOG` 缺失或损坏时，能否通过一个可插拔的鲁棒融合模块，让睡眠分期模型仍然保持较好的性能？**

### 1.3 论文式问题定义

可以把问题正式写成：

- 任务：多模态睡眠分期
- 输入：`EEG` 与 `EOG`
- 干扰：测试时 `EOG` 可能正常、缺失或损坏
- 目标：在不明显牺牲正常条件性能的前提下，提高模型面对 `EOG` 异常时的鲁棒性

### 1.4 研究假设

- `EEG + EOG` 正常情况下通常优于 `EEG only`
- 普通多模态融合模型在 `EOG` 失效时性能会明显下降
- 如果加入一个显式建模模态可靠性的融合模块，模型可以在 `EOG` 异常时自动降低对 `EOG` 的依赖，从而提升鲁棒性

## 2. 模型设计

### 2.1 总体思路

先不重新发明整套 backbone，而是在现有睡眠分期模型上增加一个**可插拔鲁棒融合模块**。

建议先选两个 backbone：

- `DeepSleepNet`
- `U-Sleep` 或其他 U-Net 风格睡眠分期模型

模块目标不是“让多模态更强”，而是“让多模态在缺失模态条件下更稳”。

### 2.2 最小模型结构

建议第一版结构：

1. `EEG` 分支提取特征
2. `EOG` 分支提取特征
3. 使用一个轻量的 `reliability gate` 估计 `EOG` 当前是否可信
4. 用 gate 对两路特征做动态融合
5. 输出送入时序模块或分类头

可以写成：

```text
EEG -> EEG encoder ----\
                        -> robust fusion -> temporal/context module -> classifier
EOG -> EOG encoder ----/
             \
              -> reliability gate
```

### 2.3 可插拔模块的最小实现

第一版不要做太重，建议从最简单、最容易解释的方案开始：

- `EEG feature = f_eeg(x_eeg)`
- `EOG feature = f_eog(x_eog)`
- `gate = sigmoid(MLP([f_eeg, f_eog]))`
- `fused = f_eeg + gate * f_eog`

解释：

- 当 `EOG` 可靠时，`gate` 倾向较大
- 当 `EOG` 异常时，`gate` 倾向减小
- 模型自动退回更依赖 `EEG`

### 2.4 后续可扩展方向

如果第一版有效，再考虑加复杂设计：

- cross-attention 融合
- 显式缺失标记 `missing indicator`
- 基于重建误差或统计特征的模态质量估计
- mixture-of-experts 式模态选择

## 3. Baseline 设计

### 3.1 必做 baseline

建议至少做下面几组：

1. `EEG only`
2. `EEG + EOG` 普通融合
3. `EEG + EOG`，测试时 `EOG = 0`
4. `EEG + EOG`，训练时随机 `EOG dropout`
5. `EEG + EOG + 你的鲁棒模块`

### 3.2 建议补充 baseline

为了让论文更扎实，建议再补 2 到 3 组：

6. `EEG + EOG`，测试时加入高噪声 `EOG`
7. `EEG + EOG`，测试时加入漂移/平线 `EOG`
8. `EEG + EOG + 简单 gate` 与 `EEG + EOG + 你的完整模块` 对比

### 3.3 对比原则

所有 baseline 都应尽量保持：

- 相同数据划分
- 相同训练轮数
- 相同优化器与学习率
- 相同 backbone
- 仅改变输入模态与融合策略

这样才能说明性能差异来自“鲁棒模块”，而不是训练预算不同。

## 4. EOG 脱落/损坏模拟策略

不要只做 `EOG = 0` 一种，建议做至少四种异常：

1. `zero-out`
   - 整段 `EOG` 直接置零
2. `flatline`
   - 整段变成常数，模拟电极脱落后无波动
3. `gaussian noise`
   - 叠加大幅噪声，模拟接触不良
4. `drift / saturation`
   - 低频漂移或信号饱和，模拟设备异常

建议设置多个强度级别：

- 轻度异常
- 中度异常
- 重度异常

## 5. 实验设计

### 5.1 数据集

第一阶段建议：

- 先在 `Sleep-EDF` 上完成全部实验

第二阶段建议：

- 如果时间允许，再补第二个数据集做泛化验证

### 5.2 评估指标

建议至少报告：

- Accuracy
- Macro-F1
- Cohen's Kappa

如果要更扎实，可以补：

- per-class F1
- confusion matrix
- 特别关注 `N1`、`REM`

### 5.3 评估场景

必须分成两类：

1. 正常场景
   - `EOG` 正常
2. 异常场景
   - `EOG` 缺失或损坏

论文重点不是“绝对最高分”，而是：

- 正常条件不明显变差
- 异常条件下降得更少

## 6. 实验表格怎么做

### 6.1 主结果表

建议主表格式：

| Model | Input | EOG Condition | Acc | Macro-F1 | Kappa |
|---|---|---|---:|---:|---:|
| DeepSleepNet | EEG only | normal |  |  |  |
| DeepSleepNet | EEG+EOG | normal |  |  |  |
| DeepSleepNet | EEG+EOG | zero-out |  |  |  |
| DeepSleepNet | EEG+EOG + EOG dropout training | zero-out |  |  |  |
| DeepSleepNet | EEG+EOG + robust module | zero-out |  |  |  |
| U-Sleep | EEG only | normal |  |  |  |
| U-Sleep | EEG+EOG | normal |  |  |  |
| U-Sleep | EEG+EOG | zero-out |  |  |  |
| U-Sleep | EEG+EOG + robust module | zero-out |  |  |  |

### 6.2 异常类型对比表

| Model | zero-out | flatline | noise | drift |
|---|---:|---:|---:|---:|
| EEG+EOG baseline |  |  |  |  |
| EEG+EOG + dropout training |  |  |  |  |
| EEG+EOG + robust module |  |  |  |  |

表中建议填：

- Macro-F1
- 或相对于正常条件的性能下降值 `delta`

### 6.3 鲁棒性下降表

这个表特别适合论文：

| Model | Normal Macro-F1 | Corrupted Macro-F1 | Drop |
|---|---:|---:|---:|
| EEG+EOG baseline |  |  |  |
| EEG+EOG + robust module |  |  |  |

如果你的模块设计有效，最想看到的是：

- 正常性能差不多
- `Drop` 更小

### 6.4 分类别结果表

| Model | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|
| baseline normal |  |  |  |  |  |
| baseline corrupted EOG |  |  |  |  |  |
| robust module corrupted EOG |  |  |  |  |  |

重点关注：

- `N1`
- `REM`

### 6.5 消融实验表

| Variant | Reliability Gate | Corruption Training | Missing Indicator | Macro-F1 |
|---|---|---|---|---:|
| baseline | x | x | x |  |
| + dropout training | x | check | x |  |
| + gate | check | x | x |  |
| + gate + corruption training | check | check | x |  |
| full model | check | check | check |  |

## 7. 最小实现路径

### 7.1 第一阶段：先验证问题存在

目标：

- 不先做新模块
- 先证明 `EOG` 脱落会明显伤害普通多模态模型

最小任务：

1. 数据改成支持 `EEG + EOG`
2. 先做 `EEG only`
3. 再做 `EEG + EOG`
4. 再测 `EOG = 0`

如果第 4 步相对第 2/3 步明显变差，说明题目成立。

### 7.2 第二阶段：做最小鲁棒版

目标：

- 先做一个最轻量的 gate 模块

最小任务：

1. 在现有 backbone 里加入 `EEG encoder` 与 `EOG encoder`
2. 做简单 gate 融合
3. 训练时随机让一部分 batch 的 `EOG` 失效
4. 测试正常条件和异常条件

### 7.3 第三阶段：可插拔验证

目标：

- 证明你的模块不是只对一个 backbone 有效

最小任务：

1. 在 `DeepSleepNet` 上跑通
2. 在 `U-Sleep` 或另一个结构上复用
3. 只替换融合部分，不改其它训练协议

### 7.4 第四阶段：写作收口

目标：

- 把题目从“工程小改动”写成“缺失模态鲁棒性研究”

写作重点：

- 问题定义要清楚
- 异常模拟要合理
- 对照实验要充分
- 结果要强调“性能下降更少”

## 8. 最小代码落地建议

结合当前仓库，建议实现顺序：

1. 在数据层支持读取 `EEG + EOG`
   - 先保证 `manifest` 能记录双通道路径或双通道数组
2. 先不改训练框架
   - 先在当前 `train-stage1 / train-stage2` 流程上延伸
3. 先做 `DeepSleepNet` 版本
4. 再考虑第二个 backbone

建议代码上尽量保持：

- 最小实现
- 少封装
- 少引入与当前代码风格不一致的抽象

## 9. 预期结论

如果实验成功，论文希望得出的结论是：

1. `EOG` 对睡眠分期，尤其是 `REM / N1` 判别有帮助
2. 普通多模态模型在 `EOG` 缺失或损坏时性能明显下降
3. 引入鲁棒融合模块后，模型能在异常条件下保持更稳定性能
4. 该模块可插拔，可在不同 backbone 上复用

## 10. 当前建议

现在不要一上来就做完整论文版模块，建议先按下面顺序推进：

1. 先做 `EEG only` vs `EEG+EOG`
2. 再做 `EOG = 0 / noise / drift`
3. 确认问题真实存在
4. 再做最小 gate 模块
5. 最后再追求“可插拔”和“跨 backbone”

如果第一阶段就看不到明显性能下降，这个题目就要及时收缩或换方向。  
如果第一阶段现象明显，这个方向就值得继续投入。
