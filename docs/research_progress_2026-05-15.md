# 科研进度记录

日期：2026-05-15

## 今日核心进展

今天完成了 `quality-guided generator v2` 的 5-fold 结果汇总，并基于结果设计实现了 `quality-guided generator v3`。

当前主线仍然是：

```text
面向 EOG 缺失/损坏场景的鲁棒睡眠分期
```

当前强 baseline 仍然是：

```text
EOG dropout p=0.5
```

## v2 五折结果

固定 pilot folds：

```text
fold_00 fold_03 fold_07 fold_13 fold_17
```

按 pooled confusion matrix 汇总：

| Setting | Acc | Macro-F1 | Kappa |
| --- | ---: | ---: | ---: |
| EEG+EOG baseline normal | 0.863862 | 0.815683 | 0.814167 |
| EEG+EOG baseline EOG=0 | 0.721717 | 0.623701 | 0.598963 |
| EOG dropout p=0.5 normal | 0.852605 | 0.807413 | 0.799706 |
| EOG dropout p=0.5 EOG=0 | 0.839766 | 0.788958 | 0.781630 |
| Quality generator v1 normal | 0.852281 | 0.802510 | 0.799108 |
| Quality generator v1 EOG=0 | 0.846257 | 0.791012 | 0.789937 |
| Quality generator v2 normal | 0.859055 | 0.810009 | 0.807551 |
| Quality generator v2 EOG=0 | 0.847900 | 0.790224 | 0.791647 |

关键对比：

```text
v2 normal vs EOG dropout:
Macro-F1 +0.002596
Acc +0.006450
Kappa +0.007845

v2 EOG=0 vs EOG dropout:
Macro-F1 +0.001267
Acc +0.008133
Kappa +0.010017
```

结论：

- v2 相比 v1 明显改善 normal 场景。
- v2 在 normal 和 EOG=0 两个场景下都小幅超过 `EOG dropout p=0.5`。
- 但提升幅度仍然偏小，不能说已经显著击败强 baseline。
- v2 的 `alpha` 可学习注入强度是有效方向，但还不够作为最终方法。

类别层面：

| Setting | W | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| EOG dropout EOG=0 | 0.918756 | 0.488305 | 0.868952 | 0.831985 | 0.836790 |
| v1 EOG=0 | 0.917995 | 0.466366 | 0.880278 | 0.857043 | 0.833378 |
| v2 EOG=0 | 0.917898 | 0.456136 | 0.883568 | 0.858393 | 0.835126 |

问题：

- v2 主要提升 `N2/N3`。
- `N1` 仍低于 dropout。
- `REM` 接近 dropout，但没有明显超过。
- 后续 v3 需要重点解决 `N1/REM` 和复杂 EOG 异常场景。

## 今日新增 v3 结构

新增文件：

```text
src/eeg_sleep/robust_schemes/scheme_d_v3_quality_guided_generator.py
configs/paper_sleep_edf_fpz_cz_eeg_eog_quality_generator_v3_p05.yaml
scripts/run_quality_generator_v3_p05_five_folds.sh
```

同时在以下文件中注册新模型名：

```text
src/eeg_sleep/trainer.py
src/eeg_sleep/models.py
```

新模型名：

```text
deepsleepnet_quality_guided_generator_v3
```

结果目录：

```text
results/deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog_quality_generator_v3_p05
```

## v3 设计

v3 的目标是保留用户原始设计，同时继续保持代码层面最小改动：

```text
质量感知器
+ EEG->EOG 特征生成器
+ 真实/生成 EOG 动态融合
+ 分类损失 + 特征教师损失
```

### 1. 连续质量感知器

类：

```text
ContinuousRuleBasedQualitySensor
```

输入：

```text
observed_eog: [B, 1, L]
```

输出：

```text
c ∈ [0, 1]
```

设计：

- 低方差检测：识别 `EOG=0` / flatline。
- 高方差检测：识别白噪声 / 高幅伪迹。
- 当前默认：

```text
min_variance = 1e-8
max_variance = 1e4
```

说明：

- `max_variance` 需要后续根据真实 EOG 方差分布继续标定。
- 本地没有完整双通道 manifest，暂时无法精确统计阈值。

### 2. 残差式 EEG->EOG 生成器

类：

```text
ResidualEOGGenerator
```

公式：

```text
base = Linear(H_eeg)
delta = MLP(LayerNorm(H_eeg))
H_fake = base + beta * delta
```

目的：

- `Linear` 做基础 EEG->EOG 特征映射。
- `LayerNorm + MLP` 学习补偿项。
- `beta` 控制补偿强度。

### 3. 防毒的残差融合器

类：

```text
ResidualQualityFusion
```

最初 v3 曾设计为：

```text
concat(H_eeg, H_real, H_fake, H_final, c)
```

但发现问题：

```text
如果 EOG 是高方差白噪声，H_real 会作为毒药特征直接穿透 FusionMLP。
```

因此今日已修正为：

```text
H_final = c * H_real + (1-c) * H_fake
trusted_real = c * H_real
backup_fake = (1-c) * H_fake
Z = concat(H_eeg, trusted_real, backup_fake, H_final, c)
H_out = H_eeg + gamma * FusionMLP(LayerNorm(Z))
```

这样：

- 当 `c≈0` 时，`trusted_real≈0`，噪声 EOG 特征不会直接穿透。
- 当 `c≈1` 时，模型主要使用真实 EOG 特征。
- EEG 主路径通过残差保留。

### 4. 新辅助损失

今日将 v3 的辅助损失从单纯 MSE 改为：

```text
Loss = CE
     + 0.1 * MSE(H_fake, H_clean.detach())
     + 0.1 * (1 - CosineSim(H_fake, H_clean.detach()))
```

原因：

- MSE 约束数值尺度。
- Cosine loss 约束特征方向。
- 特征蒸馏场景下，方向信息通常比绝对数值更重要。

当前代码默认：

```text
generator_mse_loss_weight = 0.1
generator_cosine_loss_weight = 0.1
```

## 当前 v3 训练思路

训练时：

```text
1. 输入 clean EEG + clean EOG。
2. 先用 clean EOG 编码得到 H_clean，作为 teacher。
3. 模块内部以 p=0.5 随机遮挡 EOG，得到 observed_eog。
4. observed_eog 进入质量感知器得到 c。
5. EEG 生成 H_fake。
6. 根据 c 融合真实/生成 EOG 特征。
7. 用 CE + MSE + Cosine 训练。
```

推理/评估时：

```text
不再随机遮挡。
如果测试时手动 mask EOG，则质量感知器应给出较低 c，模型更多使用 H_fake。
如果 EOG 正常，则 c 较高，模型更多使用真实 EOG。
```

## 服务器同步文件

需要同步：

```text
src/eeg_sleep/models.py
src/eeg_sleep/trainer.py
src/eeg_sleep/robust_schemes/scheme_d_v3_quality_guided_generator.py
configs/paper_sleep_edf_fpz_cz_eeg_eog_quality_generator_v3_p05.yaml
scripts/run_quality_generator_v3_p05_five_folds.sh
```

如果服务器缺 v2 文件，也同步：

```text
src/eeg_sleep/robust_schemes/scheme_d_v2_quality_guided_generator.py
```

## 服务器执行指令

进入项目：

```bash
cd ~/autodl-tmp/EggSleepNet/deepsleepnet
```

检查配置：

```bash
python main.py plan --config configs/paper_sleep_edf_fpz_cz_eeg_eog_quality_generator_v3_p05.yaml
```

先跑一折试水：

```bash
FOLDS="fold_00" nohup env PYTHONUNBUFFERED=1 bash scripts/run_quality_generator_v3_p05_five_folds.sh > quality_generator_v3_p05_fold00.log 2>&1 &
tail -f quality_generator_v3_p05_fold00.log
```

跑五折：

```bash
nohup env PYTHONUNBUFFERED=1 bash scripts/run_quality_generator_v3_p05_five_folds.sh > quality_generator_v3_p05_five_folds.log 2>&1 &
tail -f quality_generator_v3_p05_five_folds.log
```

查看进程：

```bash
pgrep -af "quality_generator_v3|train-stage|evaluate-stage"
```

## 下一步

优先级：

1. 先跑 v3 `fold_00`，确认没有训练/加载错误。
2. 如果 `fold_00` 指标没有明显异常，再跑完整五折。
3. v3 五折回来后对比：
   - `EOG dropout p=0.5`
   - `quality generator v1`
   - `quality generator v2`
   - `quality generator v3`
4. 如果 v3 相比 v2 没提升，下一步不要继续堆结构，应转向：
   - EOG noise / flatline / drift / artifact 场景；
   - 质量感知器阈值标定；
   - U-Sleep / EEGPT 跨 backbone。

## v3-safe 诊断与学习率修正

本轮对话中，`quality-guided generator v3-safe` 的第一折结果仍不理想：

```text
v3-safe fold_00 normal Macro-F1: 0.783422
v3-safe fold_00 EOG=0 Macro-F1: 0.751348
```

对比当前更强的 `quality generator v2` 和 `EOG dropout p=0.5`，v3-safe 暂时没有达到预期。结合训练日志观察：

```text
stage1 epoch 100 train_loss: 0.686528
stage2 epoch 100 train_loss: 0.428583
```

判断不是单纯“训练轮次太少”，更可能是 stage2 中新加入的鲁棒模块学习率过低。原始 stage2 optimizer 把整个 `feature_extractor` 都放进 `stage2_cnn_learning_rate=1e-6`，这会导致 `generator`、`fusion`、`quality_sensor` 等新模块在 stage2 里几乎学不动。

本轮代码修正：

```text
src/eeg_sleep/trainer.py
```

修改 `_build_stage2_optimizer()`，将 stage2 参数分为三组：

| 参数组 | 学习率 | 目的 |
| --- | ---: | --- |
| 原有 CNN encoder | `stage2_cnn_learning_rate = 1e-6` | 保持论文式微调，避免破坏已学到的特征 |
| 鲁棒模块 generator/fusion/quality_sensor/residual_logit | `stage2_sequence_learning_rate = 1e-4` | 让新增模块在 stage2 充分学习 |
| BiLSTM/shortcut/classifier | `stage2_sequence_learning_rate = 1e-4` | 保持原 stage2 序列层训练强度 |

核心思路：

```text
不是整体调大学习率，而是只给新插入的鲁棒模块更高学习率。
这样既不破坏 DeepSleepNet 原论文主流程，也避免新模块被 1e-6 级学习率冻结。
```

当前 v3-safe 结构保留此前安全修正：

- `fake_eog_features = generator(eeg_features.detach())`，防止生成损失反向拉扯 EEG encoder。
- fusion residual gate 使用接近 0 的初始化，避免随机残差分支一开始污染主干特征。
- fusion 输入端不再做跨模态 `LayerNorm`，避免把 EEG/EOG/quality score 的物理刻度混在一起归一化。
- 使用 `trusted_real = c * H_real` 和 `backup_fake = (1-c) * H_fake`，避免坏 EOG 特征直接穿透融合层。
- 辅助损失保持：

```text
Loss = CE
     + 0.1 * MSE(H_fake, H_clean.detach())
     + 0.1 * (1 - CosineSim(H_fake, H_clean.detach()))
```

下一步服务器建议先重新跑一折，不直接盲跑五折：

```bash
cd ~/autodl-tmp/EggSleepNet/deepsleepnet
FOLDS="fold_00" SKIP_DONE=0 RESULT_ROOT=results/deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog_quality_generator_v3_safe_lr_p05 nohup env PYTHONUNBUFFERED=1 bash scripts/run_quality_generator_v3_p05_five_folds.sh > quality_generator_v3_safe_lr_fold00.log 2>&1 &
tail -f quality_generator_v3_safe_lr_fold00.log
```

如果 fold_00 相比原 v3-safe 明显回升，再跑固定五折：

```bash
SKIP_DONE=0 RESULT_ROOT=results/deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog_quality_generator_v3_safe_lr_p05 nohup env PYTHONUNBUFFERED=1 bash scripts/run_quality_generator_v3_p05_five_folds.sh > quality_generator_v3_safe_lr_five_folds.log 2>&1 &
tail -f quality_generator_v3_safe_lr_five_folds.log
```

需要同步到服务器的文件：

```text
src/eeg_sleep/trainer.py
src/eeg_sleep/robust_schemes/scheme_d_v3_quality_guided_generator.py
configs/paper_sleep_edf_fpz_cz_eeg_eog_quality_generator_v3_p05.yaml
scripts/run_quality_generator_v3_p05_five_folds.sh
```

当前判断：

- v2 仍是目前最稳的改进版本，虽然提升小。
- v3-safe 的结构设计合理，但需要确认新模块学习率修正后是否能追上或超过 v2。
- 如果 v3-safe-lr 仍不理想，应停止继续堆结构，转向 v2/v2.5 简洁路线，并补充更真实的 EOG 异常场景，如 noise、flatline、drift、artifact。
