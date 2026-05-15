# 科研进度记录

日期：2026-05-13

## 当前阶段目标

围绕 `EOG` 缺失鲁棒性，完成 DeepSleepNet 上的同口径 baseline、缺失场景验证、EOG dropout 训练 baseline，并准备测试 gated fusion 鲁棒模块。

## 已完成结果

### 1. 同口径 20-fold 主表

三组结果已经完整对齐，均为当前预处理逻辑、同一套 participant-level 20-fold split。

| Setting | Acc | Macro-F1 | Kappa | Loss |
| --- | ---: | ---: | ---: | ---: |
| EEG only v2 | 0.813344 | 0.755996 | 0.740190 | 0.849549 |
| EEG+EOG | 0.826007 | 0.770517 | 0.756617 | 0.958570 |
| EEG+EOG, EOG=0 | 0.698359 | 0.576347 | 0.559210 | 1.787396 |

关键观察：

- 正常 `EEG+EOG` 相比 `EEG only` 有小幅收益。
- 测试时 `EOG=0` 会造成明显退化，且显著低于真正 `EEG only`。
- 普通双通道模型没有学会自动退回 EEG-only，说明鲁棒性问题成立。

### 2. 类别层面结果

按 20 折混淆矩阵聚合：

| Class | EEG only | EEG+EOG | EOG=0 |
| --- | ---: | ---: | ---: |
| W | 0.916158 | 0.915691 | 0.868880 |
| N1 | 0.474414 | 0.488356 | 0.271600 |
| N2 | 0.836173 | 0.844385 | 0.733175 |
| N3 | 0.780892 | 0.790639 | 0.777395 |
| REM | 0.807493 | 0.855115 | 0.302096 |

结论：

- `EOG` 正常时主要提升 `REM`。
- `EOG=0` 时 `REM` 崩塌最明显。
- 该现象直接支撑“EOG 缺失鲁棒性”研究问题。

### 3. EOG dropout training 5-fold pilot

固定 pilot folds：

- `fold_00`
- `fold_03`
- `fold_07`
- `fold_13`
- `fold_17`

结果目录：

- `results/deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog_dropout_p05`

| Setting | Acc | Macro-F1 | Kappa |
| --- | ---: | ---: | ---: |
| EEG only v2 | 0.843748 | 0.796527 | 0.787471 |
| EEG+EOG baseline normal | 0.864065 | 0.814336 | 0.813336 |
| EEG+EOG baseline EOG=0 | 0.722086 | 0.613360 | 0.596831 |
| EOG dropout p=0.5 normal | 0.852901 | 0.806994 | 0.798938 |
| EOG dropout p=0.5 EOG=0 | 0.840012 | 0.789457 | 0.780873 |

结论：

- `EOG dropout p=0.5` 只带来很小的 normal 性能代价。
- `EOG=0` 场景下性能大幅恢复。
- 这已经是一个强 baseline，后续鲁棒模块必须与它对比。

## 新增代码结构

### 鲁棒方案目录

新增：

- `src/eeg_sleep/robust_schemes/`

包含：

- `scheme_a_eog_dropout.py`
- `scheme_b_gated_fusion.py`
- `scheme_c_mixture_fusion.py`
- `README.md`

### scheme A

训练时随机遮 EOG，是当前强 baseline。

### scheme B

EEG-main gated EOG fusion：

```text
gate = sigmoid(MLP([f_eeg, f_eog]))
fused = f_eeg + gate * f_eog
```

该方案优先测试。

### scheme C

softmax mixture fusion：

```text
[w_eeg, w_eog] = softmax(MLP([f_eeg, f_eog]))
fused = w_eeg * f_eeg + w_eog * f_eog
```

作为备选模块。

## 当前接入状态

已新增 gated fusion 测试配置：

- `configs/paper_sleep_edf_fpz_cz_eeg_eog_gated_fusion_p05.yaml`

已新增 gated fusion 5 折脚本：

- `scripts/run_gated_fusion_p05_five_folds.sh`

已修改：

- `src/eeg_sleep/models.py`
- `src/eeg_sleep/trainer.py`

现在可以通过 `model.name = deepsleepnet_gated_fusion` 构建 gated fusion 模型。

## 下一步

### 第一优先级

在服务器跑 gated fusion 5-fold pilot：

```bash
python main.py plan --config configs/paper_sleep_edf_fpz_cz_eeg_eog_gated_fusion_p05.yaml
nohup env PYTHONUNBUFFERED=1 bash scripts/run_gated_fusion_p05_five_folds.sh > gated_fusion_p05_five_folds.log 2>&1 &
```

### 第二优先级

汇总 5 折结果，对比：

- `EEG+EOG baseline normal`
- `EEG+EOG baseline EOG=0`
- `EOG dropout p=0.5 normal`
- `EOG dropout p=0.5 EOG=0`
- `Gated fusion p=0.5 normal`
- `Gated fusion p=0.5 EOG=0`

重点指标：

- Macro-F1
- REM F1
- N1 F1
- EOG=0 相对 normal 的性能下降

### 第三优先级

如果 gated fusion 优于普通 EOG dropout：

- 增加 gate 统计与可解释性分析
- 测试 EOG noise / flatline / drift
- 再考虑扩展到更多 folds 或第二 backbone

如果 gated fusion 不优于普通 EOG dropout：

- 保留 EOG dropout 作为强 baseline
- 暂缓复杂模块
- 优先测试 EOG 异常类型，确认问题边界
