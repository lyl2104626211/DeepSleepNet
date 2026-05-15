# 当前计划

## 当前主线

当前课题定位：

> 面向 `EOG` 缺失/损坏的鲁棒睡眠分期。

当前 backbone：

- `DeepSleepNet`

当前阶段：

- 同口径 20-fold baseline 已完成。
- `EOG=0` 缺失场景已验证，普通双通道模型明显退化。
- `EOG dropout p=0.5` 5-fold pilot 已验证有效。
- 下一步测试 `scheme_b_gated_fusion`。

## 已完成结果

### 1. 同口径 20-fold 主表

| Setting | Acc | Macro-F1 | Kappa | Loss |
| --- | ---: | ---: | ---: | ---: |
| EEG only v2 | 0.813344 | 0.755996 | 0.740190 | 0.849549 |
| EEG+EOG | 0.826007 | 0.770517 | 0.756617 | 0.958570 |
| EEG+EOG, EOG=0 | 0.698359 | 0.576347 | 0.559210 | 1.787396 |

结论：

- `EEG+EOG` 相比 `EEG only` 有小幅收益。
- `EOG=0` 后性能显著低于 `EEG only`。
- 普通双通道模型对 EOG 有强依赖，鲁棒性问题成立。

### 2. 类别 F1

按 20 折混淆矩阵聚合：

| Class | EEG only | EEG+EOG | EOG=0 |
| --- | ---: | ---: | ---: |
| W | 0.916158 | 0.915691 | 0.868880 |
| N1 | 0.474414 | 0.488356 | 0.271600 |
| N2 | 0.836173 | 0.844385 | 0.733175 |
| N3 | 0.780892 | 0.790639 | 0.777395 |
| REM | 0.807493 | 0.855115 | 0.302096 |

重点：

- `EOG` 正常时主要提升 `REM`。
- `EOG=0` 时 `REM` 退化最严重。

### 3. EOG dropout p=0.5 5-fold pilot

固定 pilot folds：

- `fold_00`
- `fold_03`
- `fold_07`
- `fold_13`
- `fold_17`

| Setting | Acc | Macro-F1 | Kappa |
| --- | ---: | ---: | ---: |
| EEG only v2 | 0.843748 | 0.796527 | 0.787471 |
| EEG+EOG baseline normal | 0.864065 | 0.814336 | 0.813336 |
| EEG+EOG baseline EOG=0 | 0.722086 | 0.613360 | 0.596831 |
| EOG dropout p=0.5 normal | 0.852901 | 0.806994 | 0.798938 |
| EOG dropout p=0.5 EOG=0 | 0.840012 | 0.789457 | 0.780873 |

结论：

- `EOG dropout p=0.5` 是当前强 baseline。
- normal 条件只小幅下降。
- `EOG=0` 条件大幅恢复，接近 `EEG only v2`。

## 当前代码状态

### 已有鲁棒方案目录

```text
src/eeg_sleep/robust_schemes/
```

包含：

- `scheme_a_eog_dropout.py`
- `scheme_b_gated_fusion.py`
- `scheme_c_mixture_fusion.py`
- `README.md`

### 已接入 gated fusion

新增配置：

- `configs/paper_sleep_edf_fpz_cz_eeg_eog_gated_fusion_p05.yaml`

新增脚本：

- `scripts/run_gated_fusion_p05_five_folds.sh`

已修改：

- `src/eeg_sleep/models.py`
- `src/eeg_sleep/trainer.py`

支持模型名：

- `deepsleepnet_baseline`
- `deepsleepnet_gated_fusion`
- `deepsleepnet_mixture_fusion`

## 下一步

### 第一优先级：跑 gated fusion 5 折

服务器执行：

```bash
python main.py plan --config configs/paper_sleep_edf_fpz_cz_eeg_eog_gated_fusion_p05.yaml
nohup env PYTHONUNBUFFERED=1 bash scripts/run_gated_fusion_p05_five_folds.sh > gated_fusion_p05_five_folds.log 2>&1 &
```

默认 folds：

```text
fold_00 fold_03 fold_07 fold_13 fold_17
```

输出目录：

```text
results/deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog_gated_fusion_p05
```

### 第二优先级：对比 gated fusion 与 dropout baseline

比较对象：

- `EEG+EOG baseline normal`
- `EEG+EOG baseline EOG=0`
- `EOG dropout p=0.5 normal`
- `EOG dropout p=0.5 EOG=0`
- `Gated fusion p=0.5 normal`
- `Gated fusion p=0.5 EOG=0`

判断标准：

- normal Macro-F1 不应明显低于 `0.806994`
- EOG=0 Macro-F1 应高于 `0.789457`
- REM F1 under EOG=0 应高于或接近 `0.836790`

### 第三优先级：根据结果决定后续路线

如果 gated fusion 有收益：

- 加 gate 统计和可解释性分析
- 测试 EOG noise / flatline / drift
- 再考虑扩更多 folds

如果 gated fusion 没有收益：

- 保留 EOG dropout p=0.5 作为主鲁棒 baseline
- 暂缓复杂模块
- 优先补 EOG 异常类型实验

## 暂不做

- 暂不切换 U-Net / U-Sleep。
- 暂不跑完整 20 折 gated fusion。
- 暂不设计复杂 cross-attention 或 transformer fusion。
