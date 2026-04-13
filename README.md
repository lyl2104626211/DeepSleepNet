# EEG 睡眠分期

这个工作区用于复现和扩展 EEG 睡眠分期基线，当前从 DeepSleepNet 开始。

## 当前目标

- 数据集：Sleep-EDF
- 任务：5 分类睡眠分期（`W`、`N1`、`N2`、`N3`、`REM`）
- 设置：跨被试评估
- 指标：Accuracy、Macro-F1、Cohen's Kappa

## 建议顺序

1. 阅读 `paper/DeepSleepNet_2017.pdf`，先提取任务设置。
2. 确认数据格式和预处理流程。
3. 在做任何方法改进前，先搭一个能跑通的 baseline。
4. 先和 DeepSleepNet 对齐，再阅读 `paper/U-Sleep_2021.pdf`。
5. 只有在 baseline 干净稳定后，再尝试小改进。

## 目录说明

- `configs/`：实验配置
- `data/`：原始数据和处理后数据
- `docs/`：复现笔记和论文阅读笔记
- `results/`：日志、指标、混淆矩阵、模型权重
- `src/`：数据集、模型、训练、评估代码

## 第一阶段目标

第一阶段的目标不是论文级创新，而是建立一个干净、可复现的 baseline，能够：

- 读取 Sleep-EDF 数据；
- 构建带标签的 30 秒 epoch；
- 成功训练一个基线模型；
- 输出 Accuracy、Macro-F1 和 Kappa；
- 保存混淆矩阵。

## 当前代码状态

目前已经完成的部分包括：

- 配置读取；
- `Sleep-EDF` 子集预处理脚本；
- `manifest.json` 驱动的单 epoch Dataset / DataLoader；
- 按被试组织的序列 Dataset / DataLoader；
- PyTorch 版 `DeepFeatureNet` 前向；
- PyTorch 版 `DeepSleepNet` 前向；
- 训练计划生成入口；
- 指标计算接口占位。

当前还没有完成真正的两阶段训练循环和最终评估导出。

## 环境准备

本项目现在默认使用 `uv` 管理环境和依赖。

先在项目根目录执行：

```bash
uv sync
```

如果你只想先创建虚拟环境，也可以执行：

```bash
uv venv
```

然后再用 `uv sync` 按 `pyproject.toml` 安装依赖。

## 运行方式

先在项目根目录执行：

```bash
uv run python main.py plan
```

运行后会根据配置文件生成一份训练计划：

- `results/deepsleepnet_baseline/training_plan.json`

## 下一步

下一步优先做下面两件事：

1. 实现按被试划分的数据切分和训练配置固化。
2. 接入 DeepSleepNet 的两阶段训练循环。

## 预处理命令

完成 `uv sync` 后，可以运行：

```bash
uv run python main.py preprocess --input-dir data/raw/sleep_edf_subset --output-dir data/processed/sleep_edf_subset
```

这个命令会尝试完成：

- 自动匹配 `PSG` 和 `Hypnogram` 文件；
- 选择常见单通道 EEG；
- 读取睡眠标注；
- 按 30 秒切分 epoch；
- 把标签映射到 `W / N1 / N2 / N3 / REM`；
- 保存每个 epoch 的 `.npy` 文件和总索引文件。

补充说明：

- 预处理生成的 `manifest.json` 现在统一使用 `/` 作为相对路径分隔符，方便在 Windows 和 Linux 之间迁移数据。

## 检查 Dataset

完成 `uv sync` 后，可以继续运行：

```bash
uv run python main.py inspect-dataset --manifest data/processed/sleep_edf_subset/manifest.json --batch-size 4
```

这个命令会检查：

- `manifest.json` 能否被正确读取；
- 单个样本能否正确加载；
- `.npy` 信号长度是否正常；
- `DataLoader` 能否正确组成 batch。

## 检查模型前向

完成 `uv sync` 后，可以继续运行：

```bash
uv run python main.py inspect-model --manifest data/processed/sleep_edf_subset/manifest.json --batch-size 4 --sequence-batch-size 2 --sequence-length 25
```

这个命令会检查：

- `DeepFeatureNet` 是否能对真实 epoch batch 做前向；
- 表征维度和分类 logits 形状是否正确；
- `DeepSleepNet` 是否能对连续 `25` 个 epoch 的序列做前向；
- 序列输入输出形状是否符合预期。

## 完整数据集流程

如果你准备在服务器上直接下载 `Sleep-EDF Expanded` 的 `sleep-cassette`，并跑第一阶段训练，可以按下面顺序执行。

### 1. 查看下载计划

先用 `dry-run` 看看代码准备下载哪些文件：

```bash
uv run python main.py download-sleep-edf --output-dir data/raw/sleep_edf_sleep_cassette --max-records 2 --dry-run
```

如果你当前不是用 `uv`，也可以把前缀 `uv run` 去掉，直接执行 `python main.py ...`。

### 2. 下载原始 EDF 文件

下载完整 `sleep-cassette`：

```bash
uv run python main.py download-sleep-edf --output-dir data/raw/sleep_edf_sleep_cassette
```

如果你想先试一部分记录，例如前 `20` 组：

```bash
uv run python main.py download-sleep-edf --output-dir data/raw/sleep_edf_sleep_cassette --max-records 20
```

### 3. 预处理完整数据

```bash
uv run python main.py preprocess --input-dir data/raw/sleep_edf_sleep_cassette --output-dir data/processed/sleep_edf_sleep_cassette
```

### 4. 按被试划分

```bash
uv run python main.py split-subjects --manifest data/processed/sleep_edf_sleep_cassette/manifest.json --output data/processed/sleep_edf_sleep_cassette/split.json --seed 42
```

### 5. 运行第一阶段训练

完整数据默认配置文件为：

- `configs/base_experiment_full.yaml`

训练命令：

```bash
uv run python main.py train-stage1 --config configs/base_experiment_full.yaml --manifest data/processed/sleep_edf_sleep_cassette/manifest.json --split data/processed/sleep_edf_sleep_cassette/split.json --output-dir results/deepsleepnet_stage1_full/stage1
```

### 6. 完整数据默认训练配置

当前完整数据配置里默认设置为：

- `batch_size: 128`
- `epochs: 50`
- `num_workers: 8`
- `pin_memory: true`

如果你在服务器上显存不够，可以把 `configs/base_experiment_full.yaml` 里的 `batch_size` 改小，例如改成 `64`。

## 说明

- 完整数据下载现在由代码直接从 PhysioNet 拉取，不需要先手工把大数据下载到本地再上传。
- 预处理生成的 `manifest.json` 会统一写成 `/` 路径，方便 Windows 和 Linux 之间迁移。
- 当前这套流程已经足够跑通“完整数据集的第一阶段训练”，但还没有对齐论文里的 `20-fold cross-validation`。
