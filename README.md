# EEG 睡眠分期

这个仓库用于复现并扩展 `DeepSleepNet` 在 `Sleep-EDF` 上的睡眠分期实验。当前主线工作分成两部分：

- 复现原论文的单通道 `EEG` 基线
- 在尽量不改 backbone 的前提下，补一个最小实现版 `EEG+EOG` baseline，为后续 `EOG dropout / robustness` 研究做准备

相关研究想法可参考：

- [docs/deepSleep.md](/c:/Users/21046/Desktop/EEG_SLEEP/docs/deepSleep.md)
- [docs/eog_dropout_robustness_proposal.md](/c:/Users/21046/Desktop/EEG_SLEEP/docs/eog_dropout_robustness_proposal.md)
- [docs/research_progress_2026-05-07.md](/c:/Users/21046/Desktop/EEG_SLEEP/docs/research_progress_2026-05-07.md)

## 当前状态

- 已支持 `DeepSleepNet` 两阶段训练：
  - `stage1`: 单 epoch 特征提取与分类
  - `stage2`: 序列建模与最终评估
- 已支持论文复现所需的数据处理选项：
  - `--channel {Fpz-Cz, Pz-Oz}`
  - `--trim-wake-minutes`
  - `--group-by participant`
  - `--n-folds` 与 `--fold-index`
- 已补充最小实现版 `EEG+EOG baseline`
  - 预处理可导出双通道 `EEG + EOG`
  - dataset/manifest 向后兼容旧的单通道数据
  - DeepSleepNet 前端可自动适配单通道或双通道输入
- `EEG+EOG` 双通道 baseline 已完成 `fold_00` 到 `fold_19` 全部 20 折
- 当前已发现旧单通道 processed 数据与双通道 processed 数据存在部分折口径不一致；正式对比前需要修复或重跑单通道对照

## 这次最新改动

这次改动的目标不是直接做鲁棒融合模块，而是先把一个干净、可比较的 `EEG+EOG baseline` 跑通。

### 1. 预处理层支持 `EEG+EOG`

文件：

- [src/eeg_sleep/preprocess_sleep_edf.py](/c:/Users/21046/Desktop/EEG_SLEEP/src/eeg_sleep/preprocess_sleep_edf.py)

新增能力：

- `preprocess` 支持 `--include-eog`
- `preprocess` 支持 `--eog-channel`
- 每个 epoch 可以保存为：
  - 单通道 `[L]`
  - 双通道 `[C, L]`

为什么这样改：

- 原来仓库默认一条样本只保存单通道 EEG
- 现在为了做 `EEG+EOG` baseline，需要在数据层先把双通道样本准备好
- 但又不能破坏你已经在跑的单通道流程，所以这里做成向后兼容

### 2. manifest 兼容单通道和双通道

文件：

- [src/eeg_sleep/torch_dataset.py](/c:/Users/21046/Desktop/EEG_SLEEP/src/eeg_sleep/torch_dataset.py)

新增能力：

- `SampleRecord` 新增 `channel_names`
- 读取旧 manifest 时，如果只有 `channel_name`，会自动补成单元素列表

为什么这样改：

- 旧数据不需要重做也能继续训练
- 新数据可以明确记录 `EEG + EOG` 两个通道

### 3. DeepSleepNet 输入改成可适配多通道

文件：

- [src/eeg_sleep/models.py](/c:/Users/21046/Desktop/EEG_SLEEP/src/eeg_sleep/models.py)

新增能力：

- `CNNBranch` 不再把 `in_channels` 写死为 `1`
- `DeepFeatureNet` 新增 `input_channels`
- `DeepSleepNet` 同时兼容：
  - 旧格式 `[B, S, L]`
  - 新格式 `[B, S, C, L]`

为什么这样改：

- 你现在最需要的是一个“尽量保持原论文结构”的多通道 baseline
- 所以这次没有上双编码器、没有上 gate，只是让原本的 CNN 前端能接收两通道输入
- 这样后面你做 `EOG dropout`、`gate`、`robust fusion` 时，有一个很干净的 baseline 起点

### 4. 训练流程自动推断输入通道数

文件：

- [src/eeg_sleep/trainer.py](/c:/Users/21046/Desktop/EEG_SLEEP/src/eeg_sleep/trainer.py)

新增能力：

- `train_stage1`
- `train_stage2`
- `evaluate_stage2`

都会根据样本 shape 自动判断当前是：

- 单通道 `[L]`
- 多通道 `[C, L]`

为什么这样改：

- 避免你为单通道和双通道维护两套训练脚本
- 保持现有命令入口不变，只改数据与配置即可

### 5. CLI 补充 `EEG+EOG` 预处理入口

文件：

- [src/eeg_sleep/cli.py](/c:/Users/21046/Desktop/EEG_SLEEP/src/eeg_sleep/cli.py)

新增参数：

- `--include-eog`
- `--eog-channel`

同时 `inspect-dataset` 和 `inspect-model` 也能显示多通道 shape。

### 6. 新增 `EEG+EOG` baseline 配置

文件：

- [configs/paper_sleep_edf_fpz_cz_eeg_eog.yaml](/c:/Users/21046/Desktop/EEG_SLEEP/configs/paper_sleep_edf_fpz_cz_eeg_eog.yaml)

用途：

- 为论文路线下的 `Fpz-Cz + EOG` baseline 提供一份单独配置
- 不干扰现有的单通道配置 [configs/paper_sleep_edf_fpz_cz.yaml](/c:/Users/21046/Desktop/EEG_SLEEP/configs/paper_sleep_edf_fpz_cz.yaml)

## 为什么这次实现是“最小实现”

这次没有做下面这些：

- 没有做 `EEG encoder + EOG encoder` 双分支结构
- 没有做 reliability gate
- 没有做训练期 `EOG dropout`
- 没有做测试期 `EOG corruption` 注入

原因是现在你最需要先回答两个问题：

1. `EEG+EOG` 正常条件下，baseline 能不能比 `EEG only` 更好
2. 一旦 `EOG` 缺失或损坏，普通多通道 baseline 会不会明显下降

只有先把这个 baseline 跑出来，后面的 robustness 研究才有比较基础。

## 环境安装

如果本地使用 `uv`：

```bash
uv sync
uv run python main.py -h
```

如果服务器使用 `conda`：

```bash
conda create -n eggsleep python=3.11 -y
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eggsleep
pip install torch torchvision torchaudio
pip install numpy PyYAML scikit-learn tqdm mne
pip install -e .
```

## 常用流程

### 1. 单通道论文复现流程

```bash
uv run python main.py preprocess \
  --input-dir data/raw/sleep_edf_sleep_cassette \
  --output-dir data/processed/sleep_edf_paper_fpz_cz \
  --channel Fpz-Cz \
  --trim-wake-minutes 30

uv run python main.py split-subjects \
  --manifest data/processed/sleep_edf_paper_fpz_cz/manifest.json \
  --output data/processed/sleep_edf_paper_fpz_cz/fold_00.json \
  --group-by participant \
  --n-folds 20 \
  --fold-index 0 \
  --seed 42

uv run python main.py train-stage1 \
  --config configs/paper_sleep_edf_fpz_cz.yaml \
  --manifest data/processed/sleep_edf_paper_fpz_cz/manifest.json \
  --split data/processed/sleep_edf_paper_fpz_cz/fold_00.json \
  --output-dir results/deepsleepnet_sleep_edf_paper_fpz_cz/fold_00/stage1

uv run python main.py train-stage2 \
  --config configs/paper_sleep_edf_fpz_cz.yaml \
  --manifest data/processed/sleep_edf_paper_fpz_cz/manifest.json \
  --split data/processed/sleep_edf_paper_fpz_cz/fold_00.json \
  --stage1-checkpoint results/deepsleepnet_sleep_edf_paper_fpz_cz/fold_00/stage1/best_model.pt \
  --output-dir results/deepsleepnet_sleep_edf_paper_fpz_cz/fold_00/stage2

uv run python main.py evaluate-stage2 \
  --config configs/paper_sleep_edf_fpz_cz.yaml \
  --manifest data/processed/sleep_edf_paper_fpz_cz/manifest.json \
  --split data/processed/sleep_edf_paper_fpz_cz/fold_00.json \
  --checkpoint results/deepsleepnet_sleep_edf_paper_fpz_cz/fold_00/stage2/best_model.pt \
  --subset test \
  --output-dir results/deepsleepnet_sleep_edf_paper_fpz_cz/fold_00/eval_test
```

### 2. 最小实现版 `EEG+EOG` baseline

先做双通道预处理：

```bash
uv run python main.py preprocess \
  --input-dir data/raw/sleep_edf_sleep_cassette \
  --output-dir data/processed/sleep_edf_paper_fpz_cz_eeg_eog \
  --channel Fpz-Cz \
  --include-eog \
  --trim-wake-minutes 30
```

如果默认 EOG 名称没有匹配到，可以显式指定：

```bash
uv run python main.py preprocess \
  --input-dir data/raw/sleep_edf_sleep_cassette \
  --output-dir data/processed/sleep_edf_paper_fpz_cz_eeg_eog \
  --channel Fpz-Cz \
  --include-eog \
  --eog-channel "EOG horizontal" \
  --trim-wake-minutes 30
```

再做 20 折中的某一折，例如 `fold_00`：

```bash
uv run python main.py split-subjects \
  --manifest data/processed/sleep_edf_paper_fpz_cz_eeg_eog/manifest.json \
  --output data/processed/sleep_edf_paper_fpz_cz_eeg_eog/fold_00.json \
  --group-by participant \
  --n-folds 20 \
  --fold-index 0 \
  --seed 42

uv run python main.py train-stage1 \
  --config configs/paper_sleep_edf_fpz_cz_eeg_eog.yaml \
  --manifest data/processed/sleep_edf_paper_fpz_cz_eeg_eog/manifest.json \
  --split data/processed/sleep_edf_paper_fpz_cz_eeg_eog/fold_00.json \
  --output-dir results/deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog/fold_00/stage1

uv run python main.py train-stage2 \
  --config configs/paper_sleep_edf_fpz_cz_eeg_eog.yaml \
  --manifest data/processed/sleep_edf_paper_fpz_cz_eeg_eog/manifest.json \
  --split data/processed/sleep_edf_paper_fpz_cz_eeg_eog/fold_00.json \
  --stage1-checkpoint results/deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog/fold_00/stage1/best_model.pt \
  --output-dir results/deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog/fold_00/stage2

uv run python main.py evaluate-stage2 \
  --config configs/paper_sleep_edf_fpz_cz_eeg_eog.yaml \
  --manifest data/processed/sleep_edf_paper_fpz_cz_eeg_eog/manifest.json \
  --split data/processed/sleep_edf_paper_fpz_cz_eeg_eog/fold_00.json \
  --checkpoint results/deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog/fold_00/stage2/best_model.pt \
  --subset test \
  --output-dir results/deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog/fold_00/eval_test
```

## 当前建议的研究推进顺序

建议先按下面顺序推进：

1. 修复或重跑同口径 `EEG only` 对照
2. 整理 `EEG only` vs `EEG+EOG` 的 20 折正式结果
3. 测试时做 `EOG = 0`
4. 再做 `EOG dropout training`
5. 最后再做 `gate / robust fusion`

这样最容易把问题定义讲清楚，也最方便写成论文实验主线。

## 结果与输出目录

常见输出目录：

- 单通道论文路线：
  - `results/deepsleepnet_sleep_edf_paper_fpz_cz/...`
- 双通道最小 baseline：
  - `results/deepsleepnet_sleep_edf_paper_fpz_cz_eeg_eog/...`

典型文件包括：

- `training_summary.json`
- `best_model.pt`
- `evaluation_test.json`
- `confusion_matrix_test.json`

## 目录说明

- `configs/`: 实验配置
- `data/`: 原始数据与处理后数据
- `docs/`: 复现记录、研究计划和补充说明
- `paper/`: 参考论文
- `results/`: 训练结果与评估输出
- `src/`: 预处理、数据集、模型、训练与评估代码

## 说明

- 处理后的数据和训练结果默认不随 git 一起提交
- 当前 `EEG+EOG` 只是 baseline，不代表最终鲁棒模型
- 如果后续你要继续做 `EOG dropout robustness`，建议优先在这个 baseline 上扩展，而不是重新起一套结构
