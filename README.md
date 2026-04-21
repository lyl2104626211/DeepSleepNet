# EEG 睡眠分期

这个仓库用于复现和扩展 EEG 睡眠分期实验，当前主线是 `DeepSleepNet` 在 `Sleep-EDF` 上的实现与论文对齐。

## 当前状态

- 已完成 `Sleep-EDF` 固定划分 baseline：
  - `stage1` 训练完成
  - `stage2` 训练完成
  - `val / test` 评估完成
- 论文版 `fold_00` 已完整跑通：
  - `stage1` 已完成
  - `stage2` 已完成并落盘
  - `eval_test` 已导出
- 服务器侧已开始继续推进后续折：
  - 已批量生成到至少 `fold_13.json`
  - 后续折仍在分批训练与评估中
- 已补齐论文对齐所需的关键能力：
  - `preprocess` 支持 `--channel {Fpz-Cz,Pz-Oz}`
  - `preprocess` 支持 `--trim-wake-minutes`
  - `split-subjects` 支持 `--group-by participant`
  - `split-subjects` 支持 `--n-folds` 与 `--fold-index`
- 论文版配置已准备好：
  - `configs/paper_sleep_edf_fpz_cz.yaml`

## 当前最好结果

固定划分 baseline 的正式结果：

- `val accuracy = 0.9225`
- `val macro_f1 = 0.8162`
- `val kappa = 0.8473`
- `test accuracy = 0.9236`
- `test macro_f1 = 0.8021`
- `test kappa = 0.8464`

当前结果说明：

- `stage2` 明显优于 `stage1`，说明时序上下文有效。
- `val` 与 `test` 接近，说明固定划分下泛化比较稳定。
- `N1` 仍然是最难类别，这和睡眠分期任务本身的特点一致。

## 论文版 `fold_00`

当前论文版第一折测试结果：

- `accuracy = 0.8536`
- `macro_f1 = 0.7906`
- `kappa = 0.7893`
- `loss = 0.7997`

结果解读：

- 单折结果已经证明论文版流程可以完整跑通。
- `N1` 仍然最难，主要和 `W / N2 / REM` 混淆，这个现象是合理的。
- 该结果目前只能视为单折阶段性结果，最终仍要看多折平均。

## 和论文的关系

- 当前 fixed-split baseline 已经足以说明 `DeepSleepNet` 主线复现成功。
- 但严格来说，这还不是和原论文完全等价的结果，因为 baseline 不是论文中的完整评估协议。
- 原论文在 `Sleep-EDF` 上更接近 `20-fold cross-validation`，所以最终仍要看多折平均结果。
- 当前论文版只完成了 `fold_00` 的正式评估，因此还不能把单折结果直接写成最终论文对齐结论。

## 论文对齐目标

当前先只做 `Sleep-EDF`，不扩展到第二个数据集。

对齐目标是：

- 通道使用 `Fpz-Cz`
- 只保留睡眠前后各 `30` 分钟清醒期
- 按 `participant` 分组
- 做 `20-fold cross-validation`
- 当前先分批跑完 `20` 折，再做整体汇总

论文版目标训练配置：

- `stage1 = 100 epochs`
- `stage2 = 200 epochs`
- `stage2 cnn_lr = 1e-6`
- `stage2 sequence_lr = 1e-4`
- `stage2 gradient_clip_norm = 10`

实际服务器说明：

- 由于服务器资源和时长限制，当前已完成的 `fold_00` 是按 `stage2 = 100 epochs` 跑出的阶段性结果。
- 如果后续折也统一采用 `stage2 = 100 epochs`，需要在最终实验报告里明确写明这一点。

## 常用命令

安装环境：

```bash
uv sync
```

查看命令入口：

```bash
uv run python main.py -h
```

如果服务器用的是 `conda` 环境，推荐最小安装方式：

```bash
conda create -n eggsleep python=3.11 -y
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eggsleep
pip install torch torchvision torchaudio
pip install numpy PyYAML scikit-learn tqdm mne
pip install -e .
```

如果 `conda activate` 报错，需要先手动加载：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
```

## 固定划分 baseline 流程

```bash
uv run python main.py preprocess --input-dir data/raw/sleep_edf_sleep_cassette --output-dir data/processed/sleep_edf_sleep_cassette
uv run python main.py split-subjects --manifest data/processed/sleep_edf_sleep_cassette/manifest.json --output data/processed/sleep_edf_sleep_cassette/split.json --seed 42
uv run python main.py train-stage1 --config configs/base_experiment_full.yaml --manifest data/processed/sleep_edf_sleep_cassette/manifest.json --split data/processed/sleep_edf_sleep_cassette/split.json --output-dir results/deepsleepnet_stage1_full/stage1
uv run python main.py train-stage2 --config configs/base_experiment_full.yaml --manifest data/processed/sleep_edf_sleep_cassette/manifest.json --split data/processed/sleep_edf_sleep_cassette/split.json --stage1-checkpoint results/deepsleepnet_stage1_full/stage1/best_model.pt --output-dir results/deepsleepnet_stage1_full/stage2
uv run python main.py evaluate-stage2 --config configs/base_experiment_full.yaml --manifest data/processed/sleep_edf_sleep_cassette/manifest.json --split data/processed/sleep_edf_sleep_cassette/split.json --checkpoint results/deepsleepnet_stage1_full/stage2/best_model.pt --subset test --output-dir results/deepsleepnet_stage1_full/eval_test
```

## 论文版 `fold_00` 流程

```bash
uv run python main.py preprocess --input-dir data/raw/sleep_edf_sleep_cassette --output-dir data/processed/sleep_edf_paper_fpz_cz --channel Fpz-Cz --trim-wake-minutes 30
uv run python main.py split-subjects --manifest data/processed/sleep_edf_paper_fpz_cz/manifest.json --output data/processed/sleep_edf_paper_fpz_cz/fold_00.json --group-by participant --n-folds 20 --fold-index 0 --seed 42
uv run python main.py train-stage1 --config configs/paper_sleep_edf_fpz_cz.yaml --manifest data/processed/sleep_edf_paper_fpz_cz/manifest.json --split data/processed/sleep_edf_paper_fpz_cz/fold_00.json --output-dir results/deepsleepnet_sleep_edf_paper_fpz_cz/fold_00/stage1
uv run python main.py train-stage2 --config configs/paper_sleep_edf_fpz_cz.yaml --manifest data/processed/sleep_edf_paper_fpz_cz/manifest.json --split data/processed/sleep_edf_paper_fpz_cz/fold_00.json --stage1-checkpoint results/deepsleepnet_sleep_edf_paper_fpz_cz/fold_00/stage1/best_model.pt --output-dir results/deepsleepnet_sleep_edf_paper_fpz_cz/fold_00/stage2
uv run python main.py evaluate-stage2 --config configs/paper_sleep_edf_fpz_cz.yaml --manifest data/processed/sleep_edf_paper_fpz_cz/manifest.json --split data/processed/sleep_edf_paper_fpz_cz/fold_00.json --checkpoint results/deepsleepnet_sleep_edf_paper_fpz_cz/fold_00/stage2/best_model.pt --subset test --output-dir results/deepsleepnet_sleep_edf_paper_fpz_cz/fold_00/eval_test
```

## 结果位置

固定划分 baseline 结果：

- `results/deepsleepnet_stage1_full/stage1`
- `results/deepsleepnet_stage1_full/stage2`
- `results/deepsleepnet_stage1_full/eval_val`
- `results/deepsleepnet_stage1_full/eval_test`

论文版 `fold_00` 结果：

- `results/deepsleepnet_sleep_edf_paper_fpz_cz/fold_00/stage1`
- `results/deepsleepnet_sleep_edf_paper_fpz_cz/fold_00/stage2`
- `results/deepsleepnet_sleep_edf_paper_fpz_cz/fold_00/eval_test`

## 目录说明

- `configs/`：实验配置
- `conversation/`：历史对话记录
- `data/`：原始数据与处理后数据
- `docs/`：复现计划、模板和补充说明
- `results/`：训练结果、评估指标、混淆矩阵、checkpoint
- `src/`：数据处理、数据集、模型、训练、评估代码

## 说明

- 代码已经同步到 Gitee，但 `data/` 和 `results/` 默认不会跟着 git 一起推送。
- 处理后的数据如果需要迁移，建议单独打包，比如 `sleep_edf_paper_fpz_cz.tar.gz`。
- 论文版 `k-fold` 设置下，`val_subjects = []` 和 `best_val_macro_f1 = NaN` 在 `stage1` 摘要里是正常现象，因为这条线没有单独验证集。
- 如果批量训练中途中断，已经完成并落盘的折不需要重跑，只需从中断折继续。
- 如果服务器上的处理后数据和 `manifest.json` 不一致，优先重新解压 `sleep_edf_paper_fpz_cz*.tar.gz` 恢复数据目录。
- 下一次开工直接看根目录 `plan.md` 即可。
