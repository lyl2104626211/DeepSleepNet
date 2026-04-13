# DeepSleepNet 下一步复现计划

## 当前进度

目前已经完成：

- DeepSleepNet 论文主线阅读与中文拆解；
- Sleep-EDF 小子集下载；
- Sleep-EDF 预处理脚本；
- `manifest.json` / `summary.json` 生成；
- 单 epoch Dataset / DataLoader；
- 序列 Dataset / DataLoader；
- PyTorch 版 `DeepFeatureNet`；
- PyTorch 版 `DeepSleepNet`；
- `inspect-dataset` / `inspect-model`；
- 按被试划分 `split.json`；
- 第一阶段 `train-stage1` 训练命令；
- 第一阶段本地 smoke training；
- 训练进度条；
- Windows / Linux 跨平台 manifest 路径兼容。

当前未完成：

- 服务器端处理后数据完整同步；
- 第一阶段完整数据正式训练；
- 第二阶段 `DeepSleepNet` 序列微调；
- 正式评估结果整理；
- 与论文结果对比分析。

## 现在的阻塞点

当前真正阻塞训练继续推进的，不是模型结构，而是服务器端数据不完整。

现象：

- 在 cloud studio / auto_dl 上运行 `train-stage1` 时，代码已经能正确解析 Linux 路径；
- 但服务器端缺少 `data/processed/sleep_edf_subset/SC4211/epoch_00000_W.npy` 这类实际样本文件；
- 说明服务器上的 `.npy` 数据目录没有完整同步，或者 `manifest.json` 与真实目录不一致。

## 第一优先级任务

### 1. 修复服务器数据

先在服务器检查：

```bash
ls data/processed/sleep_edf_subset
ls data/processed/sleep_edf_subset/SC4211 | head
find data/processed/sleep_edf_subset/SC4211 -name '*.npy' | wc -l
```

目标：

- 确认 `SC4201/`、`SC4211/` 目录是否存在；
- 确认每个目录下是否有大量 `.npy` 文件；
- 确认 `manifest.json` 记录的路径能在服务器上真实找到。

若服务器确实缺文件：

- 重新同步整个 `data/processed/sleep_edf_subset/`；
- 或在服务器重新运行预处理，直接生成 Linux 侧处理后数据。

### 2. 在服务器重新跑第一阶段 smoke test

数据修复后先执行：

```bash
python main.py train-stage1 \
  --config configs/base_experiment.yaml \
  --manifest data/processed/sleep_edf_subset/manifest.json \
  --split data/processed/sleep_edf_subset/split.json \
  --output-dir results/deepsleepnet_baseline/stage1_server_test \
  --epochs 2
```

目标：

- 确认服务器端训练链路正常；
- 确认 GPU / CPU 环境、依赖、进度条、权重保存都能正常工作。

## 第二优先级任务

### 3. 准备更大的 Sleep-EDF 数据

不要长期停留在 2 被试子集。

建议：

- 先扩展到论文使用的 `Sleep-EDF SC 20 subjects`；
- 或至少先补到 `8-12` 个被试，保证 train / val / test 更合理；
- 若磁盘和带宽允许，再考虑完整扩展版。

目标：

- 第一阶段训练不再只是在 1 个 train subject 上跑；
- 被试划分开始具备更真实的 cross-subject 意义。

### 4. 重新做被试划分

拿到更多被试后，重新运行：

```bash
python main.py split-subjects \
  --manifest data/processed/sleep_edf_subset/manifest.json \
  --output data/processed/sleep_edf_subset/split.json \
  --seed 42
```

目标：

- 生成更合理的 train / val / test 被试划分；
- 为后续正式实验固定数据划分。

## 第三优先级任务

### 5. 正式跑第一阶段 DeepFeatureNet

在更多被试上执行第一阶段训练，重点关注：

- train loss 是否稳定下降；
- val accuracy / macro_f1 / kappa 是否提升；
- best checkpoint 是否稳定保存；
- 不同 epoch 数下是否出现明显过拟合。

建议保留：

- `training_summary.json`
- 最优权重
- 训练命令
- 数据划分文件

## 第四优先级任务

### 6. 接入第二阶段 DeepSleepNet 序列微调

这是下一块真正的模型复现主任务。

要做的工作：

- 读取第一阶段训练好的 CNN 权重；
- 把 CNN 权重加载到完整 `DeepSleepNet`；
- 用 `SleepEDFSequenceDataset` 构建序列训练集；
- 实现第二阶段序列训练循环；
- 跑验证集，输出序列分类指标。

目标：

- 对齐论文中的“两阶段训练”主流程；
- 从“单 epoch baseline”推进到“完整 DeepSleepNet”。

## 第五优先级任务

### 7. 整理论文复现结果

当两阶段训练都跑通后，整理：

- Accuracy
- Macro-F1
- Cohen's Kappa
- 混淆矩阵
- 与论文结果的差距
- 差距来源分析

要明确回答的问题：

- 当前结果和论文差多少；
- 差距主要来自数据规模、被试划分、训练细节还是硬件设置；
- 哪些部分已对齐，哪些部分还没有。

## 建议执行顺序

1. 先修复服务器上的处理后数据目录。
2. 在服务器重新跑 2 轮 `train-stage1` smoke test。
3. 扩充 Sleep-EDF 被试数量。
4. 重做被试划分。
5. 重新正式训练第一阶段。
6. 接入第二阶段序列微调。
7. 整理论文复现指标和差距分析。

## 当前最实际的一句话

现在不要继续改模型，也不要急着做第二阶段。

先把服务器上的 `data/processed/sleep_edf_subset/` 数据补全，并重新跑通第一阶段训练。
