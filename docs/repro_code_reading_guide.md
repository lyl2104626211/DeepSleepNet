# 复现论文时该看哪些代码

这份文档的目标很简单：

- 不是让你把整个项目逐行看完
- 而是告诉你，当前为了复现 DeepSleepNet，你最应该看哪些代码
- 以及每一部分到底要看懂什么

## 结论先说

现在不需要把整个项目代码全部看完。

你当前最应该看懂的，其实只有 4 类内容：

1. 模型是怎么搭的
2. 数据是怎么送进模型的
3. stage1 和 stage2 是怎么训练的
4. 实验配置参数是怎么控制训练流程的

换句话说，你现在最该看的是：

1. `src/eeg_sleep/models.py`
2. `src/eeg_sleep/trainer.py`
3. `src/eeg_sleep/torch_dataset.py`
4. `configs/base_experiment.yaml`

`src/eeg_sleep/cli.py` 只需要知道命令入口，不需要花很多时间逐行细看。

---

## 第一优先级：先看模型代码

文件：

- `src/eeg_sleep/models.py`

这一部分最重要，因为复现论文的核心就是模型结构。

你要重点看明白这些问题：

### 1. stage1 是什么模型

看 `DeepFeatureNet`

你要搞清楚：

- 输入是什么形状
- 为什么有两个 CNN 分支
- 两个分支最后怎么拼接
- stage1 最终输出的是什么

### 2. stage2 是什么模型

看 `DeepSleepNet`

你要搞清楚：

- stage2 为什么不是直接重复 stage1
- stage2 是怎么复用 stage1 的 CNN 特征提取器
- Bi-LSTM 是怎么接在 CNN 后面的
- shortcut / residual 是怎么加进去的

### 3. 输入输出 shape 是什么

这是最关键的。

你至少要能自己说清楚：

- stage1 输入：`[B, L]` 或 `[B, 1, L]`
- stage1 输出：`[B, 5]`
- stage2 输入：`[B, S, L]`
- stage2 输出：`[B, S, 5]`

如果这一步没看懂，后面训练流程也很难真正看懂。

---

## 第二优先级：再看训练流程

文件：

- `src/eeg_sleep/trainer.py`

这部分决定“模型怎么真正跑起来”。

你不用一开始把所有细节都记住，但要先抓住主线。

### 1. stage1 怎么训练

重点看：

- `train_stage1(...)`

你要看懂：

- 训练集和验证集怎么构建
- model / loss / optimizer 怎么定义
- 每个 epoch 做了什么
- 最优 checkpoint 怎么保存

### 2. stage2 怎么训练

重点看：

- `train_stage2(...)`

你要看懂：

- stage2 为什么要用 sequence dataset
- stage1 checkpoint 是怎么加载进 stage2 的
- stage2 的 optimizer 为什么和 stage1 不一样
- CNN 学习率和时序部分学习率为什么会分开

### 3. stage2 怎么评估

重点看：

- `evaluate_stage2(...)`
- `_evaluate_sequence_model(...)`

你要搞清楚：

- 为什么 stage2 验证时不是直接拿窗口结果就结束
- 为什么要把重叠窗口的预测重新聚合回 epoch 级结果

这个点非常重要，因为它直接关系到你最后的指标是否合理。

---

## 第三优先级：看数据是怎么喂进模型的

文件：

- `src/eeg_sleep/torch_dataset.py`

这部分不是论文本体，但它决定你实际训练时喂给模型的数据长什么样。

### 1. stage1 看什么

重点看：

- `SleepEDFEpochDataset`
- `create_dataloader(...)`

你要搞清楚：

- 一个 sample 里有哪些字段
- `signal` 和 `label` 是怎么来的
- 一个 batch 最后长什么样

### 2. stage2 看什么

重点看：

- `SleepEDFSequenceDataset`
- `create_sequence_dataloader(...)`

你要搞清楚：

- 为什么要把连续 epoch 组成窗口
- `sequence_length` 是怎么起作用的
- 一个 stage2 batch 的 `signals` 和 `labels` 是什么形状

如果这一步没看懂，你就会只知道“stage2 是 LSTM”，但不知道它到底吃进去的是什么。

---

## 第四优先级：看实验配置

文件：

- `configs/base_experiment.yaml`

这个文件不用久看，但一定要知道它控制了什么。

你要重点关注：

- `batch_size`
- `epochs`
- `learning_rate`
- `stage2_batch_size`
- `stage2_sequence_length`
- `stage2_sequence_stride`
- `stage2_eval_stride`
- `stage2_cnn_learning_rate`
- `stage2_sequence_learning_rate`

你至少要知道：

- 哪些参数控制 stage1
- 哪些参数控制 stage2
- 哪些参数会直接影响显存、速度、训练稳定性

---

## 第五优先级：知道 CLI 怎么进来就够了

文件：

- `src/eeg_sleep/cli.py`

这个文件现在不需要深读。

你只要知道几件事：

- `train-stage1` 命令从哪里进
- `train-stage2` 命令从哪里进
- `evaluate-stage2` 命令从哪里进
- 这些命令最后会调用 `trainer.py` 里的哪些函数

现在 `cli.py` 里参数区已经加了中文注释，所以这部分你后面回头看会轻松很多。

---

## 哪些代码现在不用花太多时间

这些不是完全不用看，而是当前复现主线里优先级没那么高。

### 暂时低优先级

- `src/eeg_sleep/preprocess_sleep_edf.py`
- `src/eeg_sleep/metrics.py`
- `src/eeg_sleep/download_sleep_edf.py`
- `src/eeg_sleep/datasets.py`
- `src/eeg_sleep/__init__.py`

### 为什么现在不用深看

因为你当前的主问题不是：

- 数据怎么下载
- 指标函数怎么封装
- 包结构怎么组织

你当前真正要搞明白的是：

- 模型结构
- 数据输入形式
- 两阶段训练逻辑

---

## 推荐阅读顺序

如果你现在准备开始细看代码，我建议按这个顺序：

1. `src/eeg_sleep/models.py`
2. `src/eeg_sleep/torch_dataset.py`
3. `src/eeg_sleep/trainer.py`
4. `configs/base_experiment.yaml`
5. `src/eeg_sleep/cli.py`

这个顺序的原因很简单：

- 先知道模型长什么样
- 再知道数据怎么进去
- 再看训练怎么把两者串起来
- 最后再看参数和命令入口

这样最省脑力，也最不容易迷路。

---

## 你现在不该做什么

### 1. 不要试图把整个项目逐行看完

这是最浪费时间的方式。

因为很多工程代码只是为了：

- 路径兼容
- 命令行解析
- 保存日志
- 保存 checkpoint

这些对“理解论文主线”不是第一优先级。

### 2. 不要一开始陷进实现细节

比如：

- 某个 helper 为什么这样命名
- 某个 DataLoader 为什么这样写
- 某个 JSON 是怎么保存的

这些都不是你现在最重要的问题。

你现在最重要的问题只有三个：

- 模型怎么搭
- 数据怎么喂
- stage1 / stage2 怎么训

---

## 最后一句话

当前复现论文，不需要把整个项目代码全部看完。

你只需要先看懂这三件事：

1. `models.py` 里的模型结构
2. `torch_dataset.py` 里的数据输入形式
3. `trainer.py` 里的两阶段训练流程

只要这三块看明白，整个项目主线你就已经抓住了。
