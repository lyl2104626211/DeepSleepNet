# 近年睡眠分期与缺失模态鲁棒性论文索引

整理时间：2026-05-15

当前课题定位：面向 EOG 缺失/损坏场景的鲁棒睡眠分期。重点关注 `EEG+EOG`、缺失/损坏模态、质量感知融合、EEG->EOG 特征恢复、跨 backbone 验证。

## 优先阅读顺序

1. `CIMSleepNet_2024.pdf`
2. `CoRe-Sleep_2023.pdf`
3. `U-Sleep_2021.pdf`
4. `FlexSleepTransformer_2024.pdf`
5. `XSleepNet_2021.pdf`
6. `SalientSleepNet_2021.pdf`
7. `EEGPT_2024.pdf`
8. `SleepGPT_2026.pdf`
9. `SleepFM_2024.pdf`
10. `ArbitrarySensorSleepStaging_2024.pdf`
11. `SMIL_2021.pdf`
12. `Missing_Modality_Survey_2024.pdf`

## 论文笔记

### 1. CIMSleepNet: Robust Sleep Staging over Incomplete Multimodal Physiological Signals via Contrastive Imagination

- 本地文件：`CIMSleepNet_2024.pdf`
- 年份/来源：NeurIPS 2024
- 实验目的：解决自动睡眠分期中 EEG、EOG、EMG 等多模态生理信号任意缺失时的性能退化问题。
- 实验方法：提出 CIMSleepNet，核心包括 modal awareness imagination module, semantic and modal calibration contrastive learning，以及多层级跨分支时序注意力；目标是在缺失模态下恢复或校准共享表示，同时保持模态特异性。
- 实验结果：在 5 个多模态睡眠数据集和多种缺失模态模式下，整体优于竞争方法。
- 对当前课题的价值：这是最直接的强相关工作。你的方法需要和它区分：你更聚焦 EOG 缺失/损坏、EOG 质量感知、EEG->EOG 特征生成，以及真实/生成 EOG 动态融合。

### 2. CoRe-Sleep: A Multimodal Fusion Framework for Time Series Robust to Imperfect Modalities

- 本地文件：`CoRe-Sleep_2023.pdf`
- 年份/来源：2023, arXiv/journal version
- 实验目的：提升睡眠分期模型在 noisy modality、missing modality segment 和不完整训练数据下的鲁棒性。
- 实验方法：提出 Coordinated Representation multimodal fusion network，通过协调多模态表示，让模型能处理噪声或缺失模态片段。
- 实验结果：在 SHHS-1 上，单个模型同时支持 multimodal 和 unimodal 测试，并在多模态/单模态测试下取得强结果；论文结论指出，多模态训练对单模态测试也有正向影响。
- 对当前课题的价值：非常适合作为你论文 related work 中的“imperfect modalities”主线文献。你的 `zero / flatline / noise / drift / artifact` 设计可以借鉴它的问题表述。

### 3. U-Sleep: resilient high-frequency sleep staging

- 本地文件：`U-Sleep_2021.pdf`
- 年份/来源：npj Digital Medicine 2021
- 实验目的：构建跨数据集、跨 PSG 协议、跨 EEG/EOG 通道组合都能工作的自动睡眠分期系统。
- 实验方法：使用 fully convolutional U-Net 风格结构，在大规模 PSG 上训练；输入为任意标准 EEG 和 EOG 通道组合，并可输出高时间分辨率睡眠阶段表示。
- 实验结果：训练和评估覆盖 15,660 名受试者、16 个临床研究；跨 21 个数据集平均 macro F1 约 0.79；在共识标注数据上达到接近最佳人工专家的水平。
- 对当前课题的价值：适合作为第二 backbone。它天然支持 EEG/EOG 通道组合，很适合验证你的鲁棒模块不是 DeepSleepNet 特例。

### 4. FlexSleepTransformer: a transformer-based sleep staging model with flexible input channel configurations

- 本地文件：`FlexSleepTransformer_2024.pdf`
- 年份/来源：Scientific Reports 2024
- 实验目的：解决不同睡眠中心 PSG 通道数量不一致导致模型难以统一训练和部署的问题。
- 实验方法：提出能处理可变输入通道数量的 Transformer 睡眠分期模型，并在 SleepEDF-78 和 SleepUHS 两个通道配置不同的数据集上联合训练。
- 实验结果：联合训练模型达到分别单独训练模型约 98% 的准确率；跨数据集测试时优于只在单一数据集训练的模型；整体超过 CNN/RNN 类基线。
- 对当前课题的价值：支撑“临床中通道配置不稳定”的现实动机。你的 EOG 缺失/损坏问题可以看作更具体的可变通道可靠性问题。

### 5. XSleepNet: Multi-View Sequential Model for Automatic Sleep Staging

- 本地文件：`XSleepNet_2021.pdf`
- 年份/来源：IEEE TPAMI 2021
- 实验目的：解决 raw PSG signal 和 time-frequency image 两种视角如何联合学习的问题。
- 实验方法：提出 sequence-to-sequence 多视图睡眠分期模型，同时学习原始信号和时频图表示；训练中根据各视图的泛化/过拟合情况动态调整梯度融合权重。
- 实验结果：在 5 个不同规模数据集上，稳定优于单视图模型和简单融合多视图模型，并提升已有 state-of-the-art。
- 对当前课题的价值：不是缺失 EOG 论文，但它的“不同视图贡献不同、需要动态融合”的思想可用于支撑你的动态 gating/fusion 设计。

### 6. SalientSleepNet: Multimodal Salient Wave Detection Network for Sleep Staging

- 本地文件：`SalientSleepNet_2021.pdf`
- 年份/来源：IJCAI 2021
- 实验目的：从多模态睡眠数据中提取对分期最关键的 salient waves，并建模睡眠阶段转移规律。
- 实验方法：使用类似 U2-Net 的双流结构提取多模态显著特征；加入多尺度转移规则建模模块和多模态注意力模块。
- 实验结果：在两个数据集上优于当时 state-of-the-art baseline，并且参数量较小。
- 对当前课题的价值：可作为 EEG/EOG 多模态融合和注意力机制的参考，但它主要解决正常多模态性能，不是缺失 EOG 鲁棒性。

### 7. EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals

- 本地文件：`EEGPT_2024.pdf`
- 年份/来源：NeurIPS 2024
- 实验目的：学习通用、鲁棒的 EEG 表征，缓解 EEG 低信噪比、跨被试差异和通道不匹配问题。
- 实验方法：提出约 10M 参数的预训练 Transformer，使用 mask-based dual self-supervised learning 和 spatio-temporal representation alignment。
- 实验结果：在多个 EEG 下游任务上，linear probing 即达到 state-of-the-art，验证预训练 EEG 表征的可扩展性。
- 对当前课题的价值：适合作为第三 backbone 或特征提取器。你的模块如果能接在 EEGPT 表征后面，会比只在 DeepSleepNet 上验证更有说服力。

### 8. A unified time-frequency foundation model for sleep decoding

- 本地文件：`SleepGPT_2026.pdf`
- 年份/来源：Nature Communications 2026
- 实验目的：构建统一的睡眠解码基础模型，覆盖睡眠分期、疾病分类、数据生成、睡眠纺锤波检测等任务。
- 实验方法：提出 SleepGPT，在 8,377 名受试者、86,335 小时 PSG 上预训练；结合 channel-adaptive mechanism 和统一时频融合 Transformer。
- 实验结果：在多个睡眠解码任务上取得强性能；生成能力可用于数据增强和 artifact 修复，增强小数据场景下的分期可靠性。
- 对当前课题的价值：可用于论文展望或高水平 related work。它强调 channel-adaptive 和 artifact repair，和你的 EOG 损坏鲁棒性有概念联系。

### 9. SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals

- 本地文件：`SleepFM_2024.pdf`
- 年份/来源：2024, arXiv
- 实验目的：用大规模多模态 PSG 学习睡眠通用表征，覆盖脑电、心电和呼吸信号。
- 实验方法：整理 14,000 多名受试者、100,000 多小时多模态睡眠记录；提出 leave-one-out contrastive learning。
- 实验结果：基于 SleepFM embedding 的 logistic regression 在睡眠分期上优于端到端 CNN，macro AUROC 0.88 vs 0.72，macro AUPRC 0.72 vs 0.48；在睡眠呼吸障碍检测上也优于 CNN。
- 对当前课题的价值：适合作为多模态睡眠 foundation model 背景。它说明跨模态对齐和表征学习对睡眠任务有效。

### 10. A Deep Generative Model for Five-Class Sleep Staging with Arbitrary Sensor Input

- 本地文件：`ArbitrarySensorSleepStaging_2024.pdf`
- 年份/来源：2024/2025, arXiv/JBHI
- 实验目的：实现任意传感器组合输入下的五分类睡眠分期，提高对信号缺失和不同采集配置的适应性。
- 实验方法：使用包含 1,947 条整夜记录、36 种信号的数据集，训练 score-based diffusion model；通过 Bayesian factorization 在任意传感器集合上 zero-shot 推理。
- 实验结果：单通道 EEG 达到 85.6% accuracy 和 0.791 kappa；仅用心肺相关传感器也达到 79.0% accuracy 和 0.697 kappa；非常规 EMG 组合达到 71.0% accuracy 和 0.575 kappa。
- 对当前课题的价值：它和你的方向都强调“不是固定通道输入”。你的方法更轻量、更聚焦 EOG 缺失，而它更偏通用生成式任意传感器建模。

### 11. SMIL: Multimodal Learning with Severely Missing Modality

- 本地文件：`SMIL_2021.pdf`
- 年份/来源：AAAI 2021
- 实验目的：研究训练集、测试集或二者同时存在严重模态缺失时的多模态学习问题。
- 实验方法：提出基于 Bayesian meta-learning 的 SMIL，处理高达 90% 样本模态不完整的场景。
- 实验结果：在 MM-IMDb、CMU-MOSI、avMNIST 三个 benchmark 上优于已有方法和生成式 baseline。
- 对当前课题的价值：不是睡眠论文，但适合支撑 missing modality learning 的方法背景。若后续考虑训练集中 EOG 本身不完整，可以引用它。

### 12. Deep Multimodal Learning with Missing Modality: A Survey

- 本地文件：`Missing_Modality_Survey_2024.pdf`
- 年份/来源：TMLR / arXiv 2024
- 实验目的：系统综述深度多模态学习中的缺失模态问题。
- 实验方法：总结 missing modality 场景的动机、设定、方法分类、应用、数据集和未来挑战。
- 实验结果：综述论文没有单一实验结果；主要贡献是把方法脉络整理为可引用的 taxonomy。
- 对当前课题的价值：适合写 related work 开头，用来组织 `modality dropout / imputation / representation alignment / adaptive fusion` 几类方法。

## 和当前实验的直接对应关系

- 你的强 baseline `EOG dropout p=0.5`：可对应 missing modality training / modality dropout 类思路。
- 你的 `quality sensor for EOG reliability`：可对应 imperfect modality / modality reliability / adaptive fusion。
- 你的 `EEG->EOG feature generator`：可对应 imagination / imputation / generative missing-modality recovery。
- 你的 `dynamic gating/fusion`：可对应 XSleepNet 的动态多视图融合、SalientSleepNet 的多模态注意力、CoRe-Sleep 的鲁棒融合。
- 你的跨 backbone 计划：DeepSleepNet 是当前验证 backbone，U-Sleep 和 EEGPT 是更有说服力的迁移验证对象。

## 写 related work 时的建议结构

### Automatic Sleep Staging Backbones

可放 DeepSleepNet、U-Sleep、XSleepNet、SalientSleepNet、FlexSleepTransformer、EEGPT、SleepGPT。

### Multimodal Sleep Staging

重点写 EEG/EOG/EMG 多模态对 REM、N1 等阶段的帮助，同时指出普通多模态模型容易过度依赖辅助模态。

### Missing or Corrupted Modality Robustness

核心放 CIMSleepNet、CoRe-Sleep、SMIL、Missing Modality Survey，再引出你的问题：已有工作多数处理任意缺失模态或通用多模态缺失，而你聚焦真实 PSG 中更常见、更具体的 EOG 缺失/损坏，设计质量感知、EEG->EOG 特征恢复和动态融合模块。

## 来源链接

- CIMSleepNet: https://nips.cc/virtual/2024/poster/94475
- CoRe-Sleep: https://arxiv.org/abs/2304.06485
- U-Sleep: https://www.nature.com/articles/s41746-021-00440-5
- FlexSleepTransformer: https://www.nature.com/articles/s41598-024-76197-0
- XSleepNet: https://arxiv.org/abs/2007.05492
- SalientSleepNet: https://arxiv.org/abs/2105.13864
- EEGPT: https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html
- SleepGPT: https://www.nature.com/articles/s41467-025-67970-4
- SleepFM: https://arxiv.org/abs/2405.17766
- Arbitrary Sensor Sleep Staging: https://arxiv.org/abs/2408.15253
- SMIL: https://arxiv.org/abs/2103.05677
- Missing Modality Survey: https://arxiv.org/abs/2409.07825
