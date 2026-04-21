# 下一次要做什么

## 当前检查点

- `Sleep-EDF` 固定划分 baseline 已完成。
- 当前正式 baseline 结果：
  - `val accuracy = 0.9225`
  - `val macro_f1 = 0.8162`
  - `val kappa = 0.8473`
  - `test accuracy = 0.9236`
  - `test macro_f1 = 0.8021`
  - `test kappa = 0.8464`
- 论文版 `fold_00` 已完成：
  - `test accuracy = 0.8536`
  - `test macro_f1 = 0.7906`
  - `test kappa = 0.7893`
- 服务器侧已开始批量生成和推进后续折：
  - split 文件至少生成到 `fold_13.json`
  - 后续训练存在中断恢复和资源受限情况

## 下次优先任务

1. 先确认服务器上的 `fold_14.json` 到 `fold_19.json` 是否补齐。
2. 确认接下来剩余折统一采用哪套口径：
   - 严格论文版：`stage2 = 200 epochs`
   - 资源受限版：`stage2 = 100 epochs`
3. 保持后续所有剩余折使用同一套 `stage2` epoch 设置。
4. 分批跑完剩余折，并把每折 `evaluation_test.json` 拉回本地。
5. 最后做 `20-fold` 平均结果汇总和对论文对比。

## 直接可用命令

如果服务器上缺少后面的 split 文件，先补齐：

```bash
for i in 14 15 16 17 18 19
do
  fold=$(printf "fold_%02d" $i)
  python main.py split-subjects \
    --manifest data/processed/sleep_edf_paper_fpz_cz/manifest.json \
    --output data/processed/sleep_edf_paper_fpz_cz/${fold}.json \
    --group-by participant \
    --n-folds 20 \
    --fold-index $i \
    --seed 42
done
```

如果要继续批量跑 `fold_01` 到 `fold_09`：

```bash
set -e
for i in 1 2 3 4 5 6 7 8 9
do
  fold=$(printf "fold_%02d" $i)
  python main.py train-stage1 \
    --config configs/paper_sleep_edf_fpz_cz.yaml \
    --manifest data/processed/sleep_edf_paper_fpz_cz/manifest.json \
    --split data/processed/sleep_edf_paper_fpz_cz/${fold}.json \
    --output-dir results/deepsleepnet_sleep_edf_paper_fpz_cz/${fold}/stage1

  python main.py train-stage2 \
    --config configs/paper_sleep_edf_fpz_cz.yaml \
    --manifest data/processed/sleep_edf_paper_fpz_cz/manifest.json \
    --split data/processed/sleep_edf_paper_fpz_cz/${fold}.json \
    --stage1-checkpoint results/deepsleepnet_sleep_edf_paper_fpz_cz/${fold}/stage1/best_model.pt \
    --output-dir results/deepsleepnet_sleep_edf_paper_fpz_cz/${fold}/stage2

  python main.py evaluate-stage2 \
    --config configs/paper_sleep_edf_fpz_cz.yaml \
    --manifest data/processed/sleep_edf_paper_fpz_cz/manifest.json \
    --split data/processed/sleep_edf_paper_fpz_cz/${fold}.json \
    --checkpoint results/deepsleepnet_sleep_edf_paper_fpz_cz/${fold}/stage2/best_model.pt \
    --subset test \
    --output-dir results/deepsleepnet_sleep_edf_paper_fpz_cz/${fold}/eval_test
done
```

如果服务器上的处理数据丢文件，优先重新解压：

```bash
cd /workspace/EGGSLEEP
rm -rf data/processed/sleep_edf_paper_fpz_cz
tar -xzf 'sleep_edf_paper_fpz_cz (1).tar.gz'
```

## 关键提醒

- 论文版 `k-fold` 下 `val_subjects` 为空是正常的，不是出错。
- 论文版 `stage1` 摘要里 `best_val_macro_f1 = NaN` 也是正常的，因为没有单独验证集。
- 当前这条线是按 `participant` 分组，不是按单晚记录分组。
- 当前最重要的是保持剩余折的 `stage2` epoch 设置一致，不要一半 `100` 一半 `200`。
- 手动中断时，已完整落盘的折不需要重跑，只需从中断折继续。
- 如果服务器不稳定，优先保住这些文件：
  - `best_model.pt`
  - `training_summary.json`
  - `evaluation_*.json`
  - `confusion_matrix_*.json`
