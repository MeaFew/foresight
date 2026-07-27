# Seasonal-Naive Baseline & Scaled-Error Evaluation

**日期**: 2026-07-27 · **数据来源**: 真实 Kaggle Store Sales（GitHub Release `v1.0-data`，`data/raw/train.csv` 3,000,888 行，54 店 × 33 品类，2013-01-01 ~ 2017-08-15）

## 动机

此前评估只报告 MAE/RMSE/MAPE/sMAPE：XGBoost MAPE 12% 本身无法回答「是否比直接抄上周同期强」。本报告加入 **seasonal-naive（t-7）朴素基线** 与 **MASE / RMSSE** 标度误差，使所有模型的误差都相对于朴素基准可归一化比较。

## 协议

- **Holdout**：与所有模型一致的时间序列尾部切分——最后 16 天（2017-07-31 ~ 2017-08-15，28,512 行），无 shuffle、无交叉验证（`VAL_DAYS=16`，`time_train_val_split`）。
- **Seasonal-naive 定义**：每条 (store, family) 序列上 ŷ(t) = y(t−7)（log1p 空间）。与其他模型的评估协议一致——XGBoost 的 lag 特征（shift(1) 体系）在验证窗内同样以真实历史值为条件，因此属于同一「以实际近期历史为条件」的滚动评估口径。
- **MASE**（Hyndman & Koehler 2006）：`MASE = 模型 holdout MAE / 训练段内 seasonal-naive 平均绝对误差`。分母在全部训练序列上按组内 diff(7) 池化（pooled scale = **0.3647**，log1p 空间）。MASE < 1 表示优于朴素基线。LSTM/Transformer 的 MASE 由同一 scale 从其聚合 MAE 推导（逐行预测未持久化）。
- **RMSSE**（M5）：RMSE 版，分母为训练段内 diff(7) 的均方根。

## 结果（真实数据，log1p 空间）

| Model | MAE ↓ | RMSE ↓ | MAPE ↓ | sMAPE | MASE ↓ | RMSSE ↓ |
|-------|------:|-------:|-------:|------:|-------:|--------:|
| Seasonal Naive (t-7) | 0.3614 | 0.5694 | 17.61% | 23.31% | 0.9909 | 0.7790 |
| **XGBoost** | **0.2561** | **0.3804** | **11.98%** | 39.46% | **0.7022** | **0.5204** |
| LSTM | 0.2692 | 0.3994 | 12.71% | 40.66% | 0.7382 | 0.5463* |
| Transformer | 0.2824 | 0.4096 | 12.76% | 40.61% | 0.7744 | 0.5606* |

\* LSTM/Transformer 的 RMSSE 由聚合 RMSE / pooled 分母推导。Prophet 在 Windows 上不可用（需 cmdstan 工具链），CI/Docker 中评估。

## 结论

1. **所有模型都显著优于朴素基线**：XGBoost 的 MASE = 0.702，即把「抄上周」的标度误差降低了约 29%；MAE 绝对值从 0.361 → 0.256（−29%），MAPE 从 17.6% → 12.0%。此前的「MAPE 12%」现在有了参照系。
2. **模型间排序不变**：XGBoost < LSTM < Transformer（误差升序），与修复泄漏后的结论一致——DL 未反超梯度提升。
3. **sMAPE 反向现象的说明**：naive 的 sMAPE（23.31%）低于所有模型。log1p 空间近零值密集，naive 直接复制真实值使分母 |y|+|ŷ| 分布与真实值一致，而模型在近零区间的平滑预测会推高对称百分比误差。sMAPE 在该设定下不适合作为首要指标，MASE/MAE 更可靠。

## 复现

```bash
bash download_data.sh                       # 真实数据（GitHub Release）
python -m foresight.preprocess              # 3.0M 行清洗
python -m foresight.feature_engineering     # 2.35M 行特征
python -m foresight.train_baseline          # seasonal-naive + XGBoost（Prophet 在 win32 跳过）
python -m foresight.evaluate                # 打印含 MASE 的对比表
```

测试：`tests/test_seasonal_naive.py`（19 例：scale/MASE/RMSSE 数值契约、组边界不交叉、周期性序列零误差、JSON 可序列化、与手工 t-7 参考值一致）。
