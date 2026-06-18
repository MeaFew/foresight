# 多元时间序列预测

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1-red?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Lightning-2.0-purple?logo=pytorchlightning&logoColor=white" alt="PyTorch Lightning">
  <a href="https://github.com/MeaFew/foresight/actions"><img src="https://github.com/MeaFew/foresight/workflows/CI/badge.svg" alt="CI"></a>
</p>

<p align="center">
  🏠 <b>主仓：<a href="https://gitee.com/zeroonei1/foresight">Gitee</a></b> &nbsp;|&nbsp;
  🔗 <a href="https://github.com/MeaFew/foresight">GitHub（自动同步）</a>
</p>

<p align="center">
  <b>中文</b> | <a href="./README.en.md">English</a>
</p>

## 项目简介

基于 Kaggle Store Sales 数据集的端到端多元时间序列预测管线。将 XGBoost / Prophet 等传统方法与 LSTM / Transformer 等深度学习架构进行系统性对比，覆盖从数据清洗到交互式仪表板的完整流程。

## 核心亮点

- **基准模型**：XGBoost Regressor + Facebook Prophet 建立预测基线
- **深度学习**：LSTM（含嵌入层）+ Transformer（含多头自注意力 + 位置编码）
- **特征工程**：滞后特征（1/7/14/28/364 天）、滚动统计量、周期性季节编码、促销聚合
- **多指标评估**：MAE、RMSE、MAPE、sMAPE，跨模型横向比较
- **交互式交付**：Streamlit 仪表板，预测值 vs 真实值可视化

## 架构流程

```
原始数据 (train, stores, oil, holidays, transactions)
    │
    ▼
数据预处理 ──> 日期解析、对数变换、外部数据合并
    │
    ▼
特征工程 ──> 滞后特征、滚动均值/标准差、季节编码、促销特征
    │
    ├───> XGBoost / Prophet（基线）
    ├───> LSTM + Embeddings（深度学习）
    ├───> Transformer + Positional Encoding（深度学习）
    │
    ▼
模型评估 ──> MAE、RMSE、MAPE、sMAPE、残差分析
    │
    ▼
仪表板 ──> 预测对比、误差分布、残差诊断
```

## 技术栈

| 层级 | 工具 | 说明 |
|------|------|------|
| 数据处理 | pandas, numpy | 按时间切分训练/验证集（不打乱顺序） |
| 特征工程 | pandas rolling, sklearn | 滞后/滚动特征，shift(1) 防泄漏 |
| 基线模型 | XGBoost, Prophet | 加法回归 + 树模型基准 |
| 深度学习 | PyTorch, PyTorch Lightning | LSTM + Transformer + 类别嵌入 |
| 模型评估 | sklearn metrics | MAE、RMSE、MAPE、sMAPE |
| 交付 | Streamlit | 多模型预测对比仪表板 |
| 质量保证 | pytest, ruff, GitHub Actions | CI 端到端验证 |

## 快速开始

```bash
# 从 Gitee 克隆（国内推荐，速度更快）
git clone https://gitee.com/zeroonei1/foresight.git

# 或从 GitHub
git clone https://github.com/MeaFew/foresight.git

cd foresight

# 下载真实数据集（GitHub Releases，约 21MB）
bash download_data.sh

# 运行完整管线
python run_all.py

# 或分步执行
make preprocess         # 数据预处理
make features           # 特征工程
make train-baseline     # XGBoost + Prophet
make train-lstm         # LSTM 模型
make train-transformer  # Transformer 模型
make evaluate           # 模型评估

# 启动仪表板
make dashboard

# 质量检查
make verify
```

## 项目结构

```
.
├── scripts/
│   ├── generate_mock_data.py     # 合成零售销售数据
│   ├── preprocess.py              # 日期解析、对数变换、外部数据合并
│   ├── feature_engineering.py     # 滞后特征、滚动统计、季节编码
│   ├── train_baseline.py          # XGBoost + Prophet
│   ├── train_lstm.py              # LSTM（PyTorch Lightning）
│   ├── train_transformer.py       # Transformer（含位置编码）
│   ├── evaluate.py                # 模型对比 & 残差分析
│   ├── predict.py                 # 模型加载与推理
│   ├── metrics.py                 # MAE/RMSE/MAPE/sMAPE, TimeSeriesDataset
│   ├── metrics_utils.py           # 纯 numpy 的 mape/smape（torch-free，便于轻量测试）
│   └── audit_consistency.py       # README 声明 vs 实际输出一致性校验
├── dashboard/
│   └── app.py                     # Streamlit 预测对比仪表板
├── tests/
│   ├── test_pipeline.py           # 单元 + 集成测试
│   └── test_metrics.py            # mape/smape 数值契约 + TimeSeriesDataset 一致性测试
├── config.py                      # 集中式路径与超参数配置
├── Makefile                       # 工作流编排
└── requirements.txt
```

## 模型对比

### 基准参照

基于 [Kaggle Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)（评估指标：RMSLE，越低越好）。

| 参照 | RMSLE | 说明 |
|------|-------|------|
| Kaggle 入门基线（朴素法） | ~0.90–1.20 | 历史均值 / 朴素预测 |
| 竞赛中位数 | ~0.60–0.80 | 基础滞后特征 + XGBoost |
| 竞赛 Top 10% | ~0.45–0.50 | 复杂特征工程 |
| 竞赛 Top 1% | ~0.35–0.40 | 细粒度外部数据利用 |
| **本方案（XGBoost CV）** | **~0.24** | 对数变换后 5 折交叉验证 |

> 注：对数变换前后的 RMSLE 不可直接对比。Kaggle 使用原始尺度 RMSLE，本地验证使用对数尺度的 MAE/MAPE 以保证训练稳定性。

### 实验结果

| 模型 | MAE | RMSE | MAPE | sMAPE* | 数据集 |
|------|-----|------|------|--------|--------|
| XGBoost | **0.258** | **0.382** | **12.03%** | 39.48% | 全量（230 万行训练 + 尾部 16 天验证） |
| Prophet | — | — | — | — | *（需 pystan 编译工具链；已在 Docker/Linux CI 验证通过）* |
| LSTM | 0.265 | 0.398 | 12.30% | 39.67% | 全量（同上，GPU 训练） |
| Transformer | 0.284 | 0.414 | 12.80% | 40.12% | 全量（同上，GPU 训练） |

> 以上均为在 **log1p(sales) 空间**评估的真实结果（非预期值）。LSTM/Transformer 在 RTX 4060 上用 PyTorch Lightning 训练（bf16 混合精度，batch=1024，early stopping），指标写入 `reports/model_results.json`。

> **诚实的结论**：在特征工程充分（lag/rolling/节假日/油价等 40+ 维特征）的表格类时序数据上，**XGBoost 略优于 DL 模型**。这是符合预期的——梯度提升对结构化特征利用更高效，而 DL 的优势（自动特征学习、长程依赖）在本数据集已被手工特征覆盖。LSTM 接近 XGBoost（MAE 差 0.007），Transformer 稍逊。改进方向：DL 模型可尝试更长的训练、更大的 d_model、或 N-BEATS/TFT 等专用时序架构。

## 数据说明

使用 Kaggle Store Sales 数据集：
- 厄瓜多尔约 1,200 家门店
- 33 个产品品类
- 2013–2017 年每日销售数据
- 外部变量：油价、节假日、促销信息

无需 Kaggle 账号即可本地测试：运行 `python scripts/generate_mock_data.py` 自动生成统计特征相似的合成数据集。

## 相关项目

| 项目 | Gitee（主仓） | GitHub（镜像） |
|------|---------------|-----------------|
| 电商用户行为分析 | [Gitee](https://gitee.com/zeroonei1/shoplytics) | [GitHub](https://github.com/MeaFew/shoplytics) |
| 营销归因与预算优化 | [Gitee](https://gitee.com/zeroonei1/attributor) | [GitHub](https://github.com/MeaFew/attributor) |
| 信用风险评分 | [Gitee](https://gitee.com/zeroonei1/riskscore) | [GitHub](https://github.com/MeaFew/riskscore) |
| 图神经网络反欺诈 | [Gitee](https://gitee.com/zeroonei1/graphguard) | [GitHub](https://github.com/MeaFew/graphguard) |

## 许可证

MIT
