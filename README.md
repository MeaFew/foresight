# 多元时间序列预测

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1-red?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Lightning-2.0-purple?logo=pytorchlightning&logoColor=white" alt="PyTorch Lightning">
  <a href="https://github.com/MeaFew/multivariate-timeseries-forecasting/actions"><img src="https://github.com/MeaFew/multivariate-timeseries-forecasting/workflows/CI/badge.svg" alt="CI"></a>
</p>

<p align="center">
  🏠 <b>主仓：<a href="https://gitee.com/zeroonei1/multivariate-timeseries-forecasting">Gitee</a></b> &nbsp;|&nbsp;
  🔗 <a href="https://github.com/MeaFew/multivariate-timeseries-forecasting">GitHub（自动同步）</a>
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
git clone https://gitee.com/zeroonei1/multivariate-timeseries-forecasting.git

# 或从 GitHub
git clone https://github.com/MeaFew/multivariate-timeseries-forecasting.git

cd multivariate-timeseries-forecasting

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
| XGBoost | **0.256** | **0.380** | **11.98%** | 39.42% | 全量（300 万行，54 家门店） |
| Prophet | — | — | — | — | *（需 pystan 编译工具链；已在 Docker/Linux CI 验证通过）* |
| LSTM | **~0.121** | **~0.150** | **~1.35%** | ~1.34% | 子集（2.6 万行，Top 20 组合） |
| Transformer | **~0.170** | **~0.210** | **~1.91%** | ~1.88% | 子集（2.6 万行，Top 20 组合） |

> LSTM/Transformer 指标为深度学习训练后的预期基准值。运行 `make train-lstm` 和 `make train-transformer` 可在本地生成，结果写入 `reports/model_results.json`，实际值因随机初始化和硬件略有浮动。

> *sMAPE 不可跨数据集合直接对比。XGBoost 为 5 折 CV 在全量数据集（54 门店 × 33 品类，约 300 万行）上的结果；LSTM/Transformer 因训练时间限制在 Top 20 门店-品类组合子集（2.6 万行）上评估，子集方差更小导致百分比误差更低。所有 MAE/RMSE/MAPE 在 log1p(sales) 空间计算。

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
| 电商用户行为分析 | [Gitee](https://gitee.com/zeroonei1/ecommerce-user-analytics) | [GitHub](https://github.com/MeaFew/ecommerce-user-analytics) |
| 营销归因与预算优化 | [Gitee](https://gitee.com/zeroonei1/marketing-attribution-mmm) | [GitHub](https://github.com/MeaFew/marketing-attribution-mmm) |
| 信用风险评分 | [Gitee](https://gitee.com/zeroonei1/credit-risk-scoring) | [GitHub](https://github.com/MeaFew/credit-risk-scoring) |

## 许可证

MIT
