<div align="center">

# Foresight

**多变量时间序列预测**

*LSTM · Transformer · XGBoost · 防泄漏*

<img src="https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-2.1-red?logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/Lightning-2.0-purple?logo=pytorchlightning&logoColor=white" alt="PyTorch Lightning">
<a href="https://github.com/MeaFew/foresight/actions"><img src="https://github.com/MeaFew/foresight/workflows/CI/badge.svg" alt="CI"></a>

🏠 **主仓：<a href="https://gitee.com/zeroonei1/foresight">Gitee</a>** &nbsp;|&nbsp;
🔗 <a href="https://github.com/MeaFew/foresight">GitHub（自动同步）</a>

**中文** | <a href="./README.en.md">English</a>

</div>

---

## 核心结论

> **XGBoost 最优：MAE = 0.257、RMSE = 0.381**（log1p 空间）。
> 一个诚实的负面结果——**深度学习（LSTM / Transformer）在本数据集上并未超过梯度提升基线**。

本管线在特征工程充分（40+ 维 lag / rolling / 节假日 / 油价特征）的表格类时序数据上系统对比了 XGBoost、Prophet、LSTM、Transformer。结论明确：**手工特征覆盖长程依赖后，梯度提升对结构化特征利用更高效，DL 未必优于基线**——这本身是有价值的工程发现。

| 模型 | MAE ↓ | RMSE ↓ | MAPE ↓ | sMAPE | 数据集 |
|------|-------|--------|--------|-------|--------|
| **XGBoost** | **0.257** | **0.381** | **12.02%** | 39.47% | 完整预处理训练集 + 尾部 16 天验证 |
| LSTM | 0.269 | 0.399 | 12.71% | 40.66% | 同一时间验证窗（GPU 训练） |
| Transformer | 0.282 | 0.410 | 12.76% | 40.61% | 同一时间验证窗（GPU 训练） |
| Prophet | — | — | — | — | *需 pystan 编译工具链，已在 Docker/Linux CI 验证通过* |

> LSTM 与 Transformer 均已按**三处泄漏修复**重跑，修复逻辑通过 `tests/test_pipeline.py::TestLeakagePrevention` 验证。指标以 `reports/model_results.json` 为准。

<p align="center">
  <img src="images/forecast_comparison.png" alt="预测对比图：XGBoost / LSTM / Transformer vs 真实值">
</p>

---

## 项目简介

基于 Kaggle Store Sales 数据集的端到端多元时间序列预测管线。将 XGBoost / Prophet 等传统方法与 LSTM / Transformer 等深度学习架构进行系统性对比，覆盖从数据清洗到交互式仪表板的完整流程。

## 核心亮点

- **基准模型**：XGBoost Regressor + Facebook Prophet 建立预测基线
- **深度学习**：LSTM（含嵌入层）+ Transformer（含多头自注意力 + 位置编码）
- **特征工程**：滞后特征（1/7/14/28/364 天）、滚动统计量、周期性季节编码、促销聚合
- **多指标评估**：MAE、RMSE、MAPE、sMAPE，跨模型横向比较
- **交互式交付**：Streamlit 仪表板，预测值 vs 真实值可视化

## 防泄漏说明（重要卖点）

时序建模中泄漏会系统性高估指标。本管线修正了三处：

- **油价滞后（oil_lag）按日序列计算**：早期对长表直接 `shift(1)` 会跨 (store,family) 组边界，现改为在 date-unique 帧上算 `shift(1)` 再 merge 回，保证 `oil_lag_1` 始终是前一日的油价。
- **油价缺失因果填充**：早期用 `interpolate(method="linear")` 是双向插值（用未来油价插值到验证窗），现改为仅 `ffill().bfill()`，任意一日的油价绝不来自更晚的观测。
- **DL 验证目标过滤**：`TimeSeriesDataset(min_target_date=val_start)` 只发射目标日期 ≥ val_start 的样本，验证集前 28 天训练尾部仅作窗口输入、绝不作预测目标混入验证 loss/MAE。

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

# 创建并激活 Python 3.11 虚拟环境
python -m venv .venv
# Linux / macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

# 安装依赖、项目包和开发工具
make setup
# Windows 无 GNU Make：python -m pip install -r requirements.txt
#                    python -m pip install -e ".[dev]"

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
├── src/foresight/
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
├── Makefile                       # 工作流编排
└── requirements.txt
```

## 评估口径

Kaggle 竞赛使用原始销量尺度上的 RMSLE；本项目当前产物报告 log1p(sales) 空间的 MAE、RMSE、MAPE 与 sMAPE，两者**不能直接横向比较**。因此 README 只展示由 `reports/model_results.json` 支撑、且在同一时间验证窗上得到的本地模型结果，不再展示缺少产物依据的 RMSLE 或排行榜预估值。

<details>
<summary><b>📊 详细实验说明（训练速度、改进方向）</b></summary>

### 训练速度（230 万样本 / RTX 4060）

早期单线程 DataLoader（`num_workers=0`）让 GPU 长时间空等，是 DL 训练慢的主因。现开启多进程数据加载后单 epoch 显著加速：

| 优化项 | 说明 |
|--------|------|
| `num_workers=4 + persistent_workers` | 4 进程并行 `__getitem__`，worker 只 spawn 一次并跨 epoch 复用 |
| `prefetch_factor=4` | 预取队列保持 GPU 不空载 |
| `pin_memory + non_blocking` | host→device 拷贝与计算重叠 |
| `cudnn.benchmark=True` | 固定 28 步窗口下自动选最快 kernel |
| `bf16-mixed` precision | Ada Tensor Core 加速 |
| `batch_size=1024` | 降低 kernel-launch 开销（bs=128 时 72s/epoch → bs=1024 时 38s/epoch） |

> 可通过 CLI 微调：`python -m foresight.train_lstm --num_workers 8 --batch_size 2048`。若长训练被中断，`--resume` 会从 `reports/checkpoints/` 的最新 checkpoint 续训。

### DL 未胜出的原因与改进方向

LSTM 接近 XGBoost（MAE 差 0.012），Transformer 稍逊。改进方向：DL 模型可尝试更长的训练、更大的 `d_model`，或 N-BEATS / TFT 等专用时序架构。

</details>

## 数据说明

使用 Kaggle Store Sales 数据集：
- 厄瓜多尔 54 家门店
- 33 个产品品类
- 2013–2017 年每日销售数据
- 外部变量：油价、节假日、促销信息

无需 Kaggle 账号即可本地测试：运行 `python -m foresight.generate_mock_data` 自动生成统计特征相似的合成数据集。

## 相关项目

| 项目 | Gitee（主仓） | GitHub（镜像） |
|------|---------------|-----------------|
| 电商用户行为分析 | [Gitee](https://gitee.com/zeroonei1/shoplytics) | [GitHub](https://github.com/MeaFew/shoplytics) |
| 营销归因与预算优化 | [Gitee](https://gitee.com/zeroonei1/attributor) | [GitHub](https://github.com/MeaFew/attributor) |
| 信用风险评分 | [Gitee](https://gitee.com/zeroonei1/riskscore) | [GitHub](https://github.com/MeaFew/riskscore) |
| 图神经网络反欺诈 | [Gitee](https://gitee.com/zeroonei1/graphguard) | [GitHub](https://github.com/MeaFew/graphguard) |

## 许可证

MIT
