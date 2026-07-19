# Contributing Guide

感谢你对本项目的兴趣. 本指南面向希望本地运行、调试或扩展该多元时序预测项目的开发者.

## 环境准备

```bash
# 1. 克隆仓库
git clone https://github.com/MeaFew/foresight.git
cd foresight

# 2. 创建虚拟环境 (推荐 Python 3.12)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

## 数据准备

项目使用 Kaggle Store Sales 数据集. 请运行下载脚本获取数据集:

```bash
bash download_data.sh
```

## 本地工作流

```bash
# 1. 数据预处理
make preprocess

# 2. 特征工程
make features

# 3. 训练 Baseline (XGBoost)
make train-baseline

# 4. 训练 LSTM
make train-lstm

# 5. 训练 Transformer
make train-transformer

# 6. 模型评估
make evaluate

# 7. 启动看板
make dashboard
```

## 代码规范

提交前请确保通过以下检查:

```bash
# Python lint
ruff check src/ tests/ dashboard/

# 单元测试
pytest tests/ -v
```

## 提交规范

- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档更新
- `refactor:` 重构
- `ci:` 持续集成相关
- `test:` 测试相关

## 扩展建议

- 新增模型: 放在 `src/foresight/` 并按 `train_{model}.py` 命名
- 新增特征: 在 `src/foresight/feature_engineering.py` 中扩展
- 新增评估指标: 在 `src/foresight/metrics_utils.py` 中添加
