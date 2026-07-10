.PHONY: all setup preprocess features train-baseline train-lstm train-transformer train-all evaluate dashboard verify clean

PYTHON := python

# ── One-shot pipeline ─────────────────────────────────────────────
all: preprocess features train-all evaluate

# ── Environment ───────────────────────────────────────────────────
setup:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

# ── Data pipeline ─────────────────────────────────────────────────
preprocess:
	$(PYTHON) -m foresight.preprocess

features:
	$(PYTHON) -m foresight.feature_engineering

# ── Modeling ──────────────────────────────────────────────────────
train-baseline:
	$(PYTHON) -m foresight.train_baseline

train-lstm:
	$(PYTHON) -m foresight.train_lstm

train-transformer:
	$(PYTHON) -m foresight.train_transformer

train-all: train-baseline train-lstm train-transformer

evaluate:
	$(PYTHON) -m foresight.evaluate

# ── Dashboard ─────────────────────────────────────────────────────
dashboard:
	streamlit run dashboard/app.py

# ── Quality gates ─────────────────────────────────────────────────
lint:
	ruff check src/ tests/ dashboard/

format:
	ruff format src/ tests/ dashboard/

format-check:
	ruff format --check src/ tests/ dashboard/

test:
	pytest tests/ -v --tb=short --cov=foresight --cov-report=term-missing --cov-fail-under=25

typecheck:
	mypy src/foresight

audit:
	$(PYTHON) -m foresight.audit_consistency

verify: lint format-check typecheck test audit
	@echo "All quality gates passed"

# ── Utilities ─────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "lightning_logs" -exec rm -rf {} + 2>/dev/null || true
	rm -f data/processed/*.csv 2>/dev/null || true
