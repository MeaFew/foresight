.PHONY: all setup preprocess features train-baseline train-lstm train-transformer train-all evaluate dashboard verify clean

PYTHON := python

# ── One-shot pipeline ─────────────────────────────────────────────
all: preprocess features train-all evaluate

# ── Environment ───────────────────────────────────────────────────
setup:
	pip install -r requirements.txt

# ── Data pipeline ─────────────────────────────────────────────────
preprocess:
	$(PYTHON) scripts/preprocess.py

features:
	$(PYTHON) scripts/feature_engineering.py

# ── Modeling ──────────────────────────────────────────────────────
train-baseline:
	$(PYTHON) scripts/train_baseline.py

train-lstm:
	$(PYTHON) scripts/train_lstm.py

train-transformer:
	$(PYTHON) scripts/train_transformer.py

train-all: train-baseline train-lstm train-transformer

evaluate:
	$(PYTHON) scripts/evaluate.py

# ── Dashboard ─────────────────────────────────────────────────────
dashboard:
	streamlit run dashboard/app.py

# ── Quality gates ─────────────────────────────────────────────────
lint:
	ruff check scripts/ tests/ --ignore E501,E402

test:
	pytest tests/ -v --tb=short

verify: lint format-check test audit
	@echo "All quality gates passed"

# ── Utilities ─────────────────────────────────────────────────────
# Note: data/processed/ can accumulate >1.4GB of generated CSVs.
# Run `make clean` to free disk space.
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "lightning_logs" -exec rm -rf {} + 2>/dev/null || true
	rm -f data/processed/*.csv 2>/dev/null || true

# === Quality gates (extended) ===

format:
	ruff format scripts/ dashboard/

format-check:
	ruff format --check scripts/ dashboard/

audit:
	$(PYTHON) scripts/audit_consistency.py
