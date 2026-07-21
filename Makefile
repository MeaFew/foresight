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
	pytest tests/ -v --tb=short --cov=foresight --cov-report=term-missing --cov-fail-under=20

typecheck:
	mypy src/foresight

audit:
	$(PYTHON) -m foresight.audit_consistency

verify: lint format-check typecheck test audit
	@echo "All quality gates passed"

# ── Utilities ─────────────────────────────────────────────────────
clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('.pytest_cache'), pathlib.Path('.ruff_cache'), pathlib.Path('lightning_logs')]]"
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('data/processed').glob('*.csv')]"
