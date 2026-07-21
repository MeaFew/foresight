FROM python:3.11-slim

LABEL org.opencontainers.image.title="Multivariate Time Series Forecasting"
LABEL org.opencontainers.image.description="LSTM + Transformer + XGBoost pipeline for Store Sales forecasting"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pinned CPU lock (compiled for Python 3.11); the torch==2.5.1+cpu wheel only
# exists on the PyTorch index.
COPY requirements-cpu.lock .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-cpu.lock

# Install the foresight package itself. The project uses a src layout, so the
# repo checkout alone is NOT importable — without this, `streamlit run
# dashboard/app.py` fails with ModuleNotFoundError: foresight.
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

COPY --chown=appuser:appuser . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
