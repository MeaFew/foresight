FROM python:3.12-slim

LABEL org.opencontainers.image.title="Multivariate Time Series Forecasting"
LABEL org.opencontainers.image.description="LSTM + Transformer + XGBoost pipeline for Store Sales forecasting"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

COPY --chown=appuser:appuser . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
