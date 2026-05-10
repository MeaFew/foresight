FROM python:3.11-slim

LABEL org.opencontainers.image.title="Multivariate Time Series Forecasting"
LABEL org.opencontainers.image.description="LSTM + Transformer + XGBoost pipeline for Store Sales forecasting"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME ["/app/data", "/app/models", "/app/reports", "/app/images"]

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
