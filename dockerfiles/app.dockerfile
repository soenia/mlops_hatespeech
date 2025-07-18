FROM python:3.11-slim

# Set default port for local testing
ENV PORT=8080

# Create working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    transformers \
    google-cloud-storage \
    prometheus-client \
    onnxruntime \
    numpy

# Copy source code
COPY src/mlops_hatespeech/app.py /app/app.py
COPY logs/run1/bert_tiny.onnx logs/run1/bert_tiny.onnx

# Expose the port for Cloud Run
EXPOSE $PORT

# Command to run the app
CMD exec uvicorn app:app --host 0.0.0.0 --port $PORT
