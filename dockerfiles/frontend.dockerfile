FROM python:3.11-slim

# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

# Copy requirements file and source code
COPY requirements_frontend.txt /app/requirements_frontend.txt
COPY src/mlops_hatespeech/frontend.py /app/frontend.py

# Install Python dependencies using cache mount
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_frontend.txt

# Set port for Cloud Run / local testing
ENV PORT=8001
EXPOSE $PORT

# Run the FastAPI app
CMD sh -c "streamlit run frontend.py --server.port=${PORT:-8001} --server.address=0.0.0.0"
