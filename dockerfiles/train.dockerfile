FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p data logs

COPY src src/
COPY reports reports/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

COPY data/processed.dvc data/processed.dvc
COPY logs/run1.dvc logs/run1.dvc
COPY .dvc .dvc
COPY .git .git

RUN pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["bash", "-c", "dvc pull data/processed.dvc && python -u src/mlops_hatespeech/train.py && dvc add logs/run1 && dvc push"]
