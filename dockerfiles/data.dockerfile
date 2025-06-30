FROM python:3.11-slim
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml


WORKDIR /
RUN pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir --verbose
RUN mkdir -p data

ENTRYPOINT ["bash", "-c", "python src/mlops_hatespeech/data.py && dvc add data/processed && dvc push"]