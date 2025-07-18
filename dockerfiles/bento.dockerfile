FROM python:3.11-slim
WORKDIR /bento
COPY src/mlops_hatespeech/bentoml_service.py .
COPY logs/run1/bert_tiny.onnx logs/run1/bert_tiny.onnx
RUN pip install onnxruntime numpy bentoml transformers
CMD ["bentoml", "serve", "bentoml_service:TextClassificationService"]
