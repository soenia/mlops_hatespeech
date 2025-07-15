from fastapi import FastAPI
from pydantic import BaseModel
from google.cloud import storage
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import tempfile
import os
from contextlib import asynccontextmanager

BUCKET_NAME = "mlops_hs_data"
MODEL_PATH = "logs/run1/checkpoint-370"
MODEL_NAME = "prajjwal1/bert-tiny"

model = None
tokenizer = None

app = FastAPI()


class InputText(BaseModel):
    text: str


def download_model_from_gcs(bucket_name, gcs_dir, local_dir):
    client = storage.Client(project="mlops-hs-project")
    blobs = client.list_blobs(bucket_name, prefix=gcs_dir)
    for blob in blobs:
        rel_path = os.path.relpath(blob.name, gcs_dir)
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print("Loading model from GCS...")
    tmpdir = tempfile.mkdtemp()
    local_model_path = os.path.join(tmpdir, "model")
    download_model_from_gcs(BUCKET_NAME, MODEL_PATH, local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
    print("Model loaded.")
    yield


app.router.lifespan_context = lifespan


@app.post("/predict")
async def predict(input_data: InputText):
    global model, tokenizer
    if not model or not tokenizer:
        return {"error": "Model not loaded."}

    inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()

    label_map = {0: "non-hate", 1: "hate"}
    predicted_label = label_map.get(prediction, "unknown")

    return {"input_text": input_data.text, "predicted_class": predicted_label}
