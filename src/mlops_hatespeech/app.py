from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from google.cloud import storage
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import tempfile
import os
import csv
from datetime import datetime
from contextlib import asynccontextmanager
from prometheus_client import Counter, make_asgi_app
from fastapi import HTTPException

import json
from datetime import datetime

BUCKET_NAME = "new-dvc-bucket"
MODEL_PATH = "logs/run1/checkpoint-670"
MODEL_NAME = "prajjwal1/bert-tiny"

model = None
tokenizer = None

error_counter = Counter("prediction_error", "Number of prediction errors")

app = FastAPI()
app.mount("/metrics", make_asgi_app())


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


def save_prediction_to_gcp(timestamp: str, input_text: str, prediction: str):
    """Save the prediction results (timestamp, text, prediction) as JSON to GCP bucket."""
    client = storage.Client(project="mlops-hs-project")
    bucket = client.bucket(BUCKET_NAME)

    data = {
        "timestamp": timestamp,
        "input_text": input_text,
        "prediction": prediction,
    }

    filename = f"predictions/prediction_{timestamp}.json"
    blob = bucket.blob(filename)
    blob.upload_from_string(json.dumps(data))

    print(f"Prediction saved to GCP bucket: {filename}")


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
async def predict(input_data: InputText, background_tasks: BackgroundTasks):
    global model, tokenizer
    if not model or not tokenizer:
        error_counter.inc()
        raise HTTPException(status_code=500, detail="Model or tokenizer not loaded.")

    try:
        inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction_idx = torch.argmax(outputs.logits, dim=-1).item()

        label_map = {0: "non-hate", 1: "hate"}
        predicted_label = label_map.get(prediction_idx, "unknown")

        now = datetime.utcnow().isoformat()
        background_tasks.add_task(save_prediction_to_gcp, now, input_data.text, predicted_label)

        return {"input_text": input_data.text, "predicted_class": predicted_label}

    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e)) from e
