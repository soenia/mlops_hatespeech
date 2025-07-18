from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from google.cloud import storage
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
from datetime import datetime
from contextlib import asynccontextmanager
from prometheus_client import Counter, make_asgi_app
from fastapi import HTTPException
import numpy as np

import json
from datetime import datetime

BUCKET_NAME = "new-dvc-bucket"
MODEL_PATH = "logs/run1/bert_tiny.onnx"
MODEL_NAME = "prajjwal1/bert-tiny"

model = None
tokenizer = None

error_counter = Counter("prediction_error", "Number of prediction errors")

app = FastAPI()
app.mount("/metrics", make_asgi_app())


class InputText(BaseModel):
    text: str


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
    print("Loading tokenizer and ONNX model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    try:
        model = InferenceSession(MODEL_PATH)
    except Exception as e:
        print(f"Local model load failed: {e}")
        tmp_model_path = "/tmp/bert_tiny.onnx"
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob("logs/run1/bert_tiny.onnx")

        blob.download_to_filename(tmp_model_path)
        print("Loaded from bucket")
        model = InferenceSession(tmp_model_path)

    print("Model and tokenizer loaded.")
    yield


app.router.lifespan_context = lifespan


@app.post("/predict")
async def predict(input_data: InputText, background_tasks: BackgroundTasks):
    global model, tokenizer
    if model is None or tokenizer is None:
        error_counter.inc()
        raise HTTPException(status_code=500, detail="Model or tokenizer not loaded.")

    try:
        # Tokenize input (return numpy arrays!)
        encoded = tokenizer(input_data.text, return_tensors="np", truncation=True, padding=True)
        ort_inputs = {
            k: v.astype(np.int64) if v.dtype == np.int64 else v.astype(np.float32)
            for k, v in encoded.items()
            if k in {inp.name for inp in model.get_inputs()}
        }

        # ONNX model inference
        outputs = model.run(None, ort_inputs)
        logits = outputs[0]
        prediction = int(np.argmax(logits, axis=-1)[0])

        label_map = {0: "non-hate", 1: "hate"}
        predicted_label = label_map.get(prediction, "unknown")

        now = datetime.utcnow().isoformat()
        background_tasks.add_task(save_prediction_to_gcp, now, input_data.text, predicted_label)

        return {
            "input_text": input_data.text,
            "predicted_class": predicted_label,
        }

    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e)) from e
