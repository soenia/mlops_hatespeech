from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as f

from mlops_hatespeech.evaluate import find_latest_checkpoint

# Import model from logs/run1
path = find_latest_checkpoint("logs/run1")

model = AutoModelForSequenceClassification.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)

app = FastAPI(title="Hate Speech Detection API")


class TextInput(BaseModel):
    text: str


@app.post("/predict")
def predict(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)

    outputs = model(**inputs)

    logits = outputs.logits
    probs = f.softmax(logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)
    label = model.config.id2label[predicted_class.item()]

    return {
        "label": label,
        "confidence": round(confidence.item(), 4),
    }
