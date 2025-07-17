import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnx
from pathlib import Path

from mlops_hatespeech.model import MODEL_STR

# Pick one example checkpoint
model_name = MODEL_STR
checkpoint_path = Path(__file__).resolve().parents[2] / "logs" / "run1" / "checkpoint-1110"
onnx_path = checkpoint_path.parent / "bert_tiny.onnx"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
model.eval()

dummy_text = "This is a test sentence."
inputs = tokenizer(dummy_text, return_tensors="pt")

torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    },
    opset_version=17,
)

print(f"Model exported to {onnx_path}")

onnx_model = onnx.load(onnx_path)
print(onnx.helper.printable_graph(onnx_model.graph))
print("ONNX model loaded and verified.")
