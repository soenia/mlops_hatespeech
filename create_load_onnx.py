import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnx
import os

# === CONFIG ===
checkpoint_path = "logs/run1/checkpoint-670"
model_name = "prajjwal1/bert-tiny"
onnx_path = "bert_tiny.onnx"

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
model.eval()

# === Prepare dummy input ===
dummy_text = "This is a test sentence."
inputs = tokenizer(dummy_text, return_tensors="pt")

# === Export to ONNX ===
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

# === Verify the ONNX model ===
onnx_model = onnx.load(onnx_path)
print(onnx.helper.printable_graph(onnx_model.graph))
print("ONNX model loaded and verified.")
