"""
Export a fine-tuned Hugging Face Transformer model (BERT-tiny) to ONNX format.
Note that this could even be put into a container since it is so lightweight:)

This script loads a trained model checkpoint, converts it to the ONNX format,
and prints the model graph for inspection. The resulting ONNX file can be used
for efficient inference in environments such as ONNX Runtime or TensorRT.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnx
from pathlib import Path

from mlops_hatespeech.model import MODEL_STR

# Pick one example checkpoint
model_name = MODEL_STR
checkpoint_path = Path(__file__).resolve().parents[2] / "logs" / "run1" / "checkpoint-1110"
onnx_path = checkpoint_path.parent / "bert_tiny.onnx"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
model.eval()

# Prepare dummy input
dummy_text = "This is a test sentence."
inputs = tokenizer(dummy_text, return_tensors="pt")

# Export to ONNX
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

# Load and inspect ONNX model
onnx_model = onnx.load(onnx_path)
print(onnx.helper.printable_graph(onnx_model.graph))
print("ONNX model loaded and verified.")
