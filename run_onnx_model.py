import onnxruntime as rt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch

# === Load tokenizer and prepare inputs ===
model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dummy_text = "Those fucking democrats, they want our children to fail!"
inputs = tokenizer(dummy_text, return_tensors="pt")

# === Convert PyTorch tensors to numpy for ONNX ===
inputs_onnx = {"input_ids": inputs["input_ids"].numpy(), "attention_mask": inputs["attention_mask"].numpy()}

# === Load ONNX model ===
onnx_path = "bert_tiny.onnx"
ort_session = rt.InferenceSession(onnx_path)
output_names = [o.name for o in ort_session.get_outputs()]
onnx_outputs = ort_session.run(output_names, inputs_onnx)

print("ONNX logits:", onnx_outputs[0])

# === Load original PyTorch model ===
pt_model = AutoModelForSequenceClassification.from_pretrained("logs/run1/checkpoint-370")
pt_model.eval()
with torch.no_grad():
    torch_outputs = pt_model(**inputs).logits

print("PyTorch logits:", torch_outputs.numpy())


# === Check similarity ===
def check_similarity(onnx_out, torch_out, rtol=1e-03, atol=1e-05):
    if np.allclose(onnx_out, torch_out, rtol=rtol, atol=atol):
        print("ONNX and PyTorch outputs are close enough.")
    else:
        print("ONNX and PyTorch outputs differ.")
        diff = np.abs(onnx_out - torch_out)
        print("Max absolute difference:", np.max(diff))


check_similarity(onnx_outputs[0], torch_outputs.numpy())
