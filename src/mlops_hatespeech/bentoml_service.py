import bentoml
import numpy as np
from onnxruntime import InferenceSession
from transformers import AutoTokenizer
from bentoml.io import Text


@bentoml.service
class TextClassificationService:
    def __init__(self):
        super().__init__()
        self.model = InferenceSession("bert_tiny.onnx")
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    @bentoml.api
    def predict(self, text: str):
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="np", padding=True, truncation=True)

        # Only allow model-relevant inputs
        allowed_inputs = {input.name for input in self.model.get_inputs()}
        ort_inputs = {k: v for k, v in inputs.items() if k in allowed_inputs}
        ort_inputs = {
            k: v.astype(np.int64) if v.dtype == np.int64 else v.astype(np.float32) for k, v in ort_inputs.items()
        }

        # Run ONNX model
        outputs = self.model.run(None, ort_inputs)
        logits = outputs[0]

        # Predict class
        prediction = int(np.argmax(logits, axis=-1)[0])

        # Map prediction to label
        label_map = {0: "non-hate", 1: "hate"}
        predicted_label = label_map.get(prediction, "unknown")

        # Return result
        return {"input_text": text, "predicted_class": predicted_label}
