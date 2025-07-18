"""
BentoML service for ONNX-based text classification (hate speech detection).

Loads a tiny BERT model in ONNX format and exposes an API for inference.
Tokenization is done with Hugging Face's Transformers, and inference is
executed using ONNX Runtime.

Note that our actual application's frontend is connected with a fastAPI version of
the code below, since our app is lightweight anyway.
"""

import bentoml
import numpy as np
from onnxruntime import InferenceSession
from transformers import AutoTokenizer
from bentoml.io import Text


@bentoml.service
class TextClassificationService:
    def __init__(self):
        """Initialize ONNX inference session and tokenizer."""
        super().__init__()
        # Note that the onnx must be there in order for this to work
        self.model = InferenceSession("logs/run1/bert_tiny.onnx")
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    @bentoml.api
    def predict(self, text: str):
        """
        Predict the class (hate / non-hate) of the input text.

        Args:
            text (str): Input sentence.

        Returns:
            str: JSON string with the input text and predicted label.
        """
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
