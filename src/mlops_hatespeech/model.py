"""
Defines the base model string used throughout the project.

This identifier is passed to Hugging Face Transformers to load the pretrained BERT model.
In this case, we are using a small and efficient version: `prajjwal1/bert-tiny`,
which is suitable for fast inference and experimentation.
"""

# Hugging Face model identifier for the tiny BERT model
MODEL_STR = "prajjwal1/bert-tiny"
