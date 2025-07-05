import torch
from datasets import Dataset, DatasetDict
from mlops_hatespeech.model import MODEL_STR
from mlops_hatespeech.train import train, train_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def test_model_instantiation():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_STR,
        num_labels=2,
        id2label={0: "non-hate", 1: "hate"},
        label2id={"non-hate": 0, "hate": 1},
    )
    assert isinstance(model, torch.nn.Module)
    assert model.config.num_labels == 2


def test_train_function_call():
    train(lr=2e-5, epochs=1, seed=42)
