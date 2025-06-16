import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import typer
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from mlops_hatespeech.model import MODEL_STR

app = typer.Typer()

@app.command()
def train(wd : float = 1e-3, lr : float = 2e-5, epochs : int = 5, seed : int = 42) -> None:
    """Train a model."""
    ds = load_from_disk("data/processed")

    idx2lbl = {
        0: "non-hate",
        1: "hate",
    }
    lbl2idx = {v: k for k, v in idx2lbl.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_STR)

    def tokenize_seqs(examples):
        texts = examples["tweet"]
        print(f"Type of examples['tweet']: {type(texts)}")
        print(f"First example: {texts[0]}")
        return tokenizer(texts, truncation=True, max_length=512)

    def is_valid(example):
        text = example["tweet"]
        return isinstance(text, str) and len(text.strip()) > 0

    ds = ds.filter(is_valid)
    ds = ds.map(tokenize_seqs, batched=True)
    ds = ds.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_STR,
        num_labels=len(lbl2idx),
        id2label=idx2lbl,
        label2id=lbl2idx,
    )

    def compute_metrics(eval_preds):
        logits, labels = eval_preds.predictions, eval_preds.label_ids
        pred_labels = np.argmax(logits, axis=-1)

        f1 = f1_score(y_true=labels, y_pred=pred_labels, average="weighted")
        acc = accuracy_score(y_true=labels, y_pred=pred_labels)

        return {
            "f1": f1,
            "accuracy": acc
        }

    training_args = TrainingArguments(
        output_dir="./logs/run1",
        per_device_train_batch_size=32,
        per_gpu_eval_batch_size=128,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        weight_decay=wd,
        num_train_epochs=epochs,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=1,
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        load_best_model_at_end=False,
        report_to=None,
        no_cuda=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
    )
    trainer.train()
    print("Training is done.")

if __name__ == "__main__":
    app()
