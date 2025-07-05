import os
import random
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import typer
from datasets import load_from_disk
from hydra import compose, initialize
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from mlops_hatespeech.model import MODEL_STR

app = typer.Typer()


def get_config(overrides: Optional[List[str]]) -> DictConfig:
    """Get the configuration from Hydra."""
    with initialize(config_path="../..", job_name="train_app", version_base="1.1"):
        return compose(config_name="config", overrides=overrides or [])


def train_model(cfg: DictConfig) -> Trainer:
    ds = load_from_disk("data/processed")

    idx2lbl = {
        0: "non-hate",
        1: "hate",
    }
    lbl2idx = {v: k for k, v in idx2lbl.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_STR)

    def tokenize_seqs(examples):
        texts = examples["tweet"]
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

        return {"f1": f1, "accuracy": acc}

    training_args = TrainingArguments(
        output_dir="./logs/run1",
        per_device_train_batch_size=cfg.hyperparameters.per_device_train_batch_size,
        per_gpu_eval_batch_size=cfg.hyperparameters.per_gpu_eval_batch_size,
        gradient_accumulation_steps=cfg.hyperparameters.gradient_accumulation_steps,
        learning_rate=cfg.hyperparameters.lr,
        weight_decay=cfg.hyperparameters.wd,
        num_train_epochs=cfg.hyperparameters.epochs,
        logging_strategy=cfg.hyperparameters.logging_strategy,
        logging_steps=cfg.hyperparameters.logging_steps,
        save_strategy=cfg.hyperparameters.save_strategy,
        eval_strategy=cfg.hyperparameters.eval_strategy,
        eval_steps=cfg.hyperparameters.eval_steps,
        save_total_limit=cfg.hyperparameters.save_total_limit,
        seed=cfg.hyperparameters.seed,
        data_seed=cfg.hyperparameters.seed,
        dataloader_num_workers=cfg.hyperparameters.dataloader_num_workers,
        load_best_model_at_end=cfg.hyperparameters.load_best_model_at_end,
        report_to=cfg.hyperparameters.report_to,
        use_cpu=cfg.hyperparameters.use_cpu,
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
    return trainer


@app.command()
def train(
    lr: Optional[float] = None,
    wd: Optional[float] = None,
    epochs: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    """Train a model."""

    overrides = []
    if lr is not None:
        overrides.append(f"hyperparameters.lr={lr}")
    if wd is not None:
        overrides.append(f"hyperparameters.wd={wd}")
    if epochs is not None:
        overrides.append(f"hyperparameters.epochs={epochs}")
    if seed is not None:
        overrides.append(f"hyperparameters.seed={seed}")

    cfg = get_config(overrides)
    trainer = train_model(cfg)
    print("Training is done.")


if __name__ == "__main__":
    app()
