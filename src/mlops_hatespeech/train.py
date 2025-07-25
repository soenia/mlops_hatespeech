"""
Train script for hate speech classification using a transformer model (e.g. BERT).
Includes preprocessing, training, evaluation, ROC plotting, and artifact logging with Weights & Biases.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
import wandb
import cProfile
import pstats
import io
from datasets import load_from_disk
from hydra import compose, initialize
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, RocCurveDisplay
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from mlops_hatespeech.model import MODEL_STR
from mlops_hatespeech.logger import logger

app = typer.Typer()


def get_config(overrides: Optional[List[str]]) -> DictConfig:
    """Get the configuration from Hydra."""
    with initialize(config_path="../..", job_name="train_app", version_base="1.1"):
        return compose(config_name="config", overrides=overrides or [])


def train_model(cfg: DictConfig) -> Trainer:
    """
    Load configuration using Hydra with optional overrides.

    Args:
        overrides (Optional[List[str]]): List of override strings

    Returns:
        DictConfig: Composed configuration object.
    """
    logger.info(f"Loading dataset from: {cfg.data_path}")
    ds = load_from_disk(cfg.data_path)

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

    preds_output = trainer.predict(ds["validation"])
    preds_probs = preds_output.predictions
    labels = preds_output.label_ids

    RocCurveDisplay.from_predictions(
        labels,
        preds_probs[:, 1],
        name="ROC Curve",
    )

    wandb.log({"roc_curve": wandb.Image(plt.gcf())})
    plt.close()

    metrics = trainer.evaluate()

    torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact(
        name="mlops_hatespeech_model",
        type="model",
        description="A model trained to detect hate speech in tweets.",
        metadata=metrics,
    )
    artifact.add_file("model.pth")
    wandb.log_artifact(artifact)

    return trainer


@app.command()
def train(
    lr: Optional[float] = None, wd: Optional[float] = None, epochs: Optional[int] = None, seed: Optional[int] = None
) -> None:
    """
    Entry point for training. Optionally override key hyperparameters.

    Args:
        lr (Optional[float]): Learning rate.
        wd (Optional[float]): Weight decay.
        epochs (Optional[int]): Number of training epochs.
        seed (Optional[int]): Random seed.
    """
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

    # wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)

    wandb.init(
        project="mlops_hatespeech",
        config={
            "learning rate": cfg.hyperparameters.lr,
            "weight decay": cfg.hyperparameters.wd,
            "epochs": cfg.hyperparameters.epochs,
            "model": MODEL_STR,
        },
    )

    profiler = cProfile.Profile()
    profiler.enable()

    # Run the actual training
    trainer = train_model(cfg)

    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumtime")
    ps.print_stats()

    with open("reports/logs/train_profile.txt", "w") as f:
        f.write(s.getvalue())

    logger.info("Training is done.")

    wandb.finish()


if __name__ == "__main__":
    app()
