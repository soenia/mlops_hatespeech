"""
Script for evaluating a fine-tuned hate speech classification model.

- Finds the latest training checkpoint.
- Loads and preprocesses the dataset.
- Runs evaluation on the test set.
- Computes accuracy and weighted F1 score.
- Saves evaluation results as JSON.
- Uploads the results JSON to a Google Cloud Storage bucket.
"""

import os
import re
import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import json
from google.cloud import storage
from typing import Dict, Any

from mlops_hatespeech.model import MODEL_STR

BUCKET_NAME = "new-dvc-bucket"


def find_latest_checkpoint(run_dir: str = "logs/run1") -> str:
    """
    Finds the latest checkpoint folder in the given directory based on the highest checkpoint number.

    Args:
        run_dir (str): Path to the directory containing checkpoint folders.

    Returns:
        str: Path to the latest checkpoint folder.

    Raises:
        FileNotFoundError: If no checkpoints are found.
    """
    checkpoints = []
    pattern = re.compile(r"^checkpoint-(\d+)$")
    for name in os.listdir(run_dir):
        match = pattern.match(name)
        if match:
            checkpoints.append((int(match.group(1)), name))

    if not checkpoints:
        raise FileNotFoundError(f"No training checkpoints found.")

    # Pick highest
    latest_checkpoint = max(checkpoints, key=lambda x: x[0])[1]
    return os.path.join(run_dir, latest_checkpoint)


def compute_metrics(eval_preds: Any) -> Dict[str, float]:
    """
    Computes accuracy and weighted F1 score from model predictions and true labels.

    Args:
        eval_preds: An object with 'predictions' (logits) and 'label_ids' (true labels).

    Returns:
        Dict[str, float]: Dictionary with 'f1' and 'accuracy' scores.
    """
    logits, labels = eval_preds.predictions, eval_preds.label_ids
    pred_labels = np.argmax(logits, axis=-1)
    f1 = f1_score(y_true=labels, y_pred=pred_labels, average="weighted")
    acc = accuracy_score(y_true=labels, y_pred=pred_labels)
    return {
        "f1": f1,
        "accuracy": acc,
    }


def main():
    """
    Main function: Loads the latest checkpoint, prepares the dataset,
    runs evaluation, and uploads results as a JSON file to a GCS bucket.
    """
    # Load highest checkpoint
    checkpoint_path = find_latest_checkpoint()

    # Load data
    ds = load_from_disk("data/processed")

    def is_valid(example):
        text = example["tweet"]
        return isinstance(text, str) and len(text.strip()) > 0

    ds = ds.filter(is_valid)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_STR)

    # Filter only valid examples where 'tweet' is a non-empty string
    def tokenize_seqs(examples):
        return tokenizer(examples["tweet"], truncation=True, max_length=512)

    ds = ds.map(tokenize_seqs, batched=True)
    ds = ds.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

    training_args = TrainingArguments(
        output_dir="./logs/eval",
        per_device_eval_batch_size=128,
        no_cuda=True,
        report_to=None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # Run evaluation on the test dataset
    eval_result = trainer.evaluate(eval_dataset=ds["test"])
    print("Evaluated the thing")
    print(eval_result)

    # Save evaluation results as JSON
    output_path = "logs/eval/gen_perf.json"
    with open(output_path, "w") as f:
        json.dump(eval_result, f, indent=2)

    # Upload results JSON file to GCS bucket
    client = storage.Client(project="mlops-hs-project")
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob("logs/eval/gen_perf.json")
    blob.upload_from_filename(output_path)

    print("Uploaded results to Bucket.")


if __name__ == "__main__":
    main()
