import os
import re
import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from mlops_hatespeech.model import MODEL_STR


def find_latest_checkpoint(run_dir: str = "logs/run1") -> str:
    # Find all checkpoints and we take the highest;)
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


def compute_metrics(eval_preds):
    logits, labels = eval_preds.predictions, eval_preds.label_ids
    pred_labels = np.argmax(logits, axis=-1)
    f1 = f1_score(y_true=labels, y_pred=pred_labels, average="weighted")
    acc = accuracy_score(y_true=labels, y_pred=pred_labels)
    return {
        "f1": f1,
        "accuracy": acc,
    }


def main():
    # Load checkpoint
    checkpoint_path = find_latest_checkpoint()

    # Data
    ds = load_from_disk("data/processed")

    def is_valid(example):
        text = example["tweet"]
        return isinstance(text, str) and len(text.strip()) > 0

    ds = ds.filter(is_valid)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_STR)

    def tokenize_seqs(examples):
        return tokenizer(examples["tweet"], truncation=True, max_length=512)

    ds = ds.map(tokenize_seqs, batched=True)
    ds = ds.rename_column("label", "labels")

    # Modell laden
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

    # Evaluation
    eval_result = trainer.evaluate(eval_dataset=ds["test"])
    print("Evaluated the thing")
    print(eval_result)


if __name__ == "__main__":
    main()