"""
Drift Detection Script for Tweet Classifier (TinyBERT)

This script loads a reference dataset (collected from user requests) and compares it to t
he latest predictions stored in a GCS bucket by computing mean BERT embeddings for the tweet texts.
It then uses the Evidently library to detect embedding-based data drift and
uploads a detailed HTML report to the GCS bucket.

Steps:
- Load reference dataset (from Hugging Face disk format)
- Download latest predictions (as JSONs) from GCS
- Generate BERT embeddings for both datasets
- Compute mean embedding values per sample
- Run drift detection using Evidently
- Save and upload HTML report to GCS

Intended to be run periodically.
"""

import pandas as pd
from datasets import load_from_disk
from evidently import Report
from evidently.presets import DataDriftPreset
from transformers import AutoTokenizer, AutoModel
import json
import tempfile
from datetime import datetime
import torch
from google.cloud import storage
from typing import Any, List
import numpy as np

from mlops_hatespeech.model import MODEL_STR

BUCKET_NAME = "new-dvc-bucket"
PREDICTIONS_PREFIX = "predictions/"
REPORT_OUTPUT_PATH = "drift_reports/drift_report_{timestamp}.html"
MODEL_NAME = MODEL_STR

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def get_bert_embeddings(texts: List[str], tokenizer: Any, model: Any, device: str = "cpu") -> np.ndarray:
    """
    Generates mean-pooled BERT embeddings for a list of texts.

    Args:
        texts (List[str]): Input texts.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
        model (transformers.PreTrainedModel): Pretrained BERT model.
        device (str): Device to run the model on, default is "cpu".

    Returns:
        np.ndarray: 2D numpy array with shape (n_texts, hidden_size).
    """
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    attention_mask = inputs["attention_mask"]
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * input_mask_expanded, 1)
    summed_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask

    return mean_pooled.cpu().numpy()


def download_predictions_from_gcs(bucket_name: str, prefix: str) -> pd.DataFrame:
    """
    Downloads prediction JSON files from a Google Cloud Storage bucket.

    Each JSON is expected to contain a 'tweet' (under 'input_text') and a 'label' (under 'prediction').
    These are collected into a DataFrame for further processing.

    Args:
        bucket_name (str): Name of the GCS bucket.
        prefix (str): Prefix path under which prediction JSON files are stored.

    Returns:
        pd.DataFrame: DataFrame with columns 'tweet' and 'label' from all valid JSON files.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    rows = []
    for blob in blobs:
        if blob.name.endswith(".json"):
            content = blob.download_as_string()
            data = json.loads(content)
            rows.append({"tweet": str(data.get("input_text", "")), "label": data.get("prediction", "")})
    return pd.DataFrame(rows)


def upload_report_to_gcs(local_path: str, bucket_name: str, destination_path: str) -> None:
    """
    Uploads a local file (e.g. an HTML report) to a specified location in a GCS bucket.

    Args:
        local_path (str): Path to the local file to be uploaded.
        bucket_name (str): Name of the target GCS bucket.
        destination_path (str): Path (including filename) in the bucket where the file should be stored.

    Returns:
        None
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(local_path)
    print(f"Drift report uploaded to gs://{bucket_name}/{destination_path}")


def main() -> None:
    """
    Main routine for detecting embedding-based data drift on tweet inputs
    using BERT embeddings and uploading the Evidently report to GCS.
    """
    # Load reference dataset
    dataset = load_from_disk("data/processed")
    reference_df = dataset["train"].to_pandas()[["tweet", "label"]].copy()
    reference_df["tweet"] = reference_df["tweet"].astype(str)
    # Load current predictions from GCS
    current_df = download_predictions_from_gcs(BUCKET_NAME, PREDICTIONS_PREFIX)
    current_df["tweet"] = current_df["tweet"].astype(str)
    current_df["label"] = current_df["label"].astype(str).str.strip().str.lower().map({"non-hate": 0, "hate": 1})

    if current_df.empty:
        print("No predictions found in the bucket.")
        return

    reference_texts = [str(t) for t in reference_df["tweet"].tolist()]
    current_texts = [str(t) for t in current_df["tweet"].tolist()]

    # Generate embeddings
    reference_embeddings = get_bert_embeddings(reference_texts, tokenizer, model)
    current_embeddings = get_bert_embeddings(current_texts, tokenizer, model)
    reference_embed_df = pd.DataFrame(
        reference_embeddings, columns=[f"dim_{i}" for i in range(reference_embeddings.shape[1])]
    )
    reference_embed_df["label"] = reference_df["label"].values

    current_embed_df = pd.DataFrame(
        current_embeddings, columns=[f"dim_{i}" for i in range(current_embeddings.shape[1])]
    )
    current_embed_df["label"] = current_df["label"].values
    reference_embed_df["tweet"] = reference_df["tweet"].values
    current_embed_df["tweet"] = current_df["tweet"].values

    reference_embed_df["embedding_mean"] = reference_embeddings.mean(axis=1)
    current_embed_df["embedding_mean"] = current_embeddings.mean(axis=1)

    # Only pick relevant columns
    reference_embed_df = reference_embed_df[["tweet", "label", "embedding_mean"]]
    current_embed_df = current_embed_df[["tweet", "label", "embedding_mean"]]

    # Drift score > 70% to be categorized as data drift
    report = Report(metrics=[DataDriftPreset(threshold=0.7)])
    # generate report
    eval = report.run(reference_data=reference_embed_df, current_data=current_embed_df)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        eval.save_html(tmp.name)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        gcs_report_path = REPORT_OUTPUT_PATH.format(timestamp=timestamp)
        upload_report_to_gcs(tmp.name, BUCKET_NAME, gcs_report_path)


if __name__ == "__main__":
    main()
