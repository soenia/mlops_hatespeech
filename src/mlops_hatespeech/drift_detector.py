import pandas as pd
from datasets import load_from_disk
from evidently import Report
from evidently.presets import DataDriftPreset
from transformers import AutoTokenizer, AutoModel
import torch
import os
import json
import tempfile
import pandas as pd
from datetime import datetime
import torch
from google.cloud import storage

BUCKET_NAME = "new-dvc-bucket"
PREDICTIONS_PREFIX = "predictions/"
REPORT_OUTPUT_PATH = "drift_reports/drift_report_{timestamp}.html"
MODEL_NAME = "prajjwal1/bert-tiny"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def get_bert_embeddings(texts, tokenizer, model, device="cpu"):
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


def download_predictions_from_gcs(bucket_name, prefix):
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


def upload_report_to_gcs(local_path, bucket_name, destination_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(local_path)
    print(f"Drift report uploaded to gs://{bucket_name}/{destination_path}")


def main():
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

    report = Report(metrics=[DataDriftPreset()])
    eval = report.run(reference_data=reference_embed_df, current_data=current_embed_df)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        eval.save_html(tmp.name)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        gcs_report_path = REPORT_OUTPUT_PATH.format(timestamp=timestamp)
        upload_report_to_gcs(tmp.name, BUCKET_NAME, gcs_report_path)


if __name__ == "__main__":
    main()
